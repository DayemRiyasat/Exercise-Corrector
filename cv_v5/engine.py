# cv_v5/engine.py - Server-side computer vision for all five exercises
# Location: project_root/cv_v5/engine.py
#
# Wraps ec_pipeline.py, the Exercise-Corrector v6 pipeline (YOLO11 pose ->
# rep counter -> rule engine + per-rep BiGRU -> fusion -> per-rep verdict),
# for the three ways the web app feeds it. Jump rope and plank have no reps:
# they are judged the same way in fixed 2 second windows instead.
#
#   analyze_video  an uploaded file, end to end, exactly as evaluated on
#                  Kaggle. Runs as a background job so the page can show
#                  progress.
#   LiveSession    webcam frames arriving one at a time. Frames are put on
#                  the pipeline's 16 fps clock, and every rep is judged by
#                  the same code the moment the rep counter closes it.
#   analyze_image  one photo. Only the rule checks that make sense without
#                  movement can run, because the BiGRU and the depth rules
#                  judge a whole rep. Jump rope cannot be judged at all.
#
# ec_pipeline.py is copied unchanged from the bundle, so this file only does
# plumbing. torch and ultralytics are imported lazily, which keeps the rest
# of the site independent of them.

import json
import math
import os
import shutil
import tempfile
import threading
import time
import uuid

import numpy as np

BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
V5_EXERCISES = ('squat', 'pushup', 'lunge', 'jump_rope', 'plank')

# Pose model for live webcam frames. Defaults to the bundle's own model
# (yolo11x-pose, what the BiGRU was trained on, ~170 ms a frame on a laptop
# CPU). A smaller file such as yolo11s-pose.pt is several times faster but
# was not part of the evaluation.
LIVE_POSE_WEIGHTS = os.environ.get('EC_LIVE_POSE_WEIGHTS', '')
MAX_VIDEO_SECONDS = float(os.environ.get('EC_MAX_VIDEO_SECONDS', '180'))

# Metrics measured across a rep's range of motion. A photo is a single
# instant, so the rules built on them cannot judge it.
MOTION_METRICS = ('hip_drop', 'shoulder_drop', 'elbow_max',
                  'ankle_lift', 'hip_bounce', 'hip_sway', 'wrist_motion')

_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """The shared engine, built on first use (loads YOLO and the BiGRUs)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = Engine()
        return _engine


def _num(v, nd=4):
    """JSON-safe number: numpy scalars to float, NaN / inf to None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, nd) if math.isfinite(f) else None


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    return obj


class Engine:
    """Loads the bundle once and serves every request."""

    def __init__(self):
        import torch
        from cv_v5 import ec_pipeline as E

        self.E = E
        self.device = os.environ.get('EC_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
        # The pose weights are not in git. Fetch them into this folder once,
        # so they are not re-downloaded into whatever the server's working
        # directory happens to be.
        weights = os.path.join(BUNDLE_DIR, load_pipeline_config()['pose_weights'])
        if not os.path.exists(weights):
            try:
                from ultralytics.utils.downloads import attempt_download_asset
                attempt_download_asset(weights)
            except Exception:                         # noqa: BLE001
                pass    # Pipeline falls back to ultralytics' own download
        # Pipeline reads pipeline_config.json + calibration_v51.json, loads the
        # five BiGRUs and YOLO11x-pose. The 3D CNN is off in v5.1.
        self.pipe = E.Pipeline(BUNDLE_DIR, device=self.device, use_cnn=False)
        self.imgsz = self.pipe.cfg.get('pose_imgsz', 640)
        self.live_yolo = self.pipe.yolo
        if LIVE_POSE_WEIGHTS:
            from ultralytics import YOLO
            local = os.path.join(BUNDLE_DIR, LIVE_POSE_WEIGHTS)
            self.live_yolo = YOLO(local if os.path.exists(local) else LIVE_POSE_WEIGHTS)
        # YOLO and the GRUs are shared between request threads.
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # Shared pieces
    # ------------------------------------------------------------------
    def weights_for(self, ex):
        """Fusion weights exactly as Pipeline.__call__ builds them."""
        w = dict(self.pipe.weights)
        if self.pipe.level1:
            w['rule_by_fault'] = self.pipe.rule_by_fault.get(ex, {})
        return w

    def judge_windows(self, xy, cf, ex, windows, counted=True):
        """Judge the given (start, low, end) windows of a cleaned clip.
        Same steps as ec_pipeline.analyze_reps, which picks its own windows."""
        E = self.E
        faults = E.FAULTS[ex]
        gru = self.pipe.gru.get(ex)
        pad = 0 if ex in E.WINDOW_EX else 2
        xr, cr = zip(*[E.rep_tensor(xy, cf, r, pad=pad) for r in windows])
        with self.lock:
            gc, gf = E.gru_probs(gru, np.stack(xr), np.stack(cr), self.device) if gru is not None else (None, None)
        w = self.weights_for(ex)
        out = []
        for i, r in enumerate(windows):
            m = E.rep_metrics(xy, cf, r, ex)
            rp = E.rule_probs(m, ex, self.pipe.rule_params)
            gp = {f: float(gf[i, k]) for k, f in enumerate(faults)} if gf is not None else None
            fu = E.fuse(rp, gp, None, ex, w)
            v = E.rep_verdict(fu, ex, self.pipe.thresholds, None if gc is None else float(gc[i]),
                              self.pipe.level1.get(ex))
            out.append(dict(rep=i + 1, start=int(r[0]), low=int(r[1]), end=int(r[2]), counted=counted, metrics=m,
                            rule=rp, gru=gp, gru_correct=None if gc is None else float(gc[i]),
                            fused={f: p for f, (p, _) in fu.items()}, **v))
        return out

    def pose_one(self, yolo, img, box=None):
        """YOLO pose on one frame, following the person in `box` (same choice
        rule as ec_pipeline.pose_frames). Returns K (17,2) px, S (17,), box."""
        E = self.E
        K = np.full((17, 2), np.nan, np.float32)
        S = np.zeros(17, np.float32)
        kw = dict(device=self.device, imgsz=self.imgsz, conf=0.25, verbose=False, **E.yolo_precision_kw(self.device))
        with self.lock:
            r = yolo.predict([img], **kw)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None or len(r.keypoints) == 0:
            return K, S, box
        bx = r.boxes.xyxy.float().cpu().numpy()
        if box is not None and max(E._iou(b, box) for b in bx) > 0.1:
            k = int(np.argmax([E._iou(b, box) for b in bx]))
        else:
            k = int(np.argmax((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])))
        K = r.keypoints.xy[k].float().cpu().numpy()
        S = r.keypoints.conf[k].float().cpu().numpy() if r.keypoints.conf is not None else np.ones(17, np.float32)
        return K, S, bx[k]

    @staticmethod
    def public_rep(r, fps, t_offset=0.0):
        """The parts of a rep verdict the browser needs, times in seconds."""
        m = r.get('metrics', {})
        return _clean({
            'rep': r['rep'],
            'counted': r['counted'],
            'start': t_offset + r['start'] / fps,
            'low': t_offset + r['low'] / fps,
            'end': t_offset + r['end'] / fps,
            'correct': r['correct'],
            'score': r['score'],
            'faults': [{'key': f, 'p': p} for f, p in r['faults']],
            'hints': [{'key': f, 'p': p} for f, p in r['hints']],
            'p_correct': r.get('gru_correct'),
            'view': m.get('view'),
            'fused': r.get('fused'),
        })

    # ------------------------------------------------------------------
    # Uploaded video
    # ------------------------------------------------------------------
    def analyze_video(self, path, ex, progress=None):
        """Full pipeline on a file, the same steps as Pipeline.__call__,
        plus the cleaned skeleton for every frame so the page can draw it."""
        E = self.E
        progress = progress or (lambda p, msg: None)
        progress(0.02, 'Reading video')
        frames, info = E.decode_video(path, max_seconds=MAX_VIDEO_SECONDS)
        if not frames:
            return {'success': False, 'error': 'Could not read any frames from this video.'}
        n = len(frames)
        H, W = frames[0].shape[:2]

        engine = self

        class _Counting:
            """Passes calls through to YOLO and reports how far pose has got."""
            def __init__(self):
                self.done = 0

            def predict(self, batch, **kw):
                with engine.lock:
                    out = engine.pipe.yolo.predict(batch, **kw)
                self.done += len(batch)
                progress(0.05 + 0.9 * self.done / n, 'Finding your pose')
                return out

        K, S = E.pose_frames(_Counting(), frames, device=self.device, imgsz=self.imgsz)
        present = np.isfinite(K).all(-1).any(-1) & (S.max(-1) > 0)
        if present.mean() < 0.3:
            return {'success': False, 'error': 'No person found in most of the video. Keep your whole body in frame.'}
        progress(0.96, 'Judging windows' if ex in E.WINDOW_EX else 'Judging reps')
        xy, cf = E.clean_clip(K, S)
        with self.lock:
            reps, _ = E.analyze_reps(xy, cf, ex, self.pipe.gru.get(ex), self.pipe.rule_params, self.pipe.thresholds,
                                     None, self.device, weights=self.weights_for(ex),
                                     level1=self.pipe.level1.get(ex))
        kp = np.concatenate([xy / np.array([W, H], np.float32), cf[..., None]], -1)
        kp[~present] = 0.0                                   # nobody there: draw nothing
        out = {
            'success': True,
            'engine': 'v5',
            'exercise': ex,
            'unit': 'window' if ex in E.WINDOW_EX else 'rep',
            'calibration': self.pipe.calibration,
            'fps': E.FPS,
            'frames': n,
            'seconds': n / E.FPS,
            'frame_size': [W, H],
            'reps_counted': int(sum(r['counted'] for r in reps)),
            'reps': [self.public_rep(r, E.FPS) for r in reps],
            'keypoints': np.round(kp, 4).tolist(),
        }
        if ex == 'jump_rope':
            jumps = E.count_jumps(xy)
            out['jumps'] = len(jumps)
            out['jump_times'] = [round(j / E.FPS, 3) for j in jumps]   # lets playback count along
        if ex == 'plank':
            out['hold_seconds'] = round(float(present.sum()) / E.FPS, 1)
        return out

    # ------------------------------------------------------------------
    # Single photo
    # ------------------------------------------------------------------
    def analyze_image(self, img, ex):
        E = self.E
        if ex == 'jump_rope':
            return {'success': False,
                    'error': 'Jump rope is judged from movement, so one photo cannot show it. '
                             'Use a video or the webcam instead.'}
        K, S, _ = self.pose_one(self.pipe.yolo, img)
        if not (S.max() > 0):
            return {'success': False, 'error': 'No person found in this photo.'}
        H, W = img.shape[:2]
        # The plank front-view ratios need three frames, so a photo is held still.
        n = 3 if ex in E.WINDOW_EX else 1
        xy, cf = np.repeat(K[None], n, 0).astype(np.float32), np.repeat(S[None], n, 0).astype(np.float32)
        m = E.rep_metrics(xy, cf, (0, 0, n - 1), ex)
        for key in MOTION_METRICS:
            if key in m:
                m[key] = float('nan')
        rp = E.rule_probs(m, ex, self.pipe.rule_params)
        w = self.weights_for(ex)
        if ex in E.WINDOW_EX:
            # v6.1 tuned these rule weights alongside the BiGRU and turned most
            # plank rules off. A photo has no BiGRU, so the rules must stay on.
            w.pop('rule_by_fault', None)
        fu = E.fuse(rp, None, None, ex, w)
        v = E.rep_verdict(fu, ex, self.pipe.thresholds)
        checked = [f for f, (p, _) in fu.items() if p is not None]
        return _clean({
            'success': True,
            'engine': 'v5',
            'exercise': ex,
            'correct': v['correct'],
            'score': v['score'],
            'faults': [{'key': f, 'p': p} for f, p in v['faults']],
            'checked': checked,
            'not_checked': [f for f in E.FAULTS[ex] if f not in checked],
            'fused': {f: p for f, (p, _) in fu.items()},
            'metrics': {k: val for k, val in m.items() if k not in ('facing', 'low')},
            'view': m.get('view'),
            'frame_size': [W, H],
            'keypoints': [[float(K[j, 0]) / W, float(K[j, 1]) / H, float(S[j])] for j in range(17)],
        })


# ----------------------------------------------------------------------
# Live webcam
# ----------------------------------------------------------------------
class LiveSession:
    """Rep-by-rep judging for frames that arrive one at a time.

    The pipeline's rep counter works on a whole clip at 16 fps. Here the
    last WINDOW_S seconds of poses are resampled onto that 16 fps clock
    after every frame, cleaned and passed to the same counter; any rep it
    closes that has not been judged yet goes through the same rules +
    BiGRU fusion as an uploaded video. Jump rope and plank work the same
    way with the pipeline's 2 second windows in place of reps, plus a
    running jump count or hold timer.
    """
    WINDOW_S = 20.0
    MIN_SECONDS = 1.0
    MAX_HOLD_GAP_S = 2.0        # a longer gap between frames is not counted as hold time
    MIN_WINDOW_PRESENT = 0.5    # a window that mostly has nobody in it is not judged

    def __init__(self, engine, ex):
        self.engine, self.ex = engine, ex
        self.reset()

    def reset(self):
        self.t0 = None
        self.ts, self.K, self.S = [], [], []
        self.box = None
        self.reps = []
        self.last_low = -1e9
        self.jumps = 0
        self.last_jump = -1e9
        self.hold_seconds = 0.0
        self.last_seen = None
        self.last_active = time.time()

    def add_frame(self, img, t_ms):
        E = self.engine.E
        self.last_active = time.time()
        H, W = img.shape[:2]
        K, S, box = self.engine.pose_one(self.engine.live_yolo, img, self.box)
        person = bool(S.max() > 0)
        if person:
            self.box = box

        t = float(t_ms) / 1000.0
        if self.t0 is None:
            self.t0 = t
        t -= self.t0
        if not self.ts or t > self.ts[-1]:
            self.ts.append(t)
            self.K.append(K)
            self.S.append(S)
            while self.ts and self.ts[0] < t - self.WINDOW_S:
                self.ts.pop(0); self.K.pop(0); self.S.pop(0)
            if person and self.last_seen is not None:
                self.hold_seconds += min(t - self.last_seen, self.MAX_HOLD_GAP_S)
            self.last_seen = t if person else None

        new_reps = self._judge_new()
        out = {
            'success': True,
            'engine': 'v5',
            'unit': 'window' if self.ex in E.WINDOW_EX else 'rep',
            'person': person,
            'keypoints': [[float(K[j, 0]) / W, float(K[j, 1]) / H, float(S[j])] for j in range(17)] if person else None,
            'rep_count': len(self.reps),
            'new_reps': new_reps,
            'buffer_seconds': (self.ts[-1] - self.ts[0]) if self.ts else 0.0,
        }
        if self.ex == 'jump_rope':
            out['jumps'] = self.jumps
        if self.ex == 'plank':
            out['hold_seconds'] = round(self.hold_seconds, 1)
        return _clean(out)

    def _resampled(self):
        """Raw samples -> 16 fps grid. Each joint is interpolated only between
        frames where it was seen; elsewhere its confidence is 0, which
        ec_pipeline.fill_gaps treats as missing, exactly as for a video."""
        E = self.engine.E
        ts = np.array(self.ts)
        K, S = np.stack(self.K), np.stack(self.S)
        grid = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / E.FPS)
        Kg = np.full((len(grid), 17, 2), np.nan, np.float32)
        Sg = np.zeros((len(grid), 17), np.float32)
        for j in range(17):
            ok = (S[:, j] >= E.CONF_THR) & np.isfinite(K[:, j]).all(-1)
            Sg[:, j] = np.interp(grid, ts, S[:, j])
            if ok.sum() == 0:
                continue
            for d in range(2):
                Kg[:, j, d] = np.interp(grid, ts[ok], K[ok, j, d])
        return grid, Kg, Sg

    def _judge_new(self):
        E = self.engine.E
        if len(self.ts) < 4 or self.ts[-1] - self.ts[0] < self.MIN_SECONDS:
            return []
        grid, Kg, Sg = self._resampled()
        present = Sg.max(-1) > 0
        if present.mean() < 0.3:
            return []
        xy, cf = E.clean_clip(Kg, Sg)
        reps, _ = E.count_reps(xy, self.ex)
        if self.ex in E.WINDOW_EX:
            fresh = self._fresh_windows(reps, grid, present)
        else:
            # Re-running the counter on a sliding window can nudge a rep's
            # edges, so a rep is new only if its bottom is clearly after the
            # bottom of the last judged rep (the counter's own minimum spacing).
            fresh = [r for r in reps if grid[r[1]] > self.last_low + E.MIN_REP_S]
        if not fresh:
            return []
        if self.ex == 'jump_rope':
            self._count_jumps(xy, grid, fresh)
        judged = self.engine.judge_windows(xy, cf, self.ex, fresh, counted=True)
        out = []
        for r in judged:
            self.last_low = max(self.last_low, float(grid[r['low']]))
            r['rep'] = len(self.reps) + 1
            pub = self.engine.public_rep(r, E.FPS, t_offset=float(grid[0]))
            self.reps.append(pub)
            out.append(pub)
        return out

    def _fresh_windows(self, windows, grid, present):
        """The buffer slides, so the fixed windows move with it. A window is
        new only if its middle is nearly a window length after the last
        judged one, and only if someone is actually in it."""
        E = self.engine.E
        gap = E.WIN_T / E.FPS - 0.25
        fresh, last = [], self.last_low
        for r in windows:
            if grid[r[1]] >= last + gap - 1e-6 and present[r[0]:r[2] + 1].mean() >= self.MIN_WINDOW_PRESENT:
                fresh.append(r)
                last = float(grid[r[1]])
        return fresh

    def _count_jumps(self, xy, grid, windows):
        """Add the jumps of newly judged windows. Windows overlap by a
        quarter second, so a jump seen by two of them only counts once."""
        E = self.engine.E
        for r in windows:
            for j in E.count_jumps(xy[r[0]:r[2] + 1]):
                t = float(grid[r[0] + j])
                if t >= self.last_jump + 0.25:
                    self.jumps += 1
                    self.last_jump = t


class LiveSessions:
    """One LiveSession per browser and exercise; idle ones are dropped."""
    IDLE_S = 600

    def __init__(self):
        self._s = {}
        self._lock = threading.Lock()

    def get(self, device_id, ex, create=True):
        key = (device_id, ex)
        with self._lock:
            now = time.time()
            for k in [k for k, s in self._s.items() if now - s.last_active > self.IDLE_S]:
                del self._s[k]
            if key not in self._s and create:
                self._s[key] = LiveSession(get_engine(), ex)
            return self._s.get(key)

    def reset(self, device_id, ex):
        s = self.get(device_id, ex, create=False)
        if s:
            s.reset()


# ----------------------------------------------------------------------
# Background jobs for uploaded videos
# ----------------------------------------------------------------------
class VideoJobs:
    """Video analysis takes from seconds to minutes on a CPU, longer than a
    request should hang, so each upload becomes a job the page polls."""
    KEEP_S = 3600

    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def start(self, file_storage, ex, owner):
        self._prune()
        job_id = uuid.uuid4().hex
        tmpdir = tempfile.mkdtemp(prefix='ec_v5_')
        name = os.path.basename(file_storage.filename or 'upload.mp4') or 'upload.mp4'
        path = os.path.join(tmpdir, name)
        file_storage.save(path)
        job = {'id': job_id, 'owner': owner, 'exercise': ex, 'status': 'queued', 'progress': 0.0,
               'message': 'Waiting to start', 'result': None, 'created': time.time()}
        with self._lock:
            self._jobs[job_id] = job

        def progress(p, msg):
            job['progress'], job['message'] = float(p), msg

        def run():
            job['status'] = 'running'
            try:
                res = get_engine().analyze_video(path, ex, progress)
                job['result'] = res
                job['status'] = 'done' if res.get('success') else 'error'
                job['message'] = 'Done' if res.get('success') else res.get('error', 'Analysis failed')
                job['progress'] = 1.0
            except Exception as exc:                  # noqa: BLE001
                job['status'], job['message'] = 'error', f'Analysis failed: {exc}'
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        threading.Thread(target=run, daemon=True, name=f'ec-v5-{job_id[:8]}').start()
        return job

    def get(self, job_id, owner):
        job = self._jobs.get(job_id)
        if not job or job['owner'] != owner:
            return None
        return job

    def _prune(self):
        now = time.time()
        with self._lock:
            for k in [k for k, j in self._jobs.items() if now - j['created'] > self.KEEP_S]:
                del self._jobs[k]


def load_pipeline_config():
    """Fault list and coaching text from the bundle, without loading torch."""
    with open(os.path.join(BUNDLE_DIR, 'pipeline_config.json')) as fh:
        return json.load(fh)
