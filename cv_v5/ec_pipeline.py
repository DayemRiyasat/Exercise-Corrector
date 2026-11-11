"""
ec_pipeline.py  -  Exercise-Corrector v6 core (v5.1 + jump rope and elbow plank).

One file shared by the Kaggle training notebook and the standalone inference script:
  pose cleaning + rep counter (same as v4), fault vocabulary, rule engine with fitted
  thresholds, per-rep pose BiGRU, FitCoach 3D CNN judge, fusion, scoring and rendering.

Skeleton: COCO-17 from YOLO11-pose, pixel coordinates, image y pointing down.

v6: jump rope and plank are judged in fixed 2-second windows (32 frames at 16 fps) instead of reps: a plank is a
hold, and jump-rope jumps are too fast and small to cut one by one. Squat, push-up and lunge are unchanged from v5.1.
"""
import json
import math
import os
import re
import subprocess
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

FPS = 16
CONF_THR = 0.3
REP_T = 32                     # every rep is resampled to this many frames for the GRU

COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
        "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
(NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR,
 LHIP, RHIP, LKN, RKN, LANK, RANK) = range(17)
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
INWARD = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5), (10, 8), (8, 6),
          (5, 0), (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
EDGES = INWARD + [(5, 6), (11, 12)]
EXERCISES = ["squat", "pushup", "lunge", "jump_rope", "plank"]
WINDOW_EX = ("jump_rope", "plank")        # judged in fixed time windows, not reps
WIN_T = 32                                 # window length in frames (2 s at 16 fps) = REP_T, so no resampling
WIN_STRIDE_TRAIN, WIN_STRIDE_INFER = 16, 32   # overlapping windows for training, back-to-back in the app

# =====================================================================================
# 1. Fault vocabulary (fixed, shared by rules, GRU, 3D CNN and the report)
# =====================================================================================
FAULTS = {
    "squat":  ["shallow", "knees_inward", "torso_lean", "knees_over_toes", "stance_wide", "stance_narrow"],
    "pushup": ["shallow", "hip_sag", "hip_pike", "head_down", "no_lockout"],
    "lunge":  ["shallow", "knee_over_toes", "torso_lean", "back_knee_straight"],
    "jump_rope": ["no_jump", "heels_only", "side_to_side", "no_wrist_circles"],
    "plank":  ["hip_sag", "hip_pike", "head_down", "head_up", "on_knees", "shoulders_shrugged", "arms_wide"],
}
FAULT_TEXT = {
    "shallow": "Go deeper",
    "knees_inward": "Knees are caving in, push them out",
    "torso_lean": "Torso leans too far forward, keep the chest up",
    "knees_over_toes": "Knees travel too far past the toes, sit back",
    "knee_over_toes": "Front knee travels past the toes, shorten or sit back",
    "stance_wide": "Feet are too wide, bring them to shoulder width",
    "stance_narrow": "Feet are too close, widen to shoulder width",
    "hip_sag": "Hips are sagging, brace the core",
    "hip_pike": "Hips are too high, keep a straight line",
    "head_down": "Head is dropping, keep the neck neutral",
    "no_lockout": "Arms do not fully extend at the top",
    "back_knee_straight": "Back knee is not bending, drop it toward the floor",
    "no_jump": "Feet are not leaving the floor, hop a little higher",
    "heels_only": "Only the heels are moving, push off from the balls of the feet",
    "side_to_side": "Jumping side to side, stay on one spot",
    "no_wrist_circles": "Turn the rope with small wrist circles",
    "head_up": "Head is lifted, look at the floor just ahead of your hands",
    "on_knees": "Knees are on the floor, lift them and hold a straight line",
    "shoulders_shrugged": "Shoulders are shrugged toward the ears, push them down",
    "arms_wide": "Elbows are too wide, place them under the shoulders",
    "form_issue": "Form differs from a clean rep",
}
SEVERITY = {"shallow": 0.6, "knees_inward": 0.7, "torso_lean": 0.5, "knees_over_toes": 0.4, "knee_over_toes": 0.5,
            "stance_wide": 0.3, "stance_narrow": 0.3, "hip_sag": 0.7, "hip_pike": 0.5, "head_down": 0.3,
            "no_lockout": 0.4, "back_knee_straight": 0.5, "form_issue": 0.4,
            "no_jump": 0.6, "heels_only": 0.4, "side_to_side": 0.3, "no_wrist_circles": 0.3, "head_up": 0.3,
            "on_knees": 0.6, "shoulders_shrugged": 0.3, "arms_wide": 0.3}

# QEVD exercise names that belong to each of our exercises
QEVD_NAMES = {
    "squat":  ["squats"],
    "pushup": ["pushups"],
    "lunge":  ["alternating forward lunges", "lunges (left leg out in front)", "lunges (right leg out in front)"],
    "jump_rope": ["air jump rope"],
    "plank":  ["elbow plank"],
}
# QEVD attribute (short tag, lower case) -> fault. Tags only; the long "User's ..." sentences duplicate them.
QEVD_FAULT_RX = {
    "squat": {
        "shallow": r"^shallow$|^insufficient$",
        "knees_inward": r"knees? (inward|caving|cave)|inward knees",
        "torso_lean": r"back not straight|leaning forward|upper body leaning",
        "knees_over_toes": r"knees? over (the )?toes",
        "stance_wide": r"^wide$|^plie$|too wide",
        "stance_narrow": r"^narrow$|too narrow|too close",
    },
    "pushup": {
        "shallow": r"^shallow$|^insufficient$",
        "hip_sag": r"butt sinking|hips? (sagging|too low|sinking)",
        "hip_pike": r"elevated hips|hips? too high|piking",
        "head_down": r"^head down$",
    },
    "lunge": {
        "shallow": r"^insufficient$|^shallow$|^depth=[12]$|insufficient depth",
        "knee_over_toes": r"knees? (going forward )?over (the )?toes",
        "torso_lean": r"leaning forward",
        "back_knee_straight": r"not bending (the )?back leg",
    },
    "jump_rope": {
        "no_jump": r"^no jump$",
        "heels_only": r"^heels only$",
        "side_to_side": r"^side to side$",
        "no_wrist_circles": r"^no wrist circles$",
    },
    "plank": {
        "hip_sag": r"butt sinking|legs and hips on the floor",
        "hip_pike": r"^high hips$",
        "head_down": r"^head down$",
        "head_up": r"^head up$",
        "on_knees": r"on knees",
        "shoulders_shrugged": r"shrugged",
        "arms_wide": r"arms too wide",
    },
}
QEVD_CLEAN_RX = r"^no obvious issue$|^average$|^good form$|no issue|no mistake|^correct$|^proper$"
QEVD_CAMERA_RX = r"not visible|not fully in view|out of frame"
# v4 aspects: the CORRECT option of each "pick one" aspect (squat stance / depth, push-up hand width / depth,
# lunge step width / depth level). A clip with only these is a correct clip, exactly like v4.
QEVD_OK_RX = {
    "squat":  r"^shoulder-width$|^90 degrees$",
    "pushup": r"^shoulder-width$|^90 degrees$",
    "lunge":  r"^normal width$|^depth=[345]$",
    "jump_rope": r"(?!x)x",
    "plank":  r"(?!x)x",
}
# v4 "too deep" aspect option: an INCORRECT clip (not the correct option), but certainly not shallow
QEVD_TOO_DEEP_RX = {"squat": r"^over 90 degrees$|too deep", "pushup": r"^deep$|too deep", "lunge": r"(?!x)x",
                    "jump_rope": r"(?!x)x", "plank": r"(?!x)x"}
# neither correct nor incorrect for per-rep form: v4's neutral tags, plus timing/tempo tags (each rep is resampled
# to 32 frames, so tempo is invisible to the per-rep models and would only add label noise)
QEVD_NEUTRAL_RX = (r"hands behind head|hands on hips|starting late|waiting position|^hold$|early stopping|"
                   r"stopping early|turnover|^normal speed$|^fast$|^slow$|^too fast$|^too slow$|speed=|rom=|"
                   r"height=|jump_height=|kick_height=|^shaking$")
QEVD_PARAM_RX = r"=\s*-?\d"
# explicit negatives (the tag says the fault is NOT present); also the 3D CNN judge's "no" evidence
QEVD_NEG_RX = {
    "squat":  {"shallow": r"^90 degrees$|^over 90 degrees$|too deep", "stance_wide": r"^shoulder-width$",
               "stance_narrow": r"^shoulder-width$"},
    "pushup": {"shallow": r"^90 degrees$|^deep$|too deep"},
    "lunge":  {"shallow": r"^depth=[345]$"},
    "jump_rope": {},
    "plank":  {},
}


def classify_tag(ex, tag):
    """Role of one canonical QEVD tag (v4 semantics):
    camera | clean | fault:<a,b> | too_deep | ok | neutral | other_issue.
    other_issue = a v4 mistake or a wrong aspect option that is not in our fault list: the clip is INCORRECT,
    but which of our faults is present is unknown."""
    a = tag.strip().lower()
    if re.search(QEVD_CAMERA_RX, a):
        return "camera"
    if re.search(QEVD_CLEAN_RX, a):
        return "clean"
    hits = [f for f, rx in QEVD_FAULT_RX[ex].items() if re.search(rx, a)]
    if hits:
        return "fault:" + ",".join(hits)
    if re.search(QEVD_TOO_DEEP_RX[ex], a):
        return "too_deep"
    if re.search(QEVD_OK_RX[ex], a):
        return "ok"
    if re.search(QEVD_NEUTRAL_RX, a) or re.search(QEVD_PARAM_RX, a):
        return "neutral"
    return "other_issue"


def qevd_targets(ex, tags, vocab=None):
    """tags: canonical tags of one clip (after the v4 tag/sentence merge).
    Returns y (K,), known (K,), flags with v4's two levels: correct (bool) + correct_known (bool)."""
    faults = FAULTS[ex]
    roles = [(t.strip().lower(), classify_tag(ex, t)) for t in tags]
    rset = {r.split(":")[0] for _, r in roles}
    pos = {f for _, r in roles if r.startswith("fault:") for f in r[6:].split(",")}
    neg = {f for t, _ in roles for f, rx in QEVD_NEG_RX[ex].items() if re.search(rx, t)}
    has_fault, has_other, too_deep = "fault" in rset, "other_issue" in rset, "too_deep" in rset
    correct_known = bool(rset & {"clean", "ok", "fault", "other_issue", "too_deep"}) and "camera" not in rset
    correct = correct_known and not (has_fault or has_other or too_deep)
    y = np.zeros(len(faults), np.float32)
    known = np.zeros(len(faults), np.float32)
    for k, f in enumerate(faults):
        in_vocab = vocab is None or f in vocab
        if f in pos:
            y[k], known[k] = 1, 1
        elif f in neg:
            known[k] = 1
        elif correct and in_vocab:
            known[k] = 1                       # a correct clip has none of the faults
        elif has_fault and not has_other and in_vocab:
            known[k] = 1                       # one instructed variation per clip: the other faults are absent
        # other_issue / too_deep / neutral-only clips: the remaining faults stay unknown (masked)
    return y, known, dict(camera="camera" in rset, correct=bool(correct), correct_known=bool(correct_known),
                          other_issue=bool(has_other or too_deep),
                          informative=bool(correct_known or known.sum() > 0), roles=roles)


# EC3D (ACCV 2022) mistake codes -> our faults
EC3D_ACT = {"SQUAT": "squat", "Lunges": "lunge"}
# 5 "Front bent" is excluded: measured on the data, those reps lean LESS than correct ones and drive the
# knees forward instead, so it does not mean torso lean and its meaning is unclear. 10 "Unknown" excluded.
EC3D_FAULT = {
    "squat": {1: None, 2: "stance_wide", 3: "knees_inward", 4: "shallow"},
    "lunge": {1: None, 4: "shallow", 6: "knee_over_toes"},
}


def ec3d_targets(ex, code):
    faults = FAULTS[ex]
    vocab = {f for f in EC3D_FAULT[ex].values() if f}
    y = np.zeros(len(faults), np.float32)
    known = np.array([1.0 if f in vocab else 0.0 for f in faults], np.float32)
    f = EC3D_FAULT[ex].get(code, "skip")
    if f == "skip":
        return None
    if f:
        y[faults.index(f)] = 1
    return y, known


# =====================================================================================
# 2. Skeleton cleaning and rep counter (unchanged from v4)
# =====================================================================================
def fill_gaps(xy, conf, thr=CONF_THR):
    xy = xy.copy()
    conf = np.clip(np.nan_to_num(conf), 0, 1).copy()
    miss = (conf < thr) | ~np.isfinite(xy).all(-1)
    t = np.arange(len(xy))
    for j in range(xy.shape[1]):
        ok = ~miss[:, j]
        if ok.sum() == 0:
            xy[:, j] = np.nan
            continue
        for d in range(2):
            xy[:, j, d] = np.interp(t, t[ok], xy[ok, j, d])
    conf[miss] = 0.0
    return xy, conf


def one_euro(x, fps=FPS, min_cutoff=1.5, beta=0.3, d_cutoff=1.0):
    if len(x) < 2:
        return x
    alpha = lambda fc: 1.0 / (1.0 + fps / (2 * math.pi * fc))
    out = np.empty_like(x)
    out[0] = x[0]
    dx_prev = np.zeros_like(x[0])
    for t in range(1, len(x)):
        dx = (x[t] - out[t - 1]) * fps
        dx_hat = dx_prev + alpha(d_cutoff) * (dx - dx_prev)
        a = alpha(min_cutoff + beta * np.abs(dx_hat))
        out[t] = out[t - 1] + a * (x[t] - out[t - 1])
        dx_prev = dx_hat
    return out


def torso_len(xy):
    d = np.linalg.norm((xy[:, 5] + xy[:, 6]) / 2 - (xy[:, 11] + xy[:, 12]) / 2, axis=-1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else 1.0


def clean_clip(kps, scores, smooth=True):
    xy, conf = fill_gaps(kps, scores)
    if np.isnan(xy).any():
        hip = np.nanmean((xy[:, 11] + xy[:, 12]) / 2, 0) if np.isfinite(xy[:, [11, 12]]).any() else np.zeros(2)
        xy = np.where(np.isnan(xy), np.nan_to_num(hip), xy)
    if smooth:
        s = max(torso_len(xy), 1e-3)
        xy = one_euro(xy / s) * s
    return xy.astype(np.float32), conf.astype(np.float32)


REP_JOINTS = {"squat": (11, 12), "pushup": (5, 6)}
MIN_RANGE, DOWN_F, UP_F, MIN_REP_S = 0.12, 0.60, 0.40, 0.5


def height_signal(xy, ex):
    if ex in WINDOW_EX:                            # hip height (bigger = lower body), used for display only
        return (xy[:, 11, 1] + xy[:, 12, 1]) / 2 / max(torso_len(xy), 1e-3)
    if ex == "lunge":
        tl = np.linalg.norm((xy[:, 5] + xy[:, 6]) / 2 - (xy[:, 11] + xy[:, 12]) / 2, axis=-1)
        return -((xy[:, 15, 1] + xy[:, 16, 1]) / 2 - (xy[:, 11, 1] + xy[:, 12, 1]) / 2) / np.maximum(tl, 1e-3)
    a, b = REP_JOINTS[ex]
    return (xy[:, a, 1] + xy[:, b, 1]) / 2 / max(torso_len(xy), 1e-3)


def time_windows(n, stride=WIN_STRIDE_INFER, win=WIN_T):
    """Fixed windows [(start, middle, end)] (end inclusive). A clip shorter than one window gets none (the caller
    then uses the whole clip). A leftover of at least 0.5 s at the end gets one more window aligned to the end."""
    if n < win:
        return []
    out = [(a, a + win // 2, a + win - 1) for a in range(0, n - win + 1, stride)]
    if n - 1 - out[-1][2] >= FPS // 2:
        out.append((n - win, n - win // 2, n - 1))
    return out


def count_jumps(xy, fps=FPS):
    """Frame indices of jump-rope jumps (peaks of hip height). Display only, the verdict comes from the windows."""
    from scipy.signal import find_peaks, savgol_filter
    s = -height_signal(xy, "jump_rope")
    if len(s) >= 5:
        s = savgol_filter(s, 5, 2)
    pk, _ = find_peaks(s, prominence=0.025, distance=max(1, int(0.25 * fps)))
    return [int(i) for i in pk]


def count_reps(xy, ex, fps=FPS, stride=None):
    """Returns ([(start, low, end), ...], smoothed signal). Bigger signal = lower body.
    jump_rope / plank: fixed time windows instead of reps (stride defaults to back-to-back windows)."""
    from scipy.signal import savgol_filter
    s = height_signal(xy, ex)
    if ex in WINDOW_EX:
        return time_windows(len(xy), stride or WIN_STRIDE_INFER), s
    if len(s) >= 7:
        s = savgol_filter(s, 7, 2)
    top, bottom = np.percentile(s, 10), np.percentile(s, 90)
    rng_ = bottom - top
    if rng_ < MIN_RANGE:
        return [], s
    down_th, up_th = top + DOWN_F * rng_, top + UP_F * rng_
    reps, state, start, low_i, seen_top = [], "up", 0, 0, False
    for i, v in enumerate(s):
        if state == "up":
            if v < up_th:
                start, seen_top = i, True
            elif v > down_th and seen_top:
                state, low_i = "down", i
        else:
            if v > s[low_i]:
                low_i = i
            if v < up_th:
                if not reps or (low_i - reps[-1][1]) >= MIN_REP_S * fps:
                    reps.append((start, low_i, i))
                state, start = "up", i
    return reps, s


def fallback_window(xy, ex):
    """One analysis window when no full rep is found: whole clip, low point = lowest body position."""
    if ex in WINDOW_EX:
        return (0, len(xy) // 2, len(xy) - 1)
    s = height_signal(xy, ex)
    return (0, int(np.nanargmax(s)) if np.isfinite(s).any() else len(xy) // 2, len(xy) - 1)


# =====================================================================================
# 3. Geometry helpers and per-rep measurements (rule engine inputs)
# =====================================================================================
def _ang(a, b, c):
    """Angle at b (degrees) for arrays (..., 2)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _mid(xy, i, j):
    return (xy[..., i, :] + xy[..., j, :]) / 2


def view_of(xy, conf=None):
    """'front' (camera faces the person), 'side', or 'diag', from shoulder/hip width vs torso length."""
    tl = np.linalg.norm(_mid(xy, LSH, RSH) - _mid(xy, LHIP, RHIP), axis=-1)
    sw = np.abs(xy[:, LSH, 0] - xy[:, RSH, 0])
    hw = np.abs(xy[:, LHIP, 0] - xy[:, RHIP, 0])
    r = float(np.nanmedian((sw + hw) / 2 / np.maximum(tl, 1e-3)))
    return ("front" if r >= 0.45 else "side" if r <= 0.28 else "diag"), r


def facing_sign(xy, conf):
    """+1 if the person faces image right, -1 if left (meaningful in side views)."""
    ears = _mid(xy, LEAR, REAR)
    ok = (conf[:, LEAR] > CONF_THR) | (conf[:, REAR] > CONF_THR) if conf is not None else np.ones(len(xy), bool)
    d = (xy[:, NOSE, 0] - ears[:, 0])[ok]
    if len(d) < 3 or abs(np.median(d)) < 1e-3:
        d = xy[:, NOSE, 0] - _mid(xy, LHIP, RHIP)[:, 0]
    return 1.0 if np.median(d) >= 0 else -1.0


def body_height(xy, ex):
    """Translation-free height of the moving part: hip above the ankles (squat, lunge),
    shoulders above the wrists (push-up). Smallest value = bottom of the rep."""
    if ex == "pushup":
        return (xy[:, LWR, 1] + xy[:, RWR, 1]) / 2 - (xy[:, LSH, 1] + xy[:, RSH, 1]) / 2
    ank = np.maximum(xy[:, LANK, 1], xy[:, RANK, 1]) if ex == "lunge" else (xy[:, LANK, 1] + xy[:, RANK, 1]) / 2
    return ank - (xy[:, LHIP, 1] + xy[:, RHIP, 1]) / 2


def rep_metrics(xy, conf, rep, ex):
    """Measurements for one rep. xy (T,17,2) pixels (whole clip), rep = (start, low, end).
    The bottom frame is re-found inside the rep as the lowest body height (camera-shake safe)."""
    s0, _, s1 = [int(v) for v in rep]
    seg, segc = xy[s0:s1 + 1], conf[s0:s1 + 1]
    view, vr = view_of(seg)
    if ex in WINDOW_EX:
        return window_metrics(seg, segc, ex, view, vr, s0 + (s1 - s0) // 2)
    fs = facing_sign(seg, segc)
    h = body_height(seg, ex)
    lo_rel, hi_rel = int(np.nanargmin(h)), int(np.nanargmax(h))
    top, bot = seg[hi_rel], seg[lo_rel]
    tl = max(torso_len(seg), 1e-3)
    m = {"view": view, "view_ratio": vr, "facing": fs, "low": s0 + lo_rel}
    trunk = lambda p: float(_ang(_mid(p[None], LHIP, RHIP)[0] + np.array([0, -1.0]),
                                 _mid(p[None], LHIP, RHIP)[0], _mid(p[None], LSH, RSH)[0]))
    if ex in ("squat", "lunge"):
        m["hip_drop"] = float(1 - h[lo_rel] / max(h[hi_rel], 1e-3))
        m["trunk_angle"] = trunk(bot)
        shin = np.median([np.linalg.norm(top[LKN] - top[LANK]), np.linalg.norm(top[RKN] - top[RANK])])
        kf = [(bot[k, 0] - bot[a, 0]) * fs / max(shin, 1e-3) for k, a in ((LKN, LANK), (RKN, RANK))]
    if ex == "squat":
        m["knee_fwd"] = float(np.mean(kf))
        ag = abs(bot[LANK, 0] - bot[RANK, 0])
        m["knee_gap_ratio"] = float(abs(bot[LKN, 0] - bot[RKN, 0]) / max(ag, 1e-3)) if ag > 0.15 * tl else np.nan
        sg = abs(top[LSH, 0] - top[RSH, 0])
        m["stance_ratio"] = float(abs(top[LANK, 0] - top[RANK, 0]) / max(sg, 1e-3)) if sg > 0.2 * tl else np.nan
    elif ex == "lunge":
        # front leg = the one whose ANKLE is further forward (the back shin is near horizontal, so its knee
        # is always far ahead of its own ankle and must not be mistaken for the front leg)
        front = 0 if bot[LANK, 0] * fs >= bot[RANK, 0] * fs else 1
        m["front_knee_fwd"] = float(kf[front])
        bk = (RHIP, RKN, RANK) if front == 0 else (LHIP, LKN, LANK)
        m["back_knee_angle"] = float(_ang(bot[bk[0]], bot[bk[1]], bot[bk[2]]))
    elif ex == "pushup":
        arm = np.nanmedian([np.linalg.norm(seg[:, a] - seg[:, b], axis=-1) + np.linalg.norm(seg[:, b] - seg[:, c], axis=-1)
                            for a, b, c in ((LSH, LEL, LWR), (RSH, REL, RWR))])
        m["shoulder_drop"] = float((h[hi_rel] - h[lo_rel]) / max(arm, 1e-3))
        sh, hp = _mid(seg, LSH, RSH), _mid(seg, LHIP, RHIP)
        an = _mid(seg, LANK, RANK)
        line = an - sh
        L = np.linalg.norm(line, axis=-1) + 1e-6
        cross = (line[:, 0] * (hp[:, 1] - sh[:, 1]) - line[:, 1] * (hp[:, 0] - sh[:, 0])) / L
        # sign so that positive = hip below the shoulder-ankle line (image y down)
        below = np.sign(line[:, 0] + 1e-9)
        m["hip_dev"] = float(np.median(cross * below / L))
        side = 0 if segc[:, [LSH, LEL, LWR]].mean() >= segc[:, [RSH, REL, RWR]].mean() else 1
        a, b, c = (LSH, LEL, LWR) if side == 0 else (RSH, REL, RWR)
        m["elbow_max"] = float(np.max(_ang(seg[:, a], seg[:, b], seg[:, c])))
    return m


def _spread(v):
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 95) - np.percentile(v, 5)) if len(v) > 2 else np.nan


def window_metrics(seg, segc, ex, view, vr, mid):
    """Measurements over one time window (jump rope, plank). seg (T,17,2) pixels."""
    tl = max(torso_len(seg), 1e-3)
    m = {"view": view, "view_ratio": vr, "facing": facing_sign(seg, segc), "low": int(mid)}
    if ex == "jump_rope":
        ank_y = (seg[:, LANK, 1] + seg[:, RANK, 1]) / 2
        hip = _mid(seg, LHIP, RHIP)
        m["ankle_lift"] = _spread(ank_y) / tl                 # how far the feet leave the floor
        m["hip_bounce"] = _spread(hip[:, 1]) / tl
        m["hip_sway"] = _spread(hip[:, 0]) / tl               # side-to-side travel
        wr = np.stack([seg[:, LWR] - seg[:, LSH], seg[:, RWR] - seg[:, RSH]], 1) / tl     # wrists relative to shoulders
        m["wrist_motion"] = float(np.nanmedian(np.linalg.norm(np.diff(wr, axis=0), axis=-1))) if len(wr) > 1 else np.nan
    elif ex == "plank":
        sh, hp, an = _mid(seg, LSH, RSH), _mid(seg, LHIP, RHIP), _mid(seg, LANK, RANK)
        line = an - sh
        L = np.linalg.norm(line, axis=-1) + 1e-6
        cross = (line[:, 0] * (hp[:, 1] - sh[:, 1]) - line[:, 1] * (hp[:, 0] - sh[:, 0])) / L
        m["hip_dev"] = float(np.median(cross * np.sign(line[:, 0] + 1e-9) / L))   # + = hips below the line (sag)
        tr = sh - an                                                               # ankles -> shoulders, extended to the head
        T_ = np.linalg.norm(tr, axis=-1) + 1e-6
        hc = (tr[:, 0] * (seg[:, NOSE, 1] - sh[:, 1]) - tr[:, 1] * (seg[:, NOSE, 0] - sh[:, 0])) / T_
        m["head_dev"] = float(np.median(hc * np.sign(tr[:, 0] + 1e-9) / tl))        # + = head dropped toward the floor
        kn = np.stack([_ang(seg[:, LHIP], seg[:, LKN], seg[:, LANK]), _ang(seg[:, RHIP], seg[:, RKN], seg[:, RANK])], 1)
        m["knee_angle"] = float(np.nanmedian(kn.mean(1)))
        sw = np.abs(seg[:, LSH, 0] - seg[:, RSH, 0])
        ok = sw > 0.2 * tl
        m["elbow_ratio"] = (float(np.median(np.abs(seg[ok, LEL, 0] - seg[ok, REL, 0]) / sw[ok])) if ok.sum() >= 3
                            else np.nan)
        ear_sh = (np.linalg.norm(seg[:, LEAR] - seg[:, LSH], axis=-1) + np.linalg.norm(seg[:, REAR] - seg[:, RSH], axis=-1)) / 2
        m["shrug_ratio"] = float(np.median(ear_sh[ok] / sw[ok])) if ok.sum() >= 3 else np.nan
    return m


# rule: fault -> (metric, direction, default threshold, valid views, (lo, hi) bounds for fitting)
RULES = {
    "squat": {
        "shallow":         ("hip_drop",       "<", 0.50, {"front", "side", "diag"}, (0.30, 0.70)),
        "knees_inward":    ("knee_gap_ratio", "<", 0.80, {"front"},                 (0.50, 1.00)),
        "torso_lean":      ("trunk_angle",    ">", 50.0, {"side", "diag"},          (30.0, 65.0)),
        "knees_over_toes": ("knee_fwd",       ">", 0.45, {"side"},                  (0.25, 0.80)),
        "stance_wide":     ("stance_ratio",   ">", 1.70, {"front"},                 (1.30, 2.50)),
        "stance_narrow":   ("stance_ratio",   "<", 0.75, {"front"},                 (0.40, 1.00)),
    },
    "pushup": {
        "shallow":    ("shoulder_drop", "<", 0.45, {"front", "side", "diag"}, (0.25, 0.70)),
        "hip_sag":    ("hip_dev",       ">", 0.05, {"side", "diag"},          (0.02, 0.15)),
        "hip_pike":   ("hip_dev",       "<", -0.06, {"side", "diag"},         (-0.20, -0.02)),
        "no_lockout": ("elbow_max",     "<", 155.0, {"side"},                 (140.0, 170.0)),
    },
    "lunge": {
        "shallow":            ("hip_drop",        "<", 0.43, {"front", "side", "diag"}, (0.20, 0.60)),
        "knee_over_toes":     ("front_knee_fwd",  ">", 0.25, {"side"},                  (0.10, 0.60)),
        "torso_lean":         ("trunk_angle",     ">", 25.0, {"side", "diag"},          (15.0, 45.0)),
        "back_knee_straight": ("back_knee_angle", ">", 135.0, {"side"},                 (115.0, 160.0)),
    },
    "jump_rope": {
        "no_jump":          ("ankle_lift",   "<", 0.05, {"front", "side", "diag"}, (0.01, 0.15)),
        "side_to_side":     ("hip_sway",     ">", 0.25, {"front", "diag"},         (0.08, 0.60)),
        "no_wrist_circles": ("wrist_motion", "<", 0.02, {"front", "side", "diag"}, (0.005, 0.08)),
    },
    "plank": {
        "hip_sag":            ("hip_dev",     ">", 0.05, {"side", "diag"},  (0.02, 0.20)),
        "hip_pike":           ("hip_dev",     "<", -0.06, {"side", "diag"}, (-0.25, -0.02)),
        "head_down":          ("head_dev",    ">", 0.25, {"side", "diag"},  (0.05, 0.60)),
        "head_up":            ("head_dev",    "<", -0.15, {"side", "diag"}, (-0.50, 0.00)),
        "on_knees":           ("knee_angle",  "<", 140.0, {"side", "diag"}, (110.0, 165.0)),
        "shoulders_shrugged": ("shrug_ratio", "<", 0.30, {"front"},         (0.10, 0.60)),
        "arms_wide":          ("elbow_ratio", ">", 1.35, {"front"},         (1.05, 2.00)),
    },
}
DEFAULT_SCALE = {"hip_drop": 0.05, "knee_gap_ratio": 0.08, "trunk_angle": 6.0, "knee_fwd": 0.08,
                 "front_knee_fwd": 0.08, "stance_ratio": 0.15, "shoulder_drop": 0.06, "hip_dev": 0.02,
                 "elbow_max": 5.0, "back_knee_angle": 8.0, "ankle_lift": 0.015, "hip_sway": 0.05,
                 "wrist_motion": 0.005, "head_dev": 0.05, "knee_angle": 8.0, "shrug_ratio": 0.05, "elbow_ratio": 0.10}


def default_rule_params():
    return {ex: {f: dict(metric=r[0], dir=r[1], thr=float(r[2]), views=sorted(r[3]),
                         scale=DEFAULT_SCALE[r[0]], fitted=False)
                 for f, r in rules.items()} for ex, rules in RULES.items()}


def rule_probs(metrics, ex, params):
    """fault -> probability (or None when the rule can't judge this view / metric)."""
    out = {}
    for f in FAULTS[ex]:
        p = params[ex].get(f)
        if p is None:
            out[f] = None
            continue
        v = metrics.get(p["metric"], np.nan)
        if metrics["view"] not in p["views"] or v is None or not np.isfinite(v):
            out[f] = None
            continue
        z = (v - p["thr"]) / p["scale"] if p["dir"] == ">" else (p["thr"] - v) / p["scale"]
        out[f] = float(1 / (1 + math.exp(-np.clip(z, -30, 30))))
    return out


def fit_rule_params(samples, min_each=15):
    """samples: list of dict(ex, metrics, y, known). Fits threshold + scale per rule by balanced accuracy,
    inside the rule's bounds; keeps the default when there are too few labelled examples."""
    from sklearn.metrics import balanced_accuracy_score
    params, report = default_rule_params(), []
    for ex, rules in RULES.items():
        for f, (metric, d, thr0, views, (lo, hi)) in rules.items():
            k = FAULTS[ex].index(f)
            v, y = [], []
            for s in samples:
                if s["ex"] != ex or not s["known"][k] or s["metrics"]["view"] not in views:
                    continue
                x = s["metrics"].get(metric, np.nan)
                if x is not None and np.isfinite(x):
                    v.append(x); y.append(int(s["y"][k]))
            v, y = np.array(v), np.array(y)
            npos, nneg = int(y.sum()), int(len(y) - y.sum())
            row = dict(exercise=ex, fault=f, metric=metric, dir=d, default=thr0, n_pos=npos, n_neg=nneg)
            if npos >= min_each and nneg >= min_each:
                grid = np.linspace(lo, hi, 61)
                ba = [balanced_accuracy_score(y, (v > g) if d == ">" else (v < g)) for g in grid]
                g = float(grid[int(np.argmax(ba))])
                iqr = np.subtract(*np.percentile(v, [75, 25])) if len(v) > 4 else 0
                params[ex][f].update(thr=g, scale=float(max(iqr / 4, DEFAULT_SCALE[metric] / 2)), fitted=True)
                row.update(threshold=g, bal_acc=float(max(ba)), fitted=True)
            else:
                row.update(threshold=thr0, bal_acc=np.nan, fitted=False)
            report.append(row)
    return params, report


# =====================================================================================
# 4. Per-rep pose BiGRU
# =====================================================================================
def resample(a, n=REP_T):
    """Linear resample along time: a (T, ...) -> (n, ...)."""
    T = len(a)
    if T == 1:
        return np.repeat(a, n, 0)
    src = np.linspace(0, T - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0).reshape(-1, *([1] * (a.ndim - 1)))
    return (a[i0] * (1 - w) + a[i1] * w).astype(np.float32)


def rep_tensor(xy, conf, rep, pad=2):
    s0, _, s1 = rep
    a, b = max(0, s0 - pad), min(len(xy), s1 + pad + 1)
    return resample(xy[a:b]), resample(conf[a:b])


def _tang(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
    return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) / math.pi


def features_torch(xy, cf):
    """xy (B,T,17,2) pixels, cf (B,T,17) -> (B,T,F) view-normalised features."""
    hip = (xy[:, :, LHIP] + xy[:, :, RHIP]) / 2
    sh = (xy[:, :, LSH] + xy[:, :, RSH]) / 2
    tl = (sh - hip).norm(dim=-1).median(dim=1).values.clamp(min=1e-3)            # B
    p = (xy - hip.mean(1, keepdim=True)[:, :, None]) / tl[:, None, None, None]
    J = lambda i: p[:, :, i]
    ang = torch.stack([
        _tang(J(LHIP), J(LKN), J(LANK)), _tang(J(RHIP), J(RKN), J(RANK)),        # knees
        _tang(J(LSH), J(LHIP), J(LKN)), _tang(J(RSH), J(RHIP), J(RKN)),          # hips
        _tang(J(LSH), J(LEL), J(LWR)), _tang(J(RSH), J(REL), J(RWR)),            # elbows
        _tang(J(LEL), J(LSH), J(LHIP)), _tang(J(REL), J(RSH), J(RHIP)),          # shoulders
        _tang((J(LSH) + J(RSH)) / 2, (J(LHIP) + J(RHIP)) / 2, (J(LANK) + J(RANK)) / 2),   # body line
    ], -1)
    hp, s_ = (J(LHIP) + J(RHIP)) / 2, (J(LSH) + J(RSH)) / 2
    up = torch.tensor([0.0, -1.0], device=xy.device)
    trunk = s_ - hp
    trunk_ang = torch.acos(((trunk * up).sum(-1) / (trunk.norm(dim=-1) + 1e-6)).clamp(-1, 1))[..., None] / math.pi
    shin_l, shin_r = J(LKN) - J(LANK), J(RKN) - J(RANK)
    shin = torch.stack([torch.acos(((shin_l * up).sum(-1) / (shin_l.norm(dim=-1) + 1e-6)).clamp(-1, 1)),
                        torch.acos(((shin_r * up).sum(-1) / (shin_r.norm(dim=-1) + 1e-6)).clamp(-1, 1))], -1) / math.pi
    eps = 1e-3
    ankle_gap = (J(LANK)[..., 0] - J(RANK)[..., 0]).abs()
    sh_gap = (J(LSH)[..., 0] - J(RSH)[..., 0]).abs()
    ratios = torch.stack([
        ((J(LKN)[..., 0] - J(RKN)[..., 0]).abs() / (ankle_gap + 0.2)).clamp(0, 3),
        (ankle_gap / (sh_gap + 0.2)).clamp(0, 4),
        ((sh_gap + (J(LHIP)[..., 0] - J(RHIP)[..., 0]).abs()) / 2).clamp(0, 2),        # view cue
        ((J(LANK)[..., 1] + J(RANK)[..., 1]) / 2 - hp[..., 1]),                          # hip height
        ((J(LWR)[..., 1] + J(RWR)[..., 1]) / 2 - s_[..., 1]),                            # shoulder height
    ], -1)
    geo = torch.cat([ang, trunk_ang, shin], -1)                                         # B,T,12
    vel = torch.zeros_like(geo)
    vel[:, 1:] = (geo[:, 1:] - geo[:, :-1]) * 4
    feats = torch.cat([p.flatten(2), cf, geo, ratios, vel], -1)
    return torch.nan_to_num(feats, 0.0, 0.0, 0.0)


N_FEAT = 17 * 2 + 17 + 12 + 5 + 12


class PoseGRU(nn.Module):
    def __init__(s, n_faults, n_in=N_FEAT, hid=128):
        super().__init__()
        s.inp = nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, hid), nn.GELU(), nn.Dropout(0.1))
        s.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        s.att = nn.Linear(2 * hid, 1)
        s.drop = nn.Dropout(0.3)
        s.correct_head = nn.Linear(2 * hid, 1)
        s.fault_head = nn.Linear(2 * hid, n_faults)

    def forward(s, x):
        h, _ = s.gru(s.inp(x))
        a = torch.softmax(s.att(h), 1)
        z = s.drop((a * h).sum(1))
        return s.correct_head(z)[:, 0], s.fault_head(z), a[..., 0]


def augment(xy, cf, flip_ok=True):
    """Train-time augmentation on pixel skeletons (B,T,17,2): mirror, small rotation/scale, noise, glitches."""
    B, dev = xy.shape[0], xy.device
    c = xy.mean((1, 2), keepdim=True)
    x = xy - c
    th = (torch.rand(B, device=dev) * 16 - 8) * math.pi / 180
    R = torch.stack([torch.stack([th.cos(), -th.sin()], -1), torch.stack([th.sin(), th.cos()], -1)], 1)
    x = torch.einsum("btvd,bed->btve", x, R)
    x = x * (torch.rand(B, 1, 1, 1, device=dev) * 0.2 + 0.9)
    x = x * torch.stack([torch.rand(B, device=dev) * 0.15 + 0.92, torch.ones(B, device=dev)], -1)[:, None, None]
    scale = (x.flatten(1, 2).std(1).mean(-1)).clamp(min=1e-3)[:, None, None, None]
    x = x + torch.randn_like(x) * 0.01 * scale
    g = torch.rand(x.shape[:3], device=dev) < 0.02
    x = x + g[..., None] * torch.randn_like(x) * 0.08 * scale
    cf = cf * (1 - 0.7 * g.float())
    if flip_ok:
        fl = torch.rand(B, device=dev) < 0.5
        fi = torch.tensor(FLIP, device=dev)
        mir = torch.tensor([-1.0, 1.0], device=dev)
        x = torch.where(fl[:, None, None, None], x[:, :, fi] * mir, x)
        cf = torch.where(fl[:, None, None], cf[:, :, fi], cf)
    return x + c, cf


@torch.no_grad()
def gru_probs(model, xy_rep, cf_rep, device="cpu"):
    """xy_rep (N,REP_T,17,2), cf_rep (N,REP_T,17) -> P(correct) (N,), P(fault) (N,K)."""
    model.eval()
    x = features_torch(torch.as_tensor(xy_rep, device=device).float(), torch.as_tensor(cf_rep, device=device).float())
    lc, lf, _ = model(x)
    return torch.sigmoid(lc).cpu().numpy(), torch.sigmoid(lf).cpu().numpy()


# =====================================================================================
# 5. EC3D: 3D skeletons -> 2D views
# =====================================================================================
B25_TO_COCO = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def ec3d_upright_fix(seq3d):
    """EC3D rotates every frame so the torso stands upright, which erases forward lean. The planted foot
    (the one whose toe-heel tilt changes least) must keep its standing tilt, so each frame is rotated
    about the left-right axis until it does. seq3d (T,3,25) BODY_25 -> corrected copy."""
    from scipy.ndimage import median_filter
    S = np.asarray(seq3d, np.float64).copy()
    pitch = np.stack([np.arctan2(S[:, 2, t] - S[:, 2, h], S[:, 1, t] - S[:, 1, h]) for t, h in ((19, 21), (22, 24))], 1)
    hip_h = S[:, 2, 8] - (S[:, 2, 11] + S[:, 2, 14]) / 2
    top = int(np.argmax(hip_h))
    delta = pitch - pitch[top]                                   # T,2
    pick = np.argmin(np.abs(delta), 1)
    d = median_filter(delta[np.arange(len(S)), pick], size=5, mode="nearest")
    c, s = np.cos(-d), np.sin(-d)
    y, z = S[:, 1].copy(), S[:, 2].copy()
    S[:, 1] = y * c[:, None] - z * s[:, None]
    S[:, 2] = y * s[:, None] + z * c[:, None]
    return S


def ec3d_project(seq3d, yaw_deg, pitch_deg=0.0, px=400.0, size=(720, 1280)):
    """seq3d (T,3,25) EC3D coords (z up, y = facing, x = person's left) -> (T,17,2) image pixels.
    EC3D is mirrored (left at +x while facing +y), so x is flipped first to get a proper rigid body.
    yaw 0 = camera in front of the person, 90 = camera on the person's left side."""
    P = np.asarray(seq3d, np.float64)[:, :, B25_TO_COCO].transpose(0, 2, 1).copy()  # T,17,3
    P[..., 0] *= -1
    a, e = math.radians(yaw_deg), math.radians(pitch_deg)
    # camera position direction (from origin), looking at the origin
    cam = np.array([math.sin(a) * math.cos(e) * -1, math.cos(a) * math.cos(e), math.sin(e)])
    fwd = -cam
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right) + 1e-9
    upv = np.cross(right, fwd)
    u = P @ right
    v = -(P @ upv)
    h, w = size
    return np.stack([u * px + w / 2, v * px + h / 2], -1).astype(np.float32)


# =====================================================================================
# 6. FitCoach 3D CNN judge (StridedInflatedEfficientNet, MIT, Twenty Billion Neurons)
# =====================================================================================
class SteppableConv3dAs2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kw):
        kernel_size, stride, dilation = _triple(kernel_size), _triple(stride), _triple(dilation)
        super().__init__(in_channels * kernel_size[0], out_channels, kernel_size[1:],
                         stride=stride[1:], dilation=dilation[1:], **kw)
        self.kernel_size_temporal, self.stride_temporal, self.dilation_temporal = kernel_size[0], stride[0], dilation[0]
        self.internal_state = None

    @property
    def temporal_footprint(self):
        return 1 + (self.kernel_size_temporal - 1) * self.dilation_temporal - self.stride_temporal

    def forward(self, x):
        if self.internal_state is None:
            self.internal_state = torch.cat(self.temporal_footprint * [torch.zeros_like(x[0:1])])
        x = torch.cat([self.internal_state, x])
        self.internal_state = x[-self.temporal_footprint:]
        k, n = self.kernel_size_temporal, x.shape[0]
        x = torch.cat([x[o:n - k + o + 1] for o in range(k)], dim=1)
        x = x[torch.arange(0, x.shape[0], self.stride_temporal, device=x.device)]
        return super().forward(x)

    def reset(self):
        self.internal_state = None
        return self

    def train(self, mode=True):
        super().train(mode)
        return self.reset()


class _ConvReLU(nn.Sequential):
    def __init__(self, i, o, k=3, stride=1, groups=1, convlayer=nn.Conv2d, padding=None):
        padding = padding if padding is not None else (k - 1) // 2
        super().__init__(convlayer(i, o, k, stride, padding=padding, groups=groups), nn.ReLU6(inplace=True))


class _IR(nn.Module):
    def __init__(self, i, o, k=3, s=1, e=1, tshift=False, tstride=False):
        super().__init__()
        hid = round(i * e)
        self.use_residual = s == 1 and i == o
        self.tstride = tstride
        layers = []
        if e != 1:
            st = 1 if not tstride else (2, 1, 1)
            cl, ks = (SteppableConv3dAs2d, (3, 1, 1)) if tshift else (nn.Conv2d, 1)
            layers.append(_ConvReLU(i, hid, ks, st, padding=0, convlayer=cl))
        layers += [_ConvReLU(hid, hid, k, s, groups=hid), nn.Conv2d(hid, o, 1, 1, 0)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            n = out.shape[0]
            out += x[[-1 - 2 * i for i in range(n)][::-1]] if self.tstride else x[-n:]
        return out


class FitCoachCNN(nn.Module):
    TASKS = {"main": (0, 2965), "count": (2965, 3014), "vis": (3014, 3026), "orient": (3026, 3031)}

    def __init__(self):
        super().__init__()
        I = _IR
        self.cnn = nn.Sequential(
            _ConvReLU(3, 32, 3, stride=2), I(32, 24, 3, 1), I(24, 32, 3, 2, 6), I(32, 32, 3, 1, 6, True),
            I(32, 32, 3, 1, 6), I(32, 32, 3, 1, 6), I(32, 56, 5, 2, 6), I(56, 56, 5, 1, 6, True, True),
            I(56, 56, 5, 1, 6), I(56, 56, 5, 1, 6), I(56, 112, 3, 2, 6), I(112, 112, 3, 1, 6, True),
            I(112, 112, 3, 1, 6), I(112, 112, 3, 1, 6), I(112, 112, 3, 1, 6, True, True), I(112, 112, 3, 1, 6),
            I(112, 160, 5, 1, 6), I(160, 160, 5, 1, 6, True), I(160, 160, 5, 1, 6), I(160, 160, 5, 1, 6),
            I(160, 160, 5, 1, 6, True), I(160, 160, 5, 1, 6), I(160, 272, 5, 2, 6), I(272, 272, 5, 1, 6, True),
            I(272, 272, 5, 1, 6), I(272, 272, 5, 1, 6, True), I(272, 272, 5, 1, 6), I(272, 272, 5, 1, 6),
            I(272, 272, 5, 1, 6), I(272, 272, 5, 1, 6), I(272, 448, 3, 1, 6), _ConvReLU(448, 1280, 1))
        self.classifier = nn.Linear(1280, 3031)

    def forward(self, frames):
        return self.classifier(self.cnn(frames).mean(-1).mean(-1))

    def reset(self):
        for m in self.modules():
            if isinstance(m, SteppableConv3dAs2d):
                m.reset()


def load_fitcoach(weights_dir, device="cpu"):
    ckpt = os.path.join(weights_dir, "efficientnet4Lite_1.8.3.checkpoint")
    lmap = os.path.join(weights_dir, "efficientnet4Lite_1.8.3_label2int.json")
    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m = FitCoachCNN()
    src = [(k, v) for k, v in sd.items() if k.startswith("features.")]
    dst = m.cnn.state_dict()
    assert len(src) == len(dst), f"backbone tensors: checkpoint {len(src)} vs model {len(dst)}"
    new = {}
    for (dk, dv), (sk, sv) in zip(dst.items(), src):
        assert dv.shape == sv.shape, f"shape mismatch {dk} vs {sk}"
        new[dk] = sv
    m.cnn.load_state_dict(new)
    m.classifier.weight.data.copy_(sd["classifier.1.weight"])
    m.classifier.bias.data.copy_(sd["classifier.1.bias"])
    m.to(device).eval()
    labels = json.load(open(lmap))["global"]
    return m, labels


def cnn_frame(frame_bgr):
    """FitCoach preprocessing: RGB, letterbox to portrait 1.4 (keeps the whole body), 224x160, 0..1."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h / w < 1.4:
        nh = int(round(w * 1.4)); t = (nh - h) // 2
        rgb = cv2.copyMakeBorder(rgb, t, nh - h - t, 0, 0, cv2.BORDER_CONSTANT, value=0)
    elif h / w > 1.4:
        nh = int(w * 1.4); o = (h - nh) // 2
        rgb = rgb[o:o + nh]
    rgb = cv2.resize(rgb, (160, 224), interpolation=cv2.INTER_LINEAR)
    return (np.float32(rgb) / 255.0).transpose(2, 0, 1)


@torch.no_grad()
def cnn_step_probs(model, frames_bgr, device="cpu"):
    """frames at 16 fps -> (n_steps, 3031) per-task softmax probabilities, step end times (s)."""
    model.reset()
    if not frames_bgr:
        return np.zeros((0, 3031), np.float32), np.zeros(0)
    x0 = torch.from_numpy(cnn_frame(frames_bgr[0]))[None].to(device)
    for _ in range(3):
        model(x0.repeat(4, 1, 1, 1))                         # fill the temporal memory (first frame x12)
    P, T = [], []
    for i in range(0, len(frames_bgr) - 3, 4):
        x = torch.from_numpy(np.stack([cnn_frame(f) for f in frames_bgr[i:i + 4]])).to(device)
        lg = model(x)[0].float().cpu().numpy().astype(np.float64)
        p = np.empty_like(lg)
        for a, b in FitCoachCNN.TASKS.values():
            z = np.exp(lg[a:b] - lg[a:b].max()); p[a:b] = z / z.sum()
        P.append(p.astype(np.float32)); T.append((i + 4) / FPS)
    return np.array(P), np.array(T)


def cnn_label_index(labels):
    """Per exercise: indices of its QEVD main-task labels, with the attribute tag of each."""
    out = {}
    for ex, names in QEVD_NAMES.items():
        out[ex] = [(i, n.split(" - ", 1)[1].strip().lower()) for n, i in labels.items()
                   if " - " in n and n.split(" - ", 1)[0] in names and i < 2965]
    return out


def cnn_rep_probs(step_p, step_t, t0, t1, ex, label_idx, min_mass=0.25):
    """Average the CNN over a rep's time span. Returns {fault: (p, weight)} or None if the CNN
    doesn't recognise the exercise. weight shrinks when the CNN has no opinion about that fault."""
    if len(step_t) == 0:
        return None
    sel = (step_t >= t0) & (step_t <= t1 + 0.3)
    if not sel.any():
        sel = np.zeros(len(step_t), bool)
        sel[int(np.argmin(np.abs(step_t - (t0 + t1) / 2)))] = True
    p = step_p[sel].mean(0)
    idx = label_idx[ex]
    mass = float(sum(p[i] for i, _ in idx))
    if mass < min_mass:
        return None
    share = {a: p[i] / mass for i, a in idx}
    clean = sum(v for a, v in share.items() if re.search(QEVD_CLEAN_RX, a))
    out = {}
    for f in FAULTS[ex]:
        rx = QEVD_FAULT_RX[ex].get(f)
        if not rx:
            continue
        pos = sum(v for a, v in share.items() if re.search(rx, a))
        neg = clean + sum(v for a, v in share.items() if QEVD_NEG_RX[ex].get(f) and re.search(QEVD_NEG_RX[ex][f], a))
        out[f] = (float(pos / max(pos + neg, 1e-6)), float(min(1.0, (pos + neg) / 0.3)))
    out["_mass"] = mass
    return out


# =====================================================================================
# 7. Fusion and scoring
# =====================================================================================
FUSION_W = {"rule": 0.45, "gru": 0.35, "cnn": 0.20}


def fuse(rule_p, gru_p, cnn_p, ex, weights=FUSION_W):
    """Each judge: {fault: p or None} (cnn: {fault: (p, w)}). Returns {fault: (fused p, judges used)}.
    v5.1: weights may carry "rule_by_fault" = {fault: rule weight} (tuned on validation people)."""
    out = {}
    rbf = weights.get("rule_by_fault", {})
    for k, f in enumerate(FAULTS[ex]):
        parts = []
        if rule_p and rule_p.get(f) is not None:
            parts.append(("rule", rule_p[f], rbf.get(f, weights["rule"])))
        if gru_p is not None and gru_p.get(f) is not None:
            parts.append(("gru", gru_p[f], weights["gru"]))
        if cnn_p and f in cnn_p:
            parts.append(("cnn", cnn_p[f][0], weights["cnn"] * cnn_p[f][1]))
        W = sum(w for _, _, w in parts)
        out[f] = (float(sum(p * w for _, p, w in parts) / W) if W > 0 else None, [n for n, _, w in parts if w > 0])
    return out


def rep_verdict(fused, ex, thresholds, p_correct=None, level1=None):
    """Two levels like v4.
    v5 (level1=None): specific faults above their cut-off; if none fires but the BiGRU's P(correct) is below
      its cut-off ("_correct"), the rep is incorrect with an unspecified 'form_issue'.
    v5.1 (level1={"correct_threshold": t, "veto": v}): level 1 is decided by the BiGRU's correct head
      (P(correct) >= t), overruled only by a fault that is very sure (p >= max(fault cut-off, v)).
      Incorrect reps list the faults above their cut-off (or 'form_issue' if none); correct reps list
      weaker detections as 'hints' instead of failing the rep."""
    th = thresholds.get(ex, {})
    above = sorted([(f, p) for f, (p, _) in fused.items() if p is not None and p >= th.get(f, 0.5)],
                   key=lambda t: -t[1])
    score = 100.0
    for f, (p, _) in fused.items():
        if p is not None:
            score *= 1 - SEVERITY.get(f, 0.5) * float(np.clip((p - 0.15) / 0.85, 0, 1))
    if p_correct is not None:
        score *= 1 - SEVERITY["form_issue"] * float(np.clip((0.85 - p_correct) / 0.85, 0, 1))
    if level1 is None:
        faults = list(above)
        if p_correct is not None and not faults and p_correct < th.get("_correct", 0.5):
            faults = [("form_issue", float(1 - p_correct))]
        return dict(correct=not faults, faults=faults, hints=[], score=int(round(score)))
    veto = any(p >= max(th.get(f, 0.5), level1["veto"]) for f, p in above)
    ok = (p_correct is None or p_correct >= level1["correct_threshold"]) and not veto
    if ok:
        return dict(correct=True, faults=[], hints=above, score=int(round(score)))
    faults = above or [("form_issue", float(1 - (p_correct or 0.0)))]
    return dict(correct=False, faults=faults, hints=[], score=int(round(score)))


# =====================================================================================
# 8. Video in / pose / full analysis / rendering
# =====================================================================================
def decode_video(path, fps=FPS, max_side=640, max_seconds=None):
    """Frames resampled to `fps`, long side <= max_side. Returns frames (BGR list), info."""
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS)
    src = src if 1 < src < 240 else 30.0
    step, nxt, i, frames, H, W = src / fps, 0.0, 0, [], 0, 0
    while cap.grab():
        if i + 1e-6 >= nxt:
            ok, fr = cap.retrieve()
            if not ok:
                break
            H, W = fr.shape[:2]
            sc = min(1.0, max_side / max(H, W))
            small = cv2.resize(fr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else fr
            while i + 1e-6 >= nxt:
                frames.append(small); nxt += step
            if max_seconds and len(frames) >= max_seconds * fps:
                break
        i += 1
    cap.release()
    return frames, dict(W=W, H=H, src_fps=src)


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ar = lambda z: (z[2] - z[0]) * (z[3] - z[1])
    return inter / (ar(a) + ar(b) - inter + 1e-6)


def yolo_precision_kw(device):
    """fp16 on GPU; newer ultralytics renamed half=True to quantize=16 (same check as v4)."""
    if not str(device).startswith("cuda"):
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
    except Exception:
        return {"half": True}


def pose_frames(yolo, frames, device="cpu", imgsz=640, batch=16, conf=0.25):
    """YOLO-pose on frames (same person tracked by IoU). Returns K (T,17,2) pixels of the given frames, S (T,17)."""
    n = len(frames)
    K = np.full((n, 17, 2), np.nan, np.float32)
    S = np.zeros((n, 17), np.float32)
    box = None
    kw = dict(device=device, imgsz=imgsz, conf=conf, verbose=False, **yolo_precision_kw(device))
    for j in range(0, n, batch):
        for t, r in enumerate(yolo.predict(frames[j:j + batch], **kw), j):
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None or len(r.keypoints) == 0:
                continue
            bx = r.boxes.xyxy.float().cpu().numpy()
            if box is not None and max(_iou(b, box) for b in bx) > 0.1:
                k = int(np.argmax([_iou(b, box) for b in bx]))
            else:
                k = int(np.argmax((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])))
            box = bx[k]
            K[t] = r.keypoints.xy[k].float().cpu().numpy()
            S[t] = r.keypoints.conf[k].float().cpu().numpy() if r.keypoints.conf is not None else 1.0
    return K, S


def analyze_reps(xy, cf, ex, gru, rule_params, thresholds, cnn=None, device="cpu", weights=FUSION_W, level1=None):
    """Judge every rep of a cleaned clip. cnn = (step_p, step_t, label_idx) or None."""
    reps, sig = count_reps(xy, ex)
    windows = reps if reps else [fallback_window(xy, ex)]
    faults = FAULTS[ex]
    pad = 0 if ex in WINDOW_EX else 2
    xr, cr = zip(*[rep_tensor(xy, cf, r, pad=pad) for r in windows])
    gc, gf = gru_probs(gru, np.stack(xr), np.stack(cr), device) if gru is not None else (None, None)
    out = []
    for i, r in enumerate(windows):
        m = rep_metrics(xy, cf, r, ex)
        rp = rule_probs(m, ex, rule_params)
        gp = {f: float(gf[i, k]) for k, f in enumerate(faults)} if gf is not None else None
        cp = None
        if cnn is not None:
            cp = cnn_rep_probs(cnn[0], cnn[1], r[0] / FPS, r[2] / FPS, ex, cnn[2])
        fu = fuse(rp, gp, cp, ex, weights)
        v = rep_verdict(fu, ex, thresholds, None if gc is None else float(gc[i]), level1)
        out.append(dict(rep=i + 1, start=int(r[0]), low=int(r[1]), end=int(r[2]), counted=bool(reps), metrics=m,
                        rule=rp, gru=gp, gru_correct=None if gc is None else float(gc[i]),
                        cnn={f: v_[0] for f, v_ in cp.items() if f != "_mass"} if cp else None,
                        fused={f: p for f, (p, _) in fu.items()}, judges={f: j for f, (_, j) in fu.items()}, **v))
    return out, sig


_COL = {"ok": (90, 200, 90), "bad": (60, 60, 235), "warn": (40, 180, 240), "text": (240, 240, 240),
        "dim": (150, 150, 150)}


def _txt(img, s, x, y, col=_COL["text"], sc=0.5, th=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def render(frames, K, S, ex, reps, out_path, title="", fps=FPS, height=480, jumps=None):
    """Skeleton + live rep panel, H.264 mp4 that plays in browsers. jumps: frame indices (jump rope counter)."""
    h0, w0 = frames[0].shape[:2]
    sc = height / h0
    W, H = int(round(w0 * sc / 2) * 2), int(round(h0 * sc / 2) * 2)
    panel_w = 320
    tmp = out_path + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + panel_w, H))
    for t, fr in enumerate(frames):
        img = cv2.resize(fr, (W, H))
        cur = next((r for r in reps if r["start"] <= t <= r["end"]), None)
        done = [r for r in reps if r["end"] < t]
        col = _COL["ok"] if cur is None or cur["correct"] else _COL["bad"]
        if np.isfinite(K[t]).all() and S[t].max() > 0:
            p = K[t] * sc
            for a, b in EDGES:
                if S[t, a] > CONF_THR and S[t, b] > CONF_THR:
                    cv2.line(img, tuple(int(v) for v in p[a]), tuple(int(v) for v in p[b]), col, 2, cv2.LINE_AA)
            for j in range(17):
                if S[t, j] > CONF_THR:
                    cv2.circle(img, tuple(int(v) for v in p[j]), 3, (255, 255, 255), -1, cv2.LINE_AA)
        pn = np.full((H, panel_w, 3), 30, np.uint8)
        _txt(pn, title[:34], 10, 22, _COL["dim"], 0.45)
        _txt(pn, ex.upper(), 10, 50, _COL["text"], 0.7, 2)
        if ex == "jump_rope":
            _txt(pn, f"Jumps: {sum(1 for j in (jumps or []) if j <= t)}", 10, 82, _COL["ok"], 0.8, 2)
        elif ex == "plank":
            _txt(pn, f"Hold: {t / fps:.1f} s", 10, 82, _COL["ok"], 0.8, 2)
        else:
            n_counted = sum(1 for r in done if r["counted"])
            _txt(pn, f"Reps: {n_counted}", 10, 82, _COL["ok"], 0.8, 2)
        last = cur or (done[-1] if done else None)
        y = 112
        if last is not None:
            unit = "2 s" if ex in WINDOW_EX else "rep"
            tag = f"current {unit}" if cur is not None else f"{unit} {last['rep']}"
            _txt(pn, f"{tag}: score {last['score']}", 10, y, _COL["text"], 0.55)
            y += 24
            if last["correct"]:
                _txt(pn, "Good form", 10, y, _COL["ok"], 0.6, 2); y += 24
                for f, p in last.get("hints", [])[:2]:
                    _txt(pn, f"minor: {f.replace('_', ' ')} {p:.2f}", 10, y, _COL["warn"], 0.45); y += 18
            for f, p in last["faults"][:4]:
                _txt(pn, f"- {f.replace('_', ' ')} {p:.2f}", 10, y, _COL["bad"], 0.5); y += 20
                words, line = FAULT_TEXT.get(f, "").split(), ""
                for w_ in words + [None]:
                    if w_ is None or len(line) + len(w_) + 1 > 40:
                        if line:
                            _txt(pn, line, 18, y, _COL["dim"], 0.38); y += 16
                        line = w_ or ""
                    else:
                        line = (line + " " + w_).strip()
                y += 4
            v = last["metrics"]["view"]
            _txt(pn, f"view: {v}", 10, H - 40, _COL["dim"], 0.42)
        x0 = 10
        for r in done[-10:]:
            c = _COL["ok"] if r["correct"] else _COL["bad"]
            cv2.rectangle(pn, (x0, H - 22), (x0 + 22, H - 8), c, -1)
            x0 += 27
        vw.write(np.hstack([img, pn]))
    vw.release()
    ff = "ffmpeg"
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "30", "-preset", "veryfast", "-movflags", "+faststart", out_path],
                       capture_output=True, text=True)
    if r.returncode == 0:
        os.remove(tmp)
    else:
        os.replace(tmp, out_path)
    return out_path


class Pipeline:
    """Full inference: video -> pose -> reps -> rules + GRU + 3D CNN -> fused per-rep verdicts."""

    def __init__(self, bundle_dir, device=None, pose_weights=None, use_cnn=None, calibration="auto"):
        """calibration: "auto" = calibration_v51.json if present, else the Kaggle v5 settings;
        "v5" = exactly the Kaggle run (3D CNN on); a path = that calibration file.
        use_cnn: None = what the calibration says (v5.1: off, v5: on)."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.load(open(os.path.join(bundle_dir, "pipeline_config.json")))
        self.cfg = cfg
        self.rule_params = cfg["rule_params"]
        self.thresholds = cfg["fusion_thresholds"]
        self.weights, self.level1, self.calibration = dict(FUSION_W), {}, "v5"
        cal_path = os.path.join(bundle_dir, "calibration_v51.json") if calibration == "auto" else calibration
        if cal_path not in (None, "v5") and os.path.exists(cal_path):
            cal = json.load(open(cal_path))
            self.thresholds = {ex: {**self.thresholds.get(ex, {}), **t} for ex, t in cal["fusion_thresholds"].items()}
            self.rule_by_fault = cal["rule_weight"]            # per exercise {fault: rule weight}
            self.level1 = cal["level1"]                        # per exercise {"correct_threshold", "veto"}
            self.calibration = cal.get("version", "v5.1")
            use_cnn = cal.get("use_cnn", False) if use_cnn is None else use_cnn
        elif calibration not in ("auto", "v5"):
            raise FileNotFoundError(cal_path)
        use_cnn = True if use_cnn is None else use_cnn
        self.gru = {}
        for ex in EXERCISES:
            p = os.path.join(bundle_dir, "models", ex, "pose_gru.pt")
            if os.path.exists(p):
                m = PoseGRU(len(FAULTS[ex]))
                m.load_state_dict(torch.load(p, map_location="cpu"))
                self.gru[ex] = m.to(self.device).eval()
        from ultralytics import YOLO
        w = pose_weights or cfg["pose_weights"]
        local = os.path.join(bundle_dir, w)
        self.yolo = YOLO(local if os.path.exists(local) else w)      # bundled file, else ultralytics downloads it
        self.cnn = None
        if use_cnn and os.path.isdir(os.path.join(bundle_dir, "fitcoach_3dcnn")):
            self.cnn, labels = load_fitcoach(os.path.join(bundle_dir, "fitcoach_3dcnn"), self.device)
            self.cnn_idx = cnn_label_index(labels)

    def __call__(self, video_path, exercise, render_to=None, title="", max_seconds=None):
        frames, info = decode_video(video_path, max_seconds=max_seconds)
        if not frames:
            return dict(video=video_path, exercise=exercise, error="could not read any frames", reps=[])
        K, S = pose_frames(self.yolo, frames, device=self.device, imgsz=self.cfg.get("pose_imgsz", 640))
        present = np.isfinite(K).all(-1).any(-1) & (S.max(-1) > 0)
        if present.mean() < 0.3:
            return dict(video=video_path, exercise=exercise, error="no person found in most frames", reps=[])
        xy, cf = clean_clip(K, S)
        cnn = None
        if self.cnn is not None:
            sp, st = cnn_step_probs(self.cnn, frames, self.device)
            cnn = (sp, st, self.cnn_idx)
        w = dict(self.weights)
        if self.level1:
            w["rule_by_fault"] = self.rule_by_fault.get(exercise, {})
        reps, _ = analyze_reps(xy, cf, exercise, self.gru.get(exercise), self.rule_params, self.thresholds,
                               cnn, self.device, weights=w, level1=self.level1.get(exercise))
        res = dict(video=video_path, exercise=exercise, seconds=len(frames) / FPS, calibration=self.calibration,
                   used_cnn=self.cnn is not None, reps_counted=sum(r["counted"] for r in reps), reps=reps,
                   unit="window" if exercise in WINDOW_EX else "rep")
        jumps = count_jumps(xy) if exercise == "jump_rope" else None
        if jumps is not None:
            res["jumps"] = len(jumps)
        if exercise == "plank":
            res["hold_seconds"] = round(float(present.sum()) / FPS, 1)
        if render_to:
            res["rendered"] = render(frames, K, S, exercise, reps, render_to, title=title, jumps=jumps)
        return res


# =====================================================================================
# 9. Training helpers (used by the notebook)
# =====================================================================================
def load_ec3d(pickle_path):
    """EC3D data_3D.pickle -> list of sequences (one rep each). Restricted unpickler: numpy arrays only."""
    import pickle
    allowed = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar")}

    class _Safe(pickle.Unpickler):
        def find_class(self, mod, name):
            if (mod, name) in allowed:
                return super().find_class(mod, name)
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")

    with open(pickle_path, "rb") as f:
        d = _Safe(f).load()
    P = np.asarray(d["poses"], np.float32)
    seqs = {}
    for k, r in enumerate(d["labels"]):
        seqs.setdefault((str(r[0]), str(r[1]), int(r[2]), int(r[3])), []).append(k)
    out = []
    for (act, sub, code, rep), idx in seqs.items():
        idx = np.array(idx)[np.argsort([int(d["labels"][k][4]) for k in idx])]
        out.append(dict(act=act, subject=sub, code=code, rep=rep, poses=P[idx]))
    return out


def ec3d_samples(seqs, yaws=tuple(range(0, 360, 30)), pitch=10.0, seed=0, val_subjects=("Vidit",)):
    """Every EC3D rep seen from many camera angles -> per-rep training samples with labels."""
    rng = np.random.RandomState(seed)
    out = []
    for s in seqs:
        ex = EC3D_ACT.get(s["act"])
        t = ec3d_targets(ex, s["code"]) if ex else None
        if t is None:
            continue
        y, known = t
        fixed = ec3d_upright_fix(s["poses"])
        for yaw in yaws:
            yj, pj = yaw + rng.uniform(-12, 12), rng.uniform(-pitch, pitch)
            xy = ec3d_project(fixed, yj, pj)
            cf = np.ones(xy.shape[:2], np.float32)
            rep = (0, 0, len(xy) - 1)
            xr, cr = rep_tensor(xy, cf, rep, pad=0)
            out.append(dict(ex=ex, src="ec3d", split="val" if s["subject"] in val_subjects else "train",
                            group=f"ec3d_{s['subject']}", clip=f"ec3d_{s['act']}_{s['subject']}_{s['code']}_{s['rep']}",
                            yaw=float(yj), xy=xr, cf=cr, metrics=rep_metrics(xy, cf, rep, ex), y=y, known=known,
                            correct=bool(s["code"] == 1), correct_known=True))
    return out


def clip_samples(xy, cf, ex, y, known, **meta):
    """One QEVD clip -> one sample per detected rep (whole clip if no rep is found). Clip labels go to every rep."""
    reps, _ = count_reps(xy, ex, stride=WIN_STRIDE_TRAIN)
    wins = reps if reps else [fallback_window(xy, ex)]
    out = []
    for r in wins:
        xr, cr = rep_tensor(xy, cf, r, pad=0 if ex in WINDOW_EX else 2)
        out.append(dict(ex=ex, src="qevd", xy=xr, cf=cr, metrics=rep_metrics(xy, cf, r, ex), y=y, known=known,
                        rep=tuple(int(v) for v in r), counted=bool(reps), **meta))
    return out


def _stack(samples):
    X = torch.from_numpy(np.stack([s["xy"] for s in samples]).astype(np.float32))
    C = torch.from_numpy(np.stack([s["cf"] for s in samples]).astype(np.float32))
    Y = torch.from_numpy(np.stack([s["y"] for s in samples]).astype(np.float32))
    M = torch.from_numpy(np.stack([s["known"] for s in samples]).astype(np.float32))
    YC = torch.tensor([float(s["correct"]) for s in samples])
    MC = torch.tensor([float(s["correct_known"]) for s in samples])
    return X, C, Y, M, YC, MC


@torch.no_grad()
def predict_gru(model, samples, device, chunk=2048):
    if not samples:
        return np.zeros(0), np.zeros((0, model.fault_head.out_features))
    X, C = _stack(samples)[:2]
    pc, pf = [], []
    for i in range(0, len(X), chunk):
        a, b = gru_probs(model, X[i:i + chunk].to(device), C[i:i + chunk].to(device), device)
        pc.append(a); pf.append(b)
    return np.concatenate(pc), np.concatenate(pf)


def fault_ap(y, known, p, min_pos=3):
    """Per-fault average precision over the known labels; NaN when too few positives/negatives."""
    from sklearn.metrics import average_precision_score
    out = []
    for k in range(y.shape[1]):
        sel = known[:, k] > 0
        yk = y[sel, k]
        out.append(average_precision_score(yk, p[sel, k]) if yk.sum() >= min_pos and (1 - yk).sum() >= min_pos
                   else np.nan)
    return np.array(out)


def train_gru(train, val, ex, device, epochs=80, patience=15, batch=256, lr=2e-3, seed=42, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    from sklearn.metrics import roc_auc_score
    K = len(FAULTS[ex])
    Xt, Ct, Yt, Mt, corr_t, has_t = [t.to(device) for t in _stack(train)]   # correct label = v4 level 1
    pos, neg = (Yt * Mt).sum(0), ((1 - Yt) * Mt).sum(0)
    pw = (neg / pos.clamp(min=1)).clamp(1, 20)
    n_c, n_i = float((corr_t * has_t).sum()), float(((1 - corr_t) * has_t).sum())
    pw_c = torch.tensor(float(np.clip(n_i / max(n_c, 1), 1, 20)), device=device)   # the rarer side gets more weight
    if n_c > n_i:
        pw_c = torch.tensor(1.0, device=device)
    vc = np.array([float(s["correct"]) for s in val]) if val else None
    vck = np.array([bool(s["correct_known"]) for s in val]) if val else None
    model = PoseGRU(K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    n = len(Xt)
    per_ep = max(1, math.ceil(n / batch)); steps = epochs * per_ep; warm = 2 * per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    yv = np.stack([s["y"] for s in val]) if val else None
    mv = np.stack([s["known"] for s in val]) if val else None
    best, best_ep, best_state, hist, t0 = -float("inf"), 0, None, [], time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xy, cf = augment(Xt[idx], Ct[idx])
            lc, lf, _ = model(features_torch(xy, cf))
            l_f = (F.binary_cross_entropy_with_logits(lf, Yt[idx], pos_weight=pw, reduction="none") * Mt[idx]).sum() \
                / Mt[idx].sum().clamp(min=1)
            l_c = (F.binary_cross_entropy_with_logits(lc, corr_t[idx], pos_weight=pw_c, reduction="none")
                   * has_t[idx]).sum() / has_t[idx].sum().clamp(min=1)
            loss = l_f + 0.5 * l_c
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        auc = np.nan
        if val:
            pcv, pf = predict_gru(model, val, device)
            aps = fault_ap(yv, mv, pf)
            if vck.sum() and 0 < vc[vck].sum() < vck.sum():
                auc = float(roc_auc_score(vc[vck], pcv[vck]))
            parts = [v for v in (float(np.nanmean(aps)) if np.isfinite(aps).any() else np.nan, auc) if np.isfinite(v)]
            score = float(np.mean(parts)) if parts else -float(tot / nb)          # v4-style: mean of the two levels
        else:
            aps, score = np.full(K, np.nan), -float(tot / nb)
        hist.append(dict(epoch=ep, loss=tot / nb, val_score=score, val_correct_auc=auc,
                         **{f"AP_{f}": a for f, a in zip(FAULTS[ex], aps)}))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep - best_ep >= patience or ep == epochs - 1:
            log(f"{ex} ep {ep:3d} loss {tot / nb:.3f} val score {score:.3f} (correct AUC {auc:.3f}) | best {best:.3f} @ep{best_ep} "
                f"| {(time.time() - t0) / (ep + 1):.1f} s/ep")
        if ep - best_ep >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def judge_records(samples, ex, gru_pf, rule_params, cnn_by_clip=None):
    """Per-sample probabilities of every judge (rule / gru / cnn) for the exercise's faults."""
    recs = []
    for i, s in enumerate(samples):
        rp = rule_probs(s["metrics"], ex, rule_params)
        gp = {f: float(gru_pf[i, k]) for k, f in enumerate(FAULTS[ex])} if gru_pf is not None else None
        cp = None
        if cnn_by_clip is not None and s.get("clip") in cnn_by_clip and s.get("rep") is not None:
            sp, st, idx = cnn_by_clip[s["clip"]]
            cp = cnn_rep_probs(sp, st, s["rep"][0] / FPS, s["rep"][2] / FPS, ex, idx)
        recs.append(dict(rule=rp, gru=gp, cnn=cp))
    return recs


def tune_fusion_thresholds(records, samples, ex, grid=np.linspace(0.3, 0.8, 26), min_pos=5):
    """Per fault: the fused-probability cut-off with the best F1 on validation (0.5 if too few positives)."""
    from sklearn.metrics import f1_score
    out, report = {}, []
    fused = [fuse(r["rule"], r["gru"], r["cnn"], ex) for r in records]
    for k, f in enumerate(FAULTS[ex]):
        p = np.array([fu[f][0] if fu[f][0] is not None else np.nan for fu in fused])
        y = np.array([s["y"][k] for s in samples])
        kn = np.array([s["known"][k] for s in samples]) > 0
        sel = kn & np.isfinite(p)
        if y[sel].sum() >= min_pos and (1 - y[sel]).sum() >= min_pos:
            f1s = [f1_score(y[sel], p[sel] >= g, zero_division=0) for g in grid]
            out[f] = float(grid[int(np.argmax(f1s))])
            report.append(dict(exercise=ex, fault=f, threshold=out[f], val_F1=float(max(f1s)),
                               n_pos=int(y[sel].sum()), n=int(sel.sum())))
        else:
            out[f] = 0.5
            report.append(dict(exercise=ex, fault=f, threshold=0.5, val_F1=np.nan, n_pos=int(y[sel].sum()),
                               n=int(sel.sum())))
    return out, report


def tune_correct_threshold(p_correct, samples, ex, grid=np.linspace(0.2, 0.9, 71)):
    """v4 level 1: the P(correct) cut-off that balances the two kinds of error on validation."""
    from sklearn.metrics import balanced_accuracy_score
    yc = np.array([float(s["correct"]) for s in samples])
    kn = np.array([bool(s["correct_known"]) for s in samples])
    if kn.sum() and 0 < yc[kn].sum() < kn.sum():
        ba = [balanced_accuracy_score(yc[kn], p_correct[kn] >= g) for g in grid]
        g = float(grid[int(np.argmax(ba))])
        return g, dict(exercise=ex, fault="_correct (level 1)", threshold=g, val_bal_acc=float(max(ba)),
                       n_pos=int(yc[kn].sum()), n=int(kn.sum()))
    return 0.5, dict(exercise=ex, fault="_correct (level 1)", threshold=0.5, val_bal_acc=np.nan,
                     n_pos=int(yc[kn].sum()), n=int(kn.sum()))
