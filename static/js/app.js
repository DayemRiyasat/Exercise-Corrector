// static/js/app.js
// Analyser page controller, with two engines behind it:
//   v5 (every exercise): the Exercise-Corrector pipeline runs on the server
//     via v5_client.js. It finds the pose itself and returns one verdict per
//     finished rep, or per 2 second window for jump rope and plank.
//   legacy (no exercise today): MediaPipe pose runs in the browser via
//     pose_client.js and the landmarks are classified frame by frame. That
//     file is only imported if an exercise needs it, since it loads from a CDN.
// Each finished set is posted to the sessions API.

import { loadModel, resetCounter, postSession, fetchStats } from './api.js';

import {
  drawSkeleton17, sendLiveFrame, analyzeVideoFile, analyzeImageFile
} from './v5_client.js';

// ---- Config ----
const ANALYZE_INTERVAL_MS = 150;   // throttle server calls (~6-7 per second)

// Standard MediaPipe pose skeleton (33-landmark indices)
const POSE_CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32]
];

// ---- State ----
let currentExercise = null;
let poseReady = false;
let webcamStream = null;
let webcamRunning = false;
let videoRunning = false;
let videoPaused = false;
let lastAnalyze = 0;
let lastTimestamp = 0;
let timerId = null;
let faultLabels = {};
let currentEngine = 'legacy';
let currentUnit = 'rep';
let pose = null;                   // pose_client.js, once a legacy exercise needs it

// v5 pipeline state: the chosen files, and the judged clip being played back.
const v5 = { videoFile: null, imageFile: null, result: null, revealed: 0 };
const isV5 = () => currentEngine === 'v5';
const isWindow = () => currentUnit === 'window';
const unitWord = (n) => `${isWindow() ? 'window' : 'rep'}${n === 1 ? '' : 's'}`;

const stats = { good: 0, bad: 0, total: 0, prevRep: 0, faults: {} };
const session = { active: false, mode: null, exercise: null, startedAt: null, startMs: 0 };

// ---- Helpers ----
const $ = (id) => document.getElementById(id);
const show = (el) => el && el.classList.remove('hidden');
const hide = (el) => el && el.classList.add('hidden');
const toast = (msg, icon) => window.shapeform && window.shapeform.toast(msg, icon);

function repCountFrom(repInfo) {
  if (!repInfo) return null;
  for (const key of ['rep_count', 'count', 'reps']) {
    if (typeof repInfo[key] === 'number') return repInfo[key];
  }
  return null;
}

function setLoading(on, msg) {
  const overlay = $('loading-overlay');
  if (msg) $('loading-message').textContent = msg;
  on ? show(overlay) : hide(overlay);
}

function formatClock(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

// ---- Lifetime totals + personal bests -------------------------------------
async function refreshTotals() {
  const res = await fetchStats();
  if (!res.success) return;
  const s = res.stats;

  faultLabels = s.faultLabels || {};
  $('rail-sessions').textContent = s.sessions;
  $('rail-reps').textContent = s.reps;
  $('rail-streak').textContent = s.streak;

  document.querySelectorAll('[data-pb]').forEach(el => {
    const reps = (s.personalBests || {})[el.dataset.pb];
    if (reps) {
      const card = el.closest('.exercise-card');
      const word = card && card.dataset.unit === 'window' ? 'windows' : 'reps';
      el.innerHTML = `<i class="bi bi-trophy-fill"></i>Best ${reps} ${word}`;
      el.classList.remove('hidden');
    } else {
      el.classList.add('hidden');
    }
  });
}

// ---- Session lifecycle -----------------------------------------------------
function startSession(mode) {
  endSession(false);
  session.active = true;
  session.mode = mode;
  session.exercise = currentExercise;
  session.startedAt = new Date().toISOString();
  session.startMs = Date.now();

  show($('stage-live'));
  $('stage-timer').textContent = '00:00';
  clearInterval(timerId);
  timerId = setInterval(() => {
    $('stage-timer').textContent = formatClock((Date.now() - session.startMs) / 1000);
  }, 1000);
}

async function endSession(announce = true) {
  clearInterval(timerId);
  hide($('stage-live'));
  if (!session.active) return;
  session.active = false;

  const durationSec = Math.round((Date.now() - session.startMs) / 1000);
  if (stats.total < 1) {
    if (announce) toast(`No ${unitWord(2)} counted, nothing saved`, 'bi-info-circle-fill');
    return;
  }

  const accuracy = Math.round((stats.good / stats.total) * 100);
  const payload = {
    exercise: session.exercise,
    mode: session.mode,
    startedAt: session.startedAt,
    durationSec,
    reps: stats.total,
    good: stats.good,
    bad: stats.bad,
    faults: { ...stats.faults }
  };

  const res = await postSession(payload);

  if (!res.success) {
    if (announce) toast(res.error || 'Could not save this session', 'bi-exclamation-triangle-fill');
    return;
  }

  refreshTotals();
  if (announce) {
    const top = Object.entries(stats.faults).sort((a, b) => b[1] - a[1])[0];
    const label = top ? (faultLabels[top[0]] || top[0].replace(/_/g, ' ')) : null;
    const note = label ? ` \u00b7 mostly ${label.toLowerCase()}` : '';
    toast(`Saved: ${stats.total} ${unitWord(stats.total)}, ${accuracy}% clean${note}`);
  }
}

$('finish-session').addEventListener('click', async () => {
  if (!session.active && stats.total < 1) {
    toast('Start a set first', 'bi-info-circle-fill');
    return;
  }
  await stopWebcam();
  await stopVideo();
  window.location.href = '/dashboard';
});

// A set in progress is still worth keeping if the tab closes.
window.addEventListener('pagehide', () => {
  if (!session.active || stats.total < 1) return;
  const payload = JSON.stringify({
    exercise: session.exercise,
    mode: session.mode,
    startedAt: session.startedAt,
    durationSec: Math.round((Date.now() - session.startMs) / 1000),
    reps: stats.total, good: stats.good, bad: stats.bad,
    faults: { ...stats.faults }
  });
  // sendBeacon survives the page teardown that would kill a fetch.
  navigator.sendBeacon('/api/sessions', new Blob([payload], { type: 'application/json' }));
  session.active = false;
});

// ---- Status panel ----------------------------------------------------------
const COLOR_MAP = {
  success: '#12C48B', danger: '#FF5C5C',
  warning: '#FFB020', secondary: '#7C8CA3'
};

const ICON_MAP = {
  success: { cls: 'is-good', icon: 'bi-check-circle-fill' },
  danger: { cls: 'is-bad', icon: 'bi-exclamation-octagon-fill' },
  warning: { cls: 'is-warn', icon: 'bi-exclamation-triangle-fill' },
  secondary: { cls: '', icon: 'bi-hourglass-split' }
};

function setStatusIcon(color) {
  const el = $('status-icon');
  const cfg = ICON_MAP[color] || ICON_MAP.secondary;
  el.className = `status-icon ${cfg.cls}`.trim();
  el.innerHTML = `<i class="bi ${cfg.icon}"></i>`;
}

function updateStatus(result) {
  const badge = $('status-badge');
  if (!result || !result.success) {
    badge.textContent = 'No pose';
    badge.style.backgroundColor = COLOR_MAP.secondary;
    $('status-message').textContent =
      (result && result.message) || 'Position yourself in frame';
    $('confidence-fill').style.width = '0%';
    $('confidence-value').textContent = '0%';
    setStatusIcon('secondary');
    return;
  }

  const fb = result.feedback || {};
  badge.textContent = fb.status || result.prediction || '-';
  badge.style.backgroundColor = COLOR_MAP[fb.color] || COLOR_MAP.secondary;
  $('status-message').textContent = fb.message || '';
  setStatusIcon(fb.color);

  const pct = Math.round((result.confidence || 0) * 100);
  $('confidence-fill').style.width = pct + '%';
  $('confidence-value').textContent = pct + '%';

  const tips = $('tips-list');
  if (fb.tips && fb.tips.length) {
    tips.innerHTML = '';
    fb.tips.forEach(t => {
      const li = document.createElement('li');
      li.textContent = t;
      tips.appendChild(li);
    });
  }
}

function updateReps(result, overlayId) {
  const count = repCountFrom(result.rep_info);
  if (count === null) return;

  if (overlayId) {
    const overlay = $(overlayId);
    if (overlay) overlay.querySelector('.rep-number').textContent = count;
  }

  // A completed rep is a rep-count increase. Classify it by the current
  // prediction: 'none' means good form, anything else is a fault.
  if (count > stats.prevRep) {
    const good = result.prediction === 'none';
    for (let i = stats.prevRep; i < count; i++) {
      stats.total += 1;
      if (good) {
        stats.good += 1;
      } else {
        stats.bad += 1;
        const key = result.prediction || 'unknown';
        stats.faults[key] = (stats.faults[key] || 0) + 1;
      }
    }
    stats.prevRep = count;
    renderStats();
  }
}

// ---- Counter overlay -------------------------------------------------------
// Reps for most exercises, jumps for jump rope, held seconds for a plank.
function counterLabel() {
  if (currentExercise === 'jump_rope') return 'Jumps';
  if (currentExercise === 'plank') return 'Hold';
  return 'Reps';
}

function setCounter(overlayId, value) {
  const overlay = $(overlayId);
  if (!overlay) return;
  overlay.querySelector('.rep-number').textContent =
    currentExercise === 'plank' ? `${Math.floor(value || 0)}s` : (value || 0);
}

function applyUnitLabels() {
  ['rep-counter-overlay', 'video-rep-counter-overlay'].forEach(id => {
    $(id).querySelector('.rep-label').textContent = counterLabel();
    setCounter(id, 0);
  });
  $('stat-reps-label').textContent = isWindow() ? 'Windows' : 'Reps';
  $('reset-reps-label').textContent = isWindow() ? 'Reset' : 'Reset reps';
  // v5 reports a 0-100 score per rep rather than a per-frame confidence.
  document.querySelector('.confidence-label').textContent =
    !isV5() ? 'Confidence' : (isWindow() ? 'Window score' : 'Rep score');
}

function renderStats() {
  $('stat-reps').textContent = stats.total;
  $('stat-good').textContent = stats.good;
  $('stat-bad').textContent = stats.bad;
  $('stat-total').textContent = stats.total;
  const acc = stats.total ? Math.round((stats.good / stats.total) * 100) : 0;
  $('stat-accuracy').textContent = acc + '%';
}

function resetStats() {
  stats.good = stats.bad = stats.total = stats.prevRep = 0;
  stats.faults = {};
  renderStats();
}

// ---- Landmark drawing ------------------------------------------------------
function drawSkeleton(canvas, mediaEl, landmarks, showLm) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!showLm || !landmarks) return;

  const w = canvas.width, h = canvas.height;
  const scale = Math.max(1, Math.min(w, h) / 480);

  ctx.strokeStyle = 'rgba(1, 211, 210, .92)';
  ctx.lineWidth = 4 * scale;
  ctx.lineCap = 'round';
  POSE_CONNECTIONS.forEach(([a, b]) => {
    const p = landmarks[a], q = landmarks[b];
    if (!p || !q) return;
    ctx.beginPath();
    ctx.moveTo(p.x * w, p.y * h);
    ctx.lineTo(q.x * w, q.y * h);
    ctx.stroke();
  });

  ctx.fillStyle = '#FFFFFF';
  ctx.strokeStyle = '#0D77E7';
  ctx.lineWidth = 2 * scale;
  landmarks.forEach(p => {
    if ((p.visibility ?? 1) < 0.4) return;
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4 * scale, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  });
}

function sizeCanvasTo(canvas, mediaEl) {
  const w = mediaEl.videoWidth || mediaEl.naturalWidth || mediaEl.clientWidth;
  const h = mediaEl.videoHeight || mediaEl.naturalHeight || mediaEl.clientHeight;
  if (w && h) { canvas.width = w; canvas.height = h; }
}

// ---- Exercise + mode selection ---------------------------------------------
document.querySelectorAll('.exercise-card').forEach(card => {
  if (card.classList.contains('disabled')) return;
  card.addEventListener('click', async () => {
    // Anything running or already judged belongs to the previous exercise.
    await stopWebcam();
    await stopVideo();
    resetStats();
    v5.result = null;
    document.querySelectorAll('.exercise-card')
      .forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    currentExercise = card.dataset.exercise;
    currentEngine = card.dataset.engine || 'legacy';
    currentUnit = card.dataset.unit || 'rep';
    applyUnitLabels();

    $('stage-exercise-name').textContent = card.dataset.name || currentExercise;
    $('stage-exercise-icon').className = `bi ${card.dataset.icon || 'bi-activity'}`;
    $('status-guide').href = `/exercises/${currentExercise}`;
    $('status-guide').querySelector('span').textContent =
      `Read the ${(card.dataset.name || '').toLowerCase()} form guide`;

    setLoading(true, 'Loading model...');
    if (!isV5() && !poseReady) {
      try {
        pose = await import('./pose_client.js');
        await pose.initPose();
        poseReady = true;
      }
      catch (e) { setLoading(false); toast('Could not start pose detection', 'bi-exclamation-triangle-fill'); return; }
    }
    const res = await loadModel(currentExercise);
    setLoading(false);
    if (!res.success) {
      toast(res.error || 'Model failed to load', 'bi-exclamation-triangle-fill');
      return;
    }

    // One photo cannot show jump rope, so the option goes away.
    const photoOk = card.dataset.photo !== 'no';
    $('image-btn').classList.toggle('hidden', !photoOk);
    if (!photoOk && !$('image-mode').classList.contains('hidden')) {
      hide($('image-mode'));
      hide($('analysis-section'));
    }

    show($('input-mode-selection'));
    $('input-mode-selection').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

$('webcam-btn').addEventListener('click', () => switchMode('webcam'));
$('video-btn').addEventListener('click', () => switchMode('video'));
$('image-btn').addEventListener('click', () => switchMode('image'));

async function switchMode(mode) {
  await stopWebcam();
  await stopVideo();
  show($('analysis-section'));
  ['webcam-mode', 'video-mode', 'image-mode'].forEach(id => hide($(id)));
  resetStats();
  updateStatus(null);
  v5.result = null;
  if (mode === 'webcam') show($('webcam-mode'));
  if (mode === 'video') show($('video-mode'));
  if (mode === 'image') show($('image-mode'));
  $('analysis-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- Webcam mode -----------------------------------------------------------
$('start-webcam').addEventListener('click', startWebcam);
$('stop-webcam').addEventListener('click', stopWebcam);
$('reset-reps').addEventListener('click', async () => {
  await resetCounter(currentExercise);
  resetStats();
  setCounter('rep-counter-overlay', 0);
  toast(isWindow() ? 'Counter reset' : 'Rep counter reset', 'bi-arrow-counterclockwise');
});

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (e) {
    toast('Camera access denied', 'bi-camera-video-off-fill');
    return;
  }

  const video = $('webcam');
  video.srcObject = webcamStream;
  await video.play();

  // play() can resolve before the frame dimensions are known, which leaves
  // the overlay canvas at its 300x150 default and puts the skeleton in the
  // wrong place. Wait for the real dimensions before sizing it.
  if (!video.videoWidth) {
    await new Promise(resolve => {
      video.addEventListener('loadedmetadata', resolve, { once: true });
      setTimeout(resolve, 2000);        // never hang if the event is missed
    });
  }
  sizeCanvasTo($('output-canvas'), video);

  webcamRunning = true;
  hide($('start-webcam')); show($('stop-webcam')); show($('reset-reps'));
  resetStats();
  setCounter('rep-counter-overlay', 0);
  await resetCounter(currentExercise);
  startSession('webcam');
  isV5() ? webcamLoopV5() : webcamLoop();
}

async function stopWebcam() {
  const wasRunning = webcamRunning;
  webcamRunning = false;
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  show($('start-webcam')); hide($('stop-webcam')); hide($('reset-reps'));
  if (wasRunning) await endSession(true);
}

async function webcamLoop() {
  if (!webcamRunning) return;
  const video = $('webcam');
  const canvas = $('output-canvas');
  const showLm = $('show-landmarks').checked;

  const now = performance.now();
  if (now - lastAnalyze >= ANALYZE_INTERVAL_MS) {
    lastAnalyze = now;
    const ts = Math.max(now, lastTimestamp + 1);
    lastTimestamp = ts;
    const landmarks = await pose.detectVideo(video, ts);
    drawSkeleton(canvas, video, landmarks, showLm);
    if (landmarks) {
      const result = await pose.analyzeFrame(currentExercise, landmarks);
      updateStatus(result);
      updateReps(result, 'rep-counter-overlay');
    } else {
      updateStatus({ success: false, message: 'Position yourself in frame' });
    }
  }
  requestAnimationFrame(webcamLoop);
}

// ---- v5 pipeline: per-rep verdicts ------------------------------------------
// The server returns one verdict per finished rep or 2 second window. Counted
// ones go into the set's stats; the latest verdict drives the status panel.
function applyV5Rep(rep, overlayId) {
  const fb = rep.feedback || {};
  if (rep.counted) {
    stats.total += 1;
    if (rep.correct) {
      stats.good += 1;
    } else {
      stats.bad += 1;
      const key = fb.key || (rep.faults[0] && rep.faults[0].key) || 'form_issue';
      stats.faults[key] = (stats.faults[key] || 0) + 1;
    }
    stats.prevRep = stats.total;
    if (overlayId && !isWindow()) setCounter(overlayId, stats.total);
    renderStats();
  }
  updateStatus({
    success: true,
    prediction: fb.key,
    confidence: (rep.score || 0) / 100,
    feedback: {
      ...fb,
      message: rep.counted
        ? `${isWindow() ? '2 s window' : 'Rep'} ${rep.rep}: ${fb.message || ''}`
        : (fb.message || '')
    }
  });
}

function waitingStatus(message) {
  const badge = $('status-badge');
  badge.textContent = 'Tracking';
  badge.style.backgroundColor = COLOR_MAP.secondary;
  $('status-message').textContent = message;
  setStatusIcon('secondary');
}

async function webcamLoopV5() {
  if (!webcamRunning) return;
  const video = $('webcam');
  const canvas = $('output-canvas');
  // One frame in flight at a time: the frame rate follows the server's pose
  // speed instead of queueing up stale frames.
  const t = performance.now();
  const result = await sendLiveFrame(currentExercise, video, t);
  if (!webcamRunning) return;
  if (result.success) {
    drawSkeleton17(canvas, result.keypoints, $('show-landmarks').checked);
    if (isWindow()) setCounter('rep-counter-overlay', result.jumps ?? result.hold_seconds);
    if (result.new_reps && result.new_reps.length) {
      result.new_reps.forEach(rep => applyV5Rep(rep, 'rep-counter-overlay'));
    } else if (!result.person) {
      updateStatus({ success: false, message: 'Position yourself in frame' });
    } else if (stats.total === 0) {
      waitingStatus(isWindow()
        ? 'Tracking you. Every 2 seconds are judged as you go.'
        : 'Tracking you. Each rep is judged when you finish it.');
    }
  } else {
    drawSkeleton17(canvas, null);
    updateStatus({ success: false, message: result.error || 'Could not analyse the frame' });
    await new Promise(r => setTimeout(r, 500));
  }
  requestAnimationFrame(webcamLoopV5);
}

// ---- Video upload mode -----------------------------------------------------
const videoUpload = $('upload-area');
videoUpload.addEventListener('click', () => $('video-input').click());
videoUpload.addEventListener('dragover', e => e.preventDefault());
videoUpload.addEventListener('drop', e => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) loadVideoFile(e.dataTransfer.files[0]);
});
$('video-input').addEventListener('change', e => {
  if (e.target.files[0]) loadVideoFile(e.target.files[0]);
});

function loadVideoFile(file) {
  v5.videoFile = file;
  v5.result = null;
  const video = $('preview-video');
  video.src = URL.createObjectURL(file);
  hide($('upload-area')); show($('video-preview'));
  video.addEventListener('loadeddata', () => {
    sizeCanvasTo($('video-output-canvas'), video);
  }, { once: true });
}

$('analyze-video').addEventListener('click', startVideoAnalysis);
$('pause-video').addEventListener('click', () => {
  videoPaused = true; $('preview-video').pause();
  hide($('pause-video')); show($('resume-video'));
});
$('resume-video').addEventListener('click', () => {
  videoPaused = false; $('preview-video').play();
  show($('pause-video')); hide($('resume-video'));
  isV5() ? videoLoopV5() : videoLoop();
});
$('stop-video').addEventListener('click', stopVideo);
$('upload-another-video').addEventListener('click', async () => {
  await stopVideo();
  v5.result = null;
  show($('upload-area')); hide($('video-preview'));
});

async function startVideoAnalysis() {
  if (isV5()) return startVideoAnalysisV5();
  const video = $('preview-video');
  await resetCounter(currentExercise);
  resetStats();
  videoRunning = true; videoPaused = false;
  hide($('analyze-video')); show($('pause-video')); show($('stop-video'));
  show($('upload-another-video')); show($('video-progress'));
  startSession('video');
  await video.play();
  videoLoop();
}

async function stopVideo() {
  const wasRunning = videoRunning;
  videoRunning = false; videoPaused = false;
  const video = $('preview-video');
  if (video) video.pause();
  show($('analyze-video'));
  hide($('pause-video')); hide($('resume-video')); hide($('stop-video'));
  if (wasRunning) await endSession(true);
}

async function videoLoop() {
  if (!videoRunning || videoPaused) return;
  const video = $('preview-video');
  const canvas = $('video-output-canvas');
  const showLm = $('show-landmarks') ? $('show-landmarks').checked : true;

  if (video.ended) { stopVideo(); return; }

  const now = performance.now();
  if (now - lastAnalyze >= ANALYZE_INTERVAL_MS) {
    lastAnalyze = now;
    const ts = Math.max(now, lastTimestamp + 1);
    lastTimestamp = ts;
    const landmarks = await pose.detectVideo(video, ts);
    drawSkeleton(canvas, video, landmarks, showLm);
    if (landmarks) {
      const result = await pose.analyzeFrame(currentExercise, landmarks);
      updateStatus(result);
      updateReps(result, 'video-rep-counter-overlay');
    }
    const pct = video.duration
      ? Math.round((video.currentTime / video.duration) * 100) : 0;
    $('progress-text').textContent = 'Processing: ' + pct + '%';
    $('video-progress-fill').style.width = pct + '%';
  }
  requestAnimationFrame(videoLoop);
}

// v5: the server judges the whole clip first (the same pipeline that was
// evaluated on Kaggle), then the clip plays back with the skeleton and
// each rep's (or 2 second window's) verdict appearing as it finishes.
async function startVideoAnalysisV5() {
  const video = $('preview-video');
  if (!v5.videoFile) { toast('Choose a video first', 'bi-info-circle-fill'); return; }

  if (!v5.result) {
    hide($('analyze-video')); show($('video-progress'));
    const setProgress = (p, msg) => {
      const pct = Math.round((p || 0) * 100);
      $('progress-text').textContent = `${msg || 'Analysing'}: ${pct}%`;
      $('progress-frames').textContent = 'Analysing on the server';
      $('video-progress-fill').style.width = pct + '%';
    };
    setProgress(0, 'Uploading video');
    const result = await analyzeVideoFile(currentExercise, v5.videoFile, setProgress);
    if (!result.success) {
      show($('analyze-video'));
      toast(result.error || 'Video analysis failed', 'bi-exclamation-triangle-fill');
      updateStatus({ success: false, message: result.error || 'Video analysis failed' });
      return;
    }
    v5.result = result;
    if (!result.reps_counted && result.reps.length) {
      toast(isWindow()
        ? 'This clip is under 2 seconds, so all of it was judged as one window'
        : 'No full rep found, so the whole clip was judged as one', 'bi-info-circle-fill');
    }
  }

  resetStats();
  v5.revealed = 0;
  setCounter('video-rep-counter-overlay', 0);
  videoRunning = true; videoPaused = false;
  hide($('analyze-video')); show($('pause-video')); show($('stop-video'));
  show($('upload-another-video')); show($('video-progress'));
  const flagged = v5.result.reps.length - v5.result.reps.filter(r => r.correct).length;
  $('progress-frames').textContent = isWindow()
    ? `${v5.result.reps_counted} 2 s ${unitWord(v5.result.reps_counted)} judged \u00b7 ${flagged} with faults`
    : `${v5.result.reps_counted} reps judged \u00b7 ${flagged} with faults`;
  startSession('video');
  waitingStatus(isWindow()
    ? 'Playing back. Each 2 second window is judged as it finishes.'
    : 'Playing back. Each rep is judged as it finishes.');
  video.currentTime = 0;
  // Playback can be refused (autoplay rules, a background tab); the loop
  // still follows the video once the user presses play on it.
  video.play().catch(() => toast(`Press play on the video to see each ${unitWord(1)}`, 'bi-play-circle'));
  videoLoopV5();
}

function videoLoopV5() {
  if (!videoRunning || videoPaused || !v5.result) return;
  const video = $('preview-video');
  const canvas = $('video-output-canvas');
  const res = v5.result;
  const showLm = $('show-landmarks') ? $('show-landmarks').checked : true;

  // The pipeline samples the clip at res.fps; show the matching skeleton.
  const idx = Math.min(res.keypoints.length - 1, Math.max(0, Math.floor(video.currentTime * res.fps)));
  drawSkeleton17(canvas, res.keypoints[idx], showLm);

  // Reveal every rep whose end has been reached (all of them at the end).
  const t = video.ended ? Infinity : video.currentTime;
  while (v5.revealed < res.reps.length && res.reps[v5.revealed].end <= t) {
    applyV5Rep(res.reps[v5.revealed], 'video-rep-counter-overlay');
    v5.revealed += 1;
  }
  if (res.jump_times) {
    setCounter('video-rep-counter-overlay', res.jump_times.filter(j => j <= t).length);
  } else if (res.hold_seconds !== undefined) {
    setCounter('video-rep-counter-overlay', Math.min(t, res.hold_seconds));
  }

  const pct = video.duration ? Math.round((video.currentTime / video.duration) * 100) : 0;
  $('progress-text').textContent = 'Playback: ' + pct + '%';
  $('video-progress-fill').style.width = pct + '%';

  if (video.ended) { stopVideo(); return; }
  requestAnimationFrame(videoLoopV5);
}

// ---- Image mode ------------------------------------------------------------
const imageUpload = $('image-upload-area');
imageUpload.addEventListener('click', () => $('image-input').click());
imageUpload.addEventListener('dragover', e => e.preventDefault());
imageUpload.addEventListener('drop', e => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) loadImageFile(e.dataTransfer.files[0]);
});
$('image-input').addEventListener('change', e => {
  if (e.target.files[0]) loadImageFile(e.target.files[0]);
});
$('upload-another').addEventListener('click', () => {
  show($('image-upload-area')); hide($('image-preview')); hide($('image-results'));
  hide($('v5-image-results'));
});

function loadImageFile(file) {
  v5.imageFile = file;
  const img = $('original-image');
  img.src = URL.createObjectURL(file);
  hide($('image-upload-area')); show($('image-preview'));
}

$('analyze-image').addEventListener('click', async () => {
  const img = $('original-image');
  if (!img.complete) { await img.decode().catch(() => { }); }

  if (isV5()) return analyzeImageV5(img);

  setLoading(true, 'Analyzing...');
  const landmarks = await pose.detectImage(img);
  if (!landmarks) {
    setLoading(false);
    updateStatus({ success: false, message: 'No pose detected in image' });
    return;
  }

  const canvas = $('analyzed-image');
  sizeCanvasTo(canvas, img);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  drawSkeleton(canvas, img, landmarks, true);

  const result = await pose.analyzeImage(currentExercise, landmarks);
  setLoading(false);
  updateStatus(result);
  populateImageResults(result);
});

// v5 on a photo: the server finds the pose and runs the rule checks that do
// not need movement. Depth and the BiGRU judge whole reps, so they cannot, and
// jump rope is not offered at all.
async function analyzeImageV5(img) {
  setLoading(true, 'Analyzing...');
  const result = await analyzeImageFile(currentExercise, v5.imageFile);
  setLoading(false);
  if (!result.success) {
    updateStatus({ success: false, message: result.error || 'No pose detected in image' });
    return;
  }
  const canvas = $('analyzed-image');
  sizeCanvasTo(canvas, img);
  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
  drawSkeleton17(canvas, result.keypoints, true, false);

  updateStatus({
    success: true, prediction: result.feedback.key,
    confidence: (result.score || 0) / 100, feedback: result.feedback
  });
  populateImageResultsV5(result);
}

function populateImageResultsV5(result) {
  hide($('image-results'));
  const box = $('v5-image-results');
  show(box);
  const label = (k) => faultLabels[k] || k.replace(/_/g, ' ');
  const flagged = new Set(result.faults.map(f => f.key));
  const rows = result.checked.map(k => {
    const p = Math.round((result.fused[k] || 0) * 100);
    const bad = flagged.has(k);
    return `<div class="analysis-card">
        <div class="analysis-icon"><i class="bi ${bad ? 'bi-exclamation-triangle-fill' : 'bi-check-circle-fill'}"></i></div>
        <div class="analysis-content">
          <h4>${label(k)}</h4>
          <p class="analysis-value">${bad ? 'Flagged' : 'Looks fine'}</p>
          <p class="analysis-detail">Fault likelihood ${p}%</p>
        </div>
      </div>`;
  });
  rows.push(`<div class="analysis-card">
      <div class="analysis-icon"><i class="bi bi-bullseye"></i></div>
      <div class="analysis-content">
        <h4>Overall</h4>
        <p class="analysis-value">${result.correct ? 'No fault seen' : 'Fault seen'}</p>
        <p class="analysis-detail">Score ${result.score} &middot; camera view: ${result.view || '-'}</p>
      </div>
    </div>`);
  $('v5-image-grid').innerHTML = rows.join('');
  const skipped = result.not_checked.map(label);
  $('v5-image-note').textContent = skipped.length
    ? `Not checked from a photo: ${skipped.join(', ').toLowerCase()}. These need a camera angle the photo does not have or real movement, so try a video for them.`
    : '';
}

function populateImageResults(result) {
  hide($('v5-image-results'));
  show($('image-results'));
  const d = result.analysis_details || {};
  const setText = (id, val) => { const el = $(id); if (el) el.textContent = val ?? '-'; };

  setText('knee-angle-value', d.knee_angle ?? '-');
  setText('left-knee', d.left_knee_angle ?? '-');
  setText('right-knee', d.right_knee_angle ?? '-');
  setText('hip-angle-value', d.hip_angle ?? '-');
  setText('depth-value', d.depth_achieved === undefined
    ? '-' : (d.depth_achieved ? 'Achieved' : 'Not deep enough'));
  setText('stance-value', d.stance_width ?? d.stance_position ?? '-');
  setText('back-lean-value', d.back_lean ?? d.back_straightness ?? '-');
  setText('lean-amount', d.lean_amount ?? '-');

  const fb = result.feedback || {};
  setText('overall-form', fb.status || result.prediction || '-');
  setText('form-confidence', Math.round((result.confidence || 0) * 100));
}

// ---- Boot ------------------------------------------------------------------
refreshTotals();
