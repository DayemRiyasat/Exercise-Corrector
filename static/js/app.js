// static/js/app.js
// Front-end controller. Runs MediaPipe pose detection in the browser
// (via pose_client.js) and sends landmarks to the Flask backend.

import {
  initPose, detectImage, detectVideo,
  loadModel, analyzeFrame, analyzeImage, resetCounter
} from './pose_client.js';

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
const stats = { good: 0, bad: 0, total: 0, prevRep: 0 };

// ---- Small helpers ----
const $ = (id) => document.getElementById(id);
const show = (el) => el && el.classList.remove('hidden');
const hide = (el) => el && el.classList.add('hidden');

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

// ---- Status panel ----
const COLOR_MAP = {
  success: '#22c55e', danger: '#ef4444',
  warning: '#f59e0b', secondary: '#9ca3af'
};

function updateStatus(result) {
  if (!result || !result.success) {
    $('status-badge').textContent = 'No pose';
    $('status-message').textContent =
      (result && result.message) || 'Position yourself in frame';
    $('confidence-fill').style.width = '0%';
    $('confidence-value').textContent = '0%';
    return;
  }
  const fb = result.feedback || {};
  const badge = $('status-badge');
  badge.textContent = fb.status || result.prediction || '-';
  badge.style.backgroundColor = COLOR_MAP[fb.color] || COLOR_MAP.secondary;
  $('status-message').textContent = fb.message || '';

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

  // A completed rep is a rep-count increase. Classify it by the
  // current prediction: 'none' means good form, anything else is a fault.
  if (count > stats.prevRep) {
    const good = result.prediction === 'none';
    for (let i = stats.prevRep; i < count; i++) {
      stats.total += 1;
      good ? (stats.good += 1) : (stats.bad += 1);
    }
    stats.prevRep = count;
    renderStats();
  }
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
  renderStats();
}

// ---- Landmark drawing ----
function drawSkeleton(canvas, video, landmarks, show) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!show || !landmarks) return;

  const w = canvas.width, h = canvas.height;
  ctx.strokeStyle = 'rgba(0,0,255,0.8)';
  ctx.lineWidth = 2;
  POSE_CONNECTIONS.forEach(([a, b]) => {
    const p = landmarks[a], q = landmarks[b];
    if (!p || !q) return;
    ctx.beginPath();
    ctx.moveTo(p.x * w, p.y * h);
    ctx.lineTo(q.x * w, q.y * h);
    ctx.stroke();
  });
  ctx.fillStyle = 'rgba(0,255,0,0.9)';
  landmarks.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 3, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function sizeCanvasTo(canvas, mediaEl) {
  const w = mediaEl.videoWidth || mediaEl.naturalWidth || mediaEl.clientWidth;
  const h = mediaEl.videoHeight || mediaEl.naturalHeight || mediaEl.clientHeight;
  if (w && h) { canvas.width = w; canvas.height = h; }
}

// ---- Exercise + mode selection ----
document.querySelectorAll('.exercise-card').forEach(card => {
  if (card.classList.contains('disabled')) return;
  card.addEventListener('click', async () => {
    document.querySelectorAll('.exercise-card')
      .forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    currentExercise = card.dataset.exercise;

    setLoading(true, 'Loading model...');
    if (!poseReady) {
      try { await initPose(); poseReady = true; }
      catch (e) { setLoading(false); alert('Could not start pose detection: ' + e); return; }
    }
    const res = await loadModel(currentExercise);
    setLoading(false);
    if (!res.success) { alert(res.error || 'Model failed to load'); return; }

    show($('input-mode-selection'));
  });
});

$('webcam-btn').addEventListener('click', () => switchMode('webcam'));
$('video-btn').addEventListener('click', () => switchMode('video'));
$('image-btn').addEventListener('click', () => switchMode('image'));

function switchMode(mode) {
  stopWebcam(); stopVideo();
  show($('analysis-section'));
  ['webcam-mode', 'video-mode', 'image-mode'].forEach(id => hide($(id)));
  resetStats();
  updateStatus(null);
  if (mode === 'webcam') show($('webcam-mode'));
  if (mode === 'video') show($('video-mode'));
  if (mode === 'image') show($('image-mode'));
}

// ---- Webcam mode ----
$('start-webcam').addEventListener('click', startWebcam);
$('stop-webcam').addEventListener('click', stopWebcam);
$('reset-reps').addEventListener('click', async () => {
  await resetCounter(currentExercise);
  resetStats();
  $('rep-counter-overlay').querySelector('.rep-number').textContent = '0';
});

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (e) { alert('Camera access denied: ' + e); return; }

  const video = $('webcam');
  video.srcObject = webcamStream;
  await video.play();
  sizeCanvasTo($('output-canvas'), video);

  webcamRunning = true;
  hide($('start-webcam')); show($('stop-webcam')); show($('reset-reps'));
  resetStats();
  webcamLoop();
}

function stopWebcam() {
  webcamRunning = false;
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  show($('start-webcam')); hide($('stop-webcam')); hide($('reset-reps'));
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
    const landmarks = await detectVideo(video, ts);
    drawSkeleton(canvas, video, landmarks, showLm);
    if (landmarks) {
      const result = await analyzeFrame(currentExercise, landmarks);
      updateStatus(result);
      updateReps(result, 'rep-counter-overlay');
    } else {
      updateStatus({ success: false, message: 'Position yourself in frame' });
    }
  }
  requestAnimationFrame(webcamLoop);
}

// ---- Video upload mode ----
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
  videoLoop();
});
$('stop-video').addEventListener('click', stopVideo);
$('upload-another-video').addEventListener('click', () => {
  stopVideo();
  show($('upload-area')); hide($('video-preview'));
});

async function startVideoAnalysis() {
  const video = $('preview-video');
  await resetCounter(currentExercise);
  resetStats();
  videoRunning = true; videoPaused = false;
  hide($('analyze-video')); show($('pause-video')); show($('stop-video'));
  show($('upload-another-video')); show($('video-progress'));
  await video.play();
  videoLoop();
}

function stopVideo() {
  videoRunning = false; videoPaused = false;
  const video = $('preview-video');
  if (video) video.pause();
  show($('analyze-video'));
  hide($('pause-video')); hide($('resume-video')); hide($('stop-video'));
}

async function videoLoop() {
  if (!videoRunning || videoPaused) return;
  const video = $('preview-video');
  const canvas = $('video-output-canvas');
  const showLm = document.getElementById('show-landmarks')
    ? $('show-landmarks').checked : true;

  if (video.ended) { stopVideo(); return; }

  const now = performance.now();
  if (now - lastAnalyze >= ANALYZE_INTERVAL_MS) {
    lastAnalyze = now;
    const ts = Math.max(now, lastTimestamp + 1);
    lastTimestamp = ts;
    const landmarks = await detectVideo(video, ts);
    drawSkeleton(canvas, video, landmarks, showLm);
    if (landmarks) {
      const result = await analyzeFrame(currentExercise, landmarks);
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

// ---- Image mode ----
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
});

function loadImageFile(file) {
  const img = $('original-image');
  img.src = URL.createObjectURL(file);
  hide($('image-upload-area')); show($('image-preview'));
}

$('analyze-image').addEventListener('click', async () => {
  const img = $('original-image');
  if (!img.complete) { await img.decode().catch(() => {}); }

  setLoading(true, 'Analyzing...');
  const landmarks = await detectImage(img);
  if (!landmarks) {
    setLoading(false);
    updateStatus({ success: false, message: 'No pose detected in image' });
    return;
  }

  // Draw the skeleton onto the analyzed-image canvas
  const canvas = $('analyzed-image');
  sizeCanvasTo(canvas, img);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  drawSkeleton(canvas, img, landmarks, true);

  const result = await analyzeImage(currentExercise, landmarks);
  setLoading(false);
  updateStatus(result);
  populateImageResults(result);
});

function populateImageResults(result) {
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