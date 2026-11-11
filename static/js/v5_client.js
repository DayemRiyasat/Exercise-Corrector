// static/js/v5_client.js
// Client for the Exercise-Corrector pipeline (all five exercises).
//
// The server does the computer vision: it runs YOLO11 pose on camera frames
// or on the uploaded clip, counts reps (or cuts 2 second windows for jump
// rope and plank) and judges each one with rules + a BiGRU. This module
// sends it frames, files and photos, and draws the 17-joint skeleton it
// returns.

// COCO-17 skeleton (the joint order YOLO pose returns)
const COCO_EDGES = [
  [5, 6], [5, 11], [6, 12], [11, 12],
  [5, 7], [7, 9], [6, 8], [8, 10],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 1], [0, 2], [1, 3], [2, 4]
];
const MIN_CONF = 0.3;            // the pipeline's own joint confidence cut-off
const FRAME_MAX_SIDE = 640;      // the pipeline decodes video at this size

async function postForm(url, form) {
  try {
    const res = await fetch(url, { method: 'POST', body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok && data.success === undefined) {
      return { success: false, error: `Request failed (${res.status})` };
    }
    return data;
  } catch (err) {
    return { success: false, error: err.message || 'Network error' };
  }
}

// ---- Drawing ---------------------------------------------------------------
// keypoints: 17 x [x, y, conf], x and y as fractions of the frame.
export function drawSkeleton17(canvas, keypoints, show = true, clear = true) {
  const ctx = canvas.getContext('2d');
  if (clear) ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!show || !keypoints) return;

  const w = canvas.width, h = canvas.height;
  const scale = Math.max(1, Math.min(w, h) / 480);
  const ok = (p) => p && p[2] >= MIN_CONF;

  ctx.strokeStyle = 'rgba(1, 211, 210, .92)';
  ctx.lineWidth = 4 * scale;
  ctx.lineCap = 'round';
  COCO_EDGES.forEach(([a, b]) => {
    const p = keypoints[a], q = keypoints[b];
    if (!ok(p) || !ok(q)) return;
    ctx.beginPath();
    ctx.moveTo(p[0] * w, p[1] * h);
    ctx.lineTo(q[0] * w, q[1] * h);
    ctx.stroke();
  });

  ctx.fillStyle = '#FFFFFF';
  ctx.strokeStyle = '#0D77E7';
  ctx.lineWidth = 2 * scale;
  keypoints.forEach(p => {
    if (!ok(p)) return;
    ctx.beginPath();
    ctx.arc(p[0] * w, p[1] * h, 4 * scale, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  });
}

// ---- Live webcam -----------------------------------------------------------
const grabCanvas = document.createElement('canvas');

// One JPEG of the current video frame, long side <= 640 px.
function grabFrame(videoEl) {
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  if (!vw || !vh) return Promise.resolve(null);
  const s = Math.min(1, FRAME_MAX_SIDE / Math.max(vw, vh));
  grabCanvas.width = Math.round(vw * s);
  grabCanvas.height = Math.round(vh * s);
  grabCanvas.getContext('2d').drawImage(videoEl, 0, 0, grabCanvas.width, grabCanvas.height);
  return new Promise(resolve => grabCanvas.toBlob(resolve, 'image/jpeg', 0.85));
}

// Send one webcam frame. timeMs is when it was captured (performance.now()),
// which the server uses to put frames on the pipeline's 16 fps clock.
export async function sendLiveFrame(exerciseType, videoEl, timeMs) {
  const blob = await grabFrame(videoEl);
  if (!blob) return { success: false, error: 'Camera not ready' };
  const form = new FormData();
  form.append('exercise_type', exerciseType);
  form.append('t', String(timeMs));
  form.append('frame', blob, 'frame.jpg');
  return postForm('/api/v5/live_frame', form);
}

// ---- Uploaded video --------------------------------------------------------
// Uploads the clip, then polls until the server has judged it.
// onProgress(fraction, message) is called while it works.
export async function analyzeVideoFile(exerciseType, file, onProgress) {
  const form = new FormData();
  form.append('exercise_type', exerciseType);
  form.append('video', file, file.name || 'video.mp4');
  onProgress && onProgress(0, 'Uploading video');
  const started = await postForm('/api/v5/analyze_video', form);
  if (!started.success) return started;

  for (;;) {
    await new Promise(r => setTimeout(r, 1000));
    let job;
    try {
      const res = await fetch(`/api/v5/jobs/${encodeURIComponent(started.job_id)}`);
      job = await res.json();
    } catch (err) {
      return { success: false, error: err.message || 'Network error' };
    }
    if (!job.success) return job;
    onProgress && onProgress(job.progress || 0, job.message || '');
    if (job.status === 'done' || job.status === 'error') {
      return job.result || { success: false, error: job.message };
    }
  }
}

// ---- Single photo ----------------------------------------------------------
export function analyzeImageFile(exerciseType, file) {
  const form = new FormData();
  form.append('exercise_type', exerciseType);
  form.append('image', file, file.name || 'photo.jpg');
  return postForm('/api/v5/analyze_image', form);
}
