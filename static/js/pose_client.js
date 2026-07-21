// static/js/pose_client.js
// Browser-side pose detection with MediaPipe Tasks for JavaScript.
// Extracts the 33 landmarks and posts them to the Flask backend.

import { FilesetResolver, PoseLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

let poseLandmarker = null;
let runningMode = "IMAGE";

export async function initPose() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
      delegate: "GPU"
    },
    runningMode: "IMAGE",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  return poseLandmarker;
}

function toServerLandmarks(result) {
  if (!result || !result.landmarks || result.landmarks.length === 0) return null;
  // result.landmarks[0] is the 33-point normalized landmark array,
  // the same convention the Python models were trained on.
  return result.landmarks[0].map(p => ({
    x: p.x, y: p.y, z: p.z, visibility: (p.visibility ?? 0)
  }));
}

async function ensureMode(mode) {
  if (runningMode !== mode) {
    runningMode = mode;
    await poseLandmarker.setOptions({ runningMode: mode });
  }
}

// Detect on a still <img> element. Returns landmark array or null.
export async function detectImage(imageEl) {
  await ensureMode("IMAGE");
  return toServerLandmarks(poseLandmarker.detect(imageEl));
}

// Detect on a <video> element frame. timestampMs must increase each call
// (use performance.now()). Returns landmark array or null.
export async function detectVideo(videoEl, timestampMs) {
  await ensureMode("VIDEO");
  return toServerLandmarks(poseLandmarker.detectForVideo(videoEl, timestampMs));
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return res.json();
}

export function loadModel(exerciseType) {
  return postJson("/api/load_model", { exercise_type: exerciseType });
}

export function analyzeFrame(exerciseType, landmarks) {
  return postJson("/api/analyze_frame", { exercise_type: exerciseType, landmarks });
}

export function analyzeImage(exerciseType, landmarks) {
  return postJson("/api/analyze_image", { exercise_type: exerciseType, landmarks });
}

export function resetCounter(exerciseType) {
  return postJson("/api/reset_counter", { exercise_type: exerciseType });
}