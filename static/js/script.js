// script.js - Enhanced with Image Upload Support

let selectedExercise = null;
let selectedMode = null;
let webcamStream = null;
let isProcessing = false;
let processingInterval = null;
let stats = { good: 0, bad: 0, total: 0 };
let frameCount = 0;
let repCount = 0;
let videoCapture = null;
let isVideoProcessing = false;
let videoPaused = false;
let videoResults = [];
let currentVideoTime = 0;
let videoTotalFrames = 0;
let videoProcessedFrames = 0;
let videoRepCount = 0;
let currentSquatState = "standing";
let videoFrameQueue = [];
let isProcessingQueue = false;
let maxQueueSize = 30;
let processingThreadActive = false;

// Add these variables at the top with other globals
let lastState = "";
let lastPrediction = "";
let lastRepCount = 0;
let lastUpdateTime = 0;

// DOM Elements
const exerciseCards = document.querySelectorAll('.exercise-card');
const exerciseSection = document.getElementById('exercise-selection');
const modeSection = document.getElementById('input-mode-selection');
const analysisSection = document.getElementById('analysis-section');
const webcamBtn = document.getElementById('webcam-btn');
const videoBtn = document.getElementById('video-btn');
const imageBtn = document.getElementById('image-btn'); // NEW
const webcamMode = document.getElementById('webcam-mode');
const videoMode = document.getElementById('video-mode');
const imageMode = document.getElementById('image-mode'); // NEW
const startWebcamBtn = document.getElementById('start-webcam');
const stopWebcamBtn = document.getElementById('stop-webcam');
const resetRepsBtn = document.getElementById('reset-reps');
const webcamElement = document.getElementById('webcam');
const outputCanvas = document.getElementById('output-canvas');
const showLandmarksCheckbox = document.getElementById('show-landmarks');
const uploadArea = document.getElementById('upload-area');
const videoInput = document.getElementById('video-input');
const videoPreview = document.getElementById('video-preview');
const previewVideo = document.getElementById('preview-video');
const analyzeVideoBtn = document.getElementById('analyze-video');
const pauseVideoBtn = document.getElementById('pause-video');
const resumeVideoBtn = document.getElementById('resume-video');
const stopVideoBtn = document.getElementById('stop-video');
const uploadAnotherVideoBtn = document.getElementById('upload-another-video'); // NEW
const videoProgress = document.getElementById('video-progress');
const progressText = document.getElementById('progress-text');
const progressFrames = document.getElementById('progress-frames');
const videoProgressFill = document.getElementById('video-progress-fill');
const videoOutputCanvas = document.getElementById('video-output-canvas');
const videoResultsDiv = document.getElementById('video-results');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingMessage = document.getElementById('loading-message');

// NEW: Image Mode Elements
const imageUploadArea = document.getElementById('image-upload-area');
const imageInput = document.getElementById('image-input');
const imagePreview = document.getElementById('image-preview');
const originalImage = document.getElementById('original-image');
const analyzedImage = document.getElementById('analyzed-image');
const analyzeImageBtn = document.getElementById('analyze-image');
const uploadAnotherBtn = document.getElementById('upload-another');
const imageResults = document.getElementById('image-results');

// Status Panel Elements
const statusBadge = document.getElementById('status-badge');
const statusIcon = document.getElementById('status-icon');
const statusMessage = document.getElementById('status-message');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceValue = document.getElementById('confidence-value');
const tipsList = document.getElementById('tips-list');
const statGood = document.getElementById('stat-good');
const statBad = document.getElementById('stat-bad');
const statTotal = document.getElementById('stat-total');
const statAccuracy = document.getElementById('stat-accuracy');
const statReps = document.getElementById('stat-reps');
const repCounterOverlay = document.getElementById('rep-counter-overlay');
const videoRepCounterOverlay = document.getElementById('video-rep-counter-overlay');

// Step 1: Exercise Selection
exerciseCards.forEach(card => {
    card.addEventListener('click', async () => {
        const exercise = card.dataset.exercise;
        
        if (card.classList.contains('disabled')) {
            return;
        }
        
        exerciseCards.forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        
        selectedExercise = exercise;
        
        showLoading('Loading model...');
        
        try {
            const response = await fetch('/api/load_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ exercise_type: exercise })
            });
            
            const data = await response.json();
            
            hideLoading();
            
            if (data.success) {
                modeSection.classList.remove('hidden');
                modeSection.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert('Error: ' + data.error);
                card.classList.remove('selected');
                selectedExercise = null;
            }
        } catch (error) {
            hideLoading();
            alert('Error loading model: ' + error.message);
            card.classList.remove('selected');
            selectedExercise = null;
        }
    });
});

// Step 2: Mode Selection
webcamBtn.addEventListener('click', () => {
    selectedMode = 'webcam';
    showAnalysisSection();
    webcamMode.classList.remove('hidden');
    videoMode.classList.add('hidden');
    imageMode.classList.add('hidden'); // NEW
});

videoBtn.addEventListener('click', () => {
    selectedMode = 'video';
    showAnalysisSection();
    webcamMode.classList.add('hidden');
    videoMode.classList.remove('hidden');
    imageMode.classList.add('hidden'); // NEW
});

// NEW: Image Mode Selection
imageBtn.addEventListener('click', () => {
    selectedMode = 'image';
    showAnalysisSection();
    webcamMode.classList.add('hidden');
    videoMode.classList.add('hidden');
    imageMode.classList.remove('hidden');
});

function showAnalysisSection() {
    analysisSection.classList.remove('hidden');
    analysisSection.scrollIntoView({ behavior: 'smooth' });
    resetStats();
    resetRepCount();
}

// Webcam Mode
startWebcamBtn.addEventListener('click', async () => {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 1280, height: 720 } 
        });
        
        webcamElement.srcObject = webcamStream;
        
        webcamElement.onloadedmetadata = () => {
            webcamElement.play();
            
            outputCanvas.width = webcamElement.videoWidth;
            outputCanvas.height = webcamElement.videoHeight;
            
            startWebcamBtn.classList.add('hidden');
            stopWebcamBtn.classList.remove('hidden');
            if (resetRepsBtn) resetRepsBtn.classList.remove('hidden');
            
            isProcessing = true;
            processWebcamFrame();
        };
        
    } catch (error) {
        alert('Error accessing webcam: ' + error.message);
    }
});

stopWebcamBtn.addEventListener('click', () => {
    stopWebcam();
});

if (resetRepsBtn) {
    resetRepsBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/reset_counter', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            
            if (data.success) {
                resetRepCount();
                showNotification('Rep counter reset!', 'success');
            }
        } catch (error) {
            console.error('Error resetting counter:', error);
        }
    });
}

function stopWebcam() {
    isProcessing = false;
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    webcamElement.srcObject = null;
    
    startWebcamBtn.classList.remove('hidden');
    stopWebcamBtn.classList.add('hidden');
    if (resetRepsBtn) resetRepsBtn.classList.add('hidden');
    
    if (processingInterval) {
        cancelAnimationFrame(processingInterval);
    }
    
    frameCount = 0;
}

async function processWebcamFrame() {
    if (!isProcessing) return;
    
    frameCount++;
    if (frameCount % 3 !== 0) {
        processingInterval = requestAnimationFrame(processWebcamFrame);
        return;
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight;
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(webcamElement, 0, 0);
    
    const frameData = canvas.toDataURL('image/jpeg', 0.5);
    
    try {
        const response = await fetch('/api/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: frameData })
        });
        
        const data = await response.json();
        const currentTime = Date.now();
        
        if (data.success) {
            if (showLandmarksCheckbox.checked) {
                const img = new Image();
                img.onload = () => {
                    const canvasCtx = outputCanvas.getContext('2d');
                    canvasCtx.drawImage(img, 0, 0);
                };
                img.src = data.processed_frame;
            }
            
            const stateChanged = data.rep_info && (
                data.rep_info.state !== lastState || 
                data.prediction !== lastPrediction
            );
            
            if (stateChanged || currentTime - lastUpdateTime > 1000) {
                updateStatusPanel(data.prediction, data.confidence, data.feedback, data.rep_info);
                lastState = data.rep_info ? data.rep_info.state : "";
                lastPrediction = data.prediction;
                lastUpdateTime = currentTime;
            }
            
            if (data.rep_info) {
                const repInfo = data.rep_info;
                currentSquatState = repInfo.state;
                repCount = repInfo.rep_count;
                
                if (repCount !== lastRepCount) {
                    updateRepCount();
                    lastRepCount = repCount;
                }
                
                if (repInfo.rep_counted) {
                    flashRepCounter();
                    showNotification(
                        `Rep #${repCount} counted! Form: ${(repInfo.form_quality * 100).toFixed(0)}%`, 
                        'success'
                    );
                    
                    if (repCounterOverlay) {
                        const repNumber = repCounterOverlay.querySelector('.rep-number');
                        if (repNumber) {
                            repNumber.textContent = repCount;
                            repNumber.style.transform = 'scale(1.5)';
                            repNumber.style.color = '#10B981';
                            setTimeout(() => {
                                repNumber.style.transform = 'scale(1)';
                                repNumber.style.color = '';
                            }, 500);
                        }
                    }
                }
                
                if (repInfo.state_changed) {
                    updateStateDisplay(repInfo);
                }
            }
            
            const isCorrect = data.prediction === 'none';
            if (isCorrect) {
                stats.good++;
            } else {
                stats.bad++;
            }
            stats.total++;
            updateStatsDisplay();
            
        } else {
            if (lastPrediction !== 'unknown') {
                updateStatusPanel('unknown', 0, {
                    status: 'NO POSE DETECTED',
                    message: data.message || 'Position yourself in frame',
                    tips: ['Ensure full body is visible', 'Stand in good lighting'],
                    color: 'secondary'
                });
                lastPrediction = 'unknown';
            }
        }
        
    } catch (error) {
        console.error('Error processing frame:', error);
    }
    
    processingInterval = requestAnimationFrame(processWebcamFrame);
}

function updateStatusPanel(prediction, confidence, feedback, repInfo) {
    statusBadge.textContent = feedback.status;
    statusBadge.className = 'status-badge';
    
    if (feedback.color === 'success') {
        statusBadge.classList.add('correct');
    } else if (feedback.color === 'danger') {
        statusBadge.classList.add('incorrect');
    } else if (feedback.color === 'warning') {
        statusBadge.classList.add('warning');
    }
    
    const icons = {
        'success': '✅',
        'danger': '❌',
        'warning': '⚠️',
        'secondary': '⏳'
    };
    statusIcon.textContent = icons[feedback.color] || '⏳';
    
    statusMessage.textContent = feedback.message;
    
    const confidencePercent = Math.round(confidence * 100);
    confidenceFill.style.width = confidencePercent + '%';
    confidenceValue.textContent = confidencePercent + '%';
    
    tipsList.innerHTML = '';
    feedback.tips.forEach(tip => {
        const li = document.createElement('li');
        li.textContent = tip;
        tipsList.appendChild(li);
    });
}

let stateDisplayTimeout = null;
function updateStateDisplay(repInfo) {
    if (stateDisplayTimeout) {
        clearTimeout(stateDisplayTimeout);
    }
    
    stateDisplayTimeout = setTimeout(() => {
        if (repCounterOverlay) {
            const repLabel = repCounterOverlay.querySelector('.rep-label');
            if (repLabel) {
            const stateEmojis = {
                    'standing': '🧍 READY',
                    'descending': '⬇️ DOWN',
                    'bottom': '🔄 HOLD',
                    'ascending': '⬆️ UP',
                    'lifting': '💪 LIFT',
                    'up': '🔝 READY',
                    'down': '⬇️ ARMS DOWN',      // Bicep curl
                    'curling': '💪 CURLING',      // Bicep curl
                    'lowering': '⬇️ LOWERING',    // Bicep curl
                };
                repLabel.textContent = stateEmojis[repInfo.state] || 'REPS';
            }
        }
    }, 100);
}

// Video Upload Mode
uploadArea.addEventListener('click', () => {
    videoInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary-color)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--border-color)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--border-color)';
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
        handleVideoFile(file);
    }
});

videoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleVideoFile(file);
    }
});

function handleVideoFile(file) {
    const url = URL.createObjectURL(file);
    previewVideo.src = url;
    
    previewVideo.onloadedmetadata = () => {
        videoOutputCanvas.width = previewVideo.videoWidth;
        videoOutputCanvas.height = previewVideo.videoHeight;
    };
    
    uploadArea.classList.add('hidden');
    videoPreview.classList.remove('hidden');
    videoResultsDiv.classList.add('hidden');
    videoProgress.classList.add('hidden');
    
    analyzeVideoBtn.classList.remove('hidden');
    pauseVideoBtn.classList.add('hidden');
    resumeVideoBtn.classList.add('hidden');
    stopVideoBtn.classList.add('hidden');
    uploadAnotherVideoBtn.classList.add('hidden'); // Hide initially
    
    videoPreview.videoFile = file;
}

// NEW: Upload Another Video Handler
uploadAnotherVideoBtn.addEventListener('click', () => {
    // Reset everything
    isVideoProcessing = false;
    videoPaused = false;
    videoResults = [];
    currentVideoTime = 0;
    videoProcessedFrames = 0;
    videoRepCount = 0;
    currentSquatState = "standing";
    videoFrameQueue = [];
    isProcessingQueue = false;
    
    // Clear video
    previewVideo.src = '';
    previewVideo.videoFile = null;
    videoInput.value = ''; // Reset file input
    
    // Clear canvas
    const ctx = videoOutputCanvas.getContext('2d');
    ctx.clearRect(0, 0, videoOutputCanvas.width, videoOutputCanvas.height);
    
    // Reset rep counter
    if (videoRepCounterOverlay) {
        const repNumber = videoRepCounterOverlay.querySelector('.rep-number');
        if (repNumber) {
            repNumber.textContent = '0';
        }
    }
    
    // Hide preview and results
    videoPreview.classList.add('hidden');
    videoResultsDiv.classList.add('hidden');
    videoProgress.classList.add('hidden');
    
    // Show upload area
    uploadArea.classList.remove('hidden');
    
    // Reset status panel
    updateStatusPanel('unknown', 0, {
        status: 'WAITING',
        message: 'Upload a video to analyze',
        tips: ['Upload your exercise video', 'Ensure full body is visible', 'Good lighting recommended'],
        color: 'secondary'
    });
    
    // Reset stats
    resetStats();
    
    showNotification('Ready to upload another video!', 'info');
});

analyzeVideoBtn.addEventListener('click', async () => {
    const file = videoPreview.videoFile;
    
    if (!file) {
        alert('No video file selected');
        return;
    }
    
    isVideoProcessing = true;
    videoPaused = false;
    videoResults = [];
    currentVideoTime = 0;
    videoProcessedFrames = 0;
    videoRepCount = 0;
    currentSquatState = "standing";
    videoFrameQueue = [];
    isProcessingQueue = false;
    
    if (videoRepCounterOverlay) {
        const repNumber = videoRepCounterOverlay.querySelector('.rep-number');
        if (repNumber) {
            repNumber.textContent = '0';
        }
    }
    
    if (statReps) {
        statReps.textContent = '0';
    }
    
    previewVideo.currentTime = 0;
    videoOutputCanvas.width = previewVideo.videoWidth || 640;
    videoOutputCanvas.height = previewVideo.videoHeight || 480;
    
    videoTotalFrames = Math.floor(previewVideo.duration * 30 / 5);
    
    analyzeVideoBtn.classList.add('hidden');
    pauseVideoBtn.classList.remove('hidden');
    stopVideoBtn.classList.remove('hidden');
    uploadAnotherVideoBtn.classList.add('hidden'); // Hide during analysis
    videoProgress.classList.remove('hidden');
    videoResultsDiv.classList.add('hidden');
    
    updateStatusPanel('unknown', 0, {
        status: 'ANALYZING VIDEO',
        message: 'Processing video frames...',
        tips: ['Video analysis in progress', 'Using threaded processing'],
        color: 'secondary'
    });
    
    startThreadedVideoProcessing();
});

pauseVideoBtn.addEventListener('click', () => {
    videoPaused = true;
    pauseVideoBtn.classList.add('hidden');
    resumeVideoBtn.classList.remove('hidden');
});

resumeVideoBtn.addEventListener('click', () => {
    videoPaused = false;
    resumeVideoBtn.classList.add('hidden');
    pauseVideoBtn.classList.remove('hidden');
    
    if (!isProcessingQueue) {
        processVideoQueue();
    }
    continueVideoCapture();
});

stopVideoBtn.addEventListener('click', () => {
    isVideoProcessing = false;
    videoPaused = false;
    videoFrameQueue = [];
    
    analyzeVideoBtn.classList.remove('hidden');
    pauseVideoBtn.classList.add('hidden');
    resumeVideoBtn.classList.add('hidden');
    stopVideoBtn.classList.add('hidden');
    uploadAnotherVideoBtn.classList.remove('hidden'); // Show upload another button
    videoProgress.classList.add('hidden');
    
    if (videoResults.length > 0) {
        displayVideoResults({
            success: true,
            total_frames: videoProcessedFrames,
            correct_frames: videoResults.filter(r => r.prediction === 'none').length,
            accuracy: (videoResults.filter(r => r.prediction === 'none').length / videoProcessedFrames) * 100,
            results: videoResults,
            reps: videoRepCount
        });
    }
});

async function startThreadedVideoProcessing() {
    captureVideoFrames();
    processVideoQueue();
}

async function captureVideoFrames() {
    if (!isVideoProcessing || videoPaused) {
        if (isVideoProcessing && videoPaused) {
            setTimeout(captureVideoFrames, 100);
        }
        return;
    }
    
    if (currentVideoTime >= previewVideo.duration) {
        isVideoProcessing = false;
        
        const checkQueue = setInterval(() => {
            if (videoFrameQueue.length === 0 && !isProcessingQueue) {
                clearInterval(checkQueue);
                
                pauseVideoBtn.classList.add('hidden');
                stopVideoBtn.classList.add('hidden');
                analyzeVideoBtn.classList.remove('hidden');
                uploadAnotherVideoBtn.classList.remove('hidden'); // Show upload another button
                
                displayVideoResults({
                    success: true,
                    total_frames: videoProcessedFrames,
                    correct_frames: videoResults.filter(r => r.prediction === 'none').length,
                    accuracy: (videoResults.filter(r => r.prediction === 'none').length / videoProcessedFrames) * 100,
                    results: videoResults,
                    reps: videoRepCount
                });
            }
        }, 100);
        
        return;
    }
    
    previewVideo.currentTime = currentVideoTime;
    
    await new Promise(resolve => {
        previewVideo.onseeked = resolve;
    });
    
    const frameNumber = Math.floor(currentVideoTime * 30);
    
    if (frameNumber % 5 === 0) {
        const canvas = document.createElement('canvas');
        canvas.width = previewVideo.videoWidth;
        canvas.height = previewVideo.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(previewVideo, 0, 0);
        
        const frameData = canvas.toDataURL('image/jpeg', 0.5);
        
        if (videoFrameQueue.length < maxQueueSize) {
            videoFrameQueue.push({
                data: frameData,
                number: frameNumber,
                time: currentVideoTime.toFixed(2)
            });
        }
    }
    
    currentVideoTime += 0.033;
    
    setTimeout(captureVideoFrames, 10);
}

function continueVideoCapture() {
    if (isVideoProcessing && !videoPaused) {
        captureVideoFrames();
    }
}

async function processVideoQueue() {
    if (isProcessingQueue) return;
    
    isProcessingQueue = true;
    
    while ((isVideoProcessing || videoFrameQueue.length > 0) && !videoPaused) {
        if (videoFrameQueue.length === 0) {
            await new Promise(resolve => setTimeout(resolve, 50));
            continue;
        }
        
        const frameInfo = videoFrameQueue.shift();
        
        try {
            const response = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame: frameInfo.data })
            });
            
            const data = await response.json();
            
            if (data.success) {
                videoProcessedFrames++;
                
                videoResults.push({
                    frame: frameInfo.number,
                    time: frameInfo.time,
                    prediction: data.prediction,
                    confidence: data.confidence
                });
                
                if (showLandmarksCheckbox.checked && data.processed_frame) {
                    const img = new Image();
                    img.onload = () => {
                        const outputCtx = videoOutputCanvas.getContext('2d');
                        outputCtx.drawImage(img, 0, 0, videoOutputCanvas.width, videoOutputCanvas.height);
                    };
                    img.src = data.processed_frame;
                }
                
                updateStatusPanel(data.prediction, data.confidence, data.feedback);
                
                if (data.rep_info) {
                    const repInfo = data.rep_info;
                    currentSquatState = repInfo.state;
                    
                    if (repInfo.rep_counted) {
                        videoRepCount++;
                        updateVideoRepCount();
                        
                        console.log(`✅ Rep #${videoRepCount} - Quality: ${(repInfo.form_quality * 100).toFixed(0)}%`);
                    }
                    
                    updateVideoStateDisplay(repInfo);
                }
                
                const progress = videoProcessedFrames > 0 ? 
                    Math.min(Math.round((parseFloat(frameInfo.time) / previewVideo.duration) * 100), 100) : 0;
                progressText.textContent = `Processing: ${progress}%`;
                progressFrames.textContent = `Frame: ${videoProcessedFrames} | Reps: ${videoRepCount} | State: ${currentSquatState}`;
                videoProgressFill.style.width = progress + '%';
            }
        } catch (error) {
            console.error('Error processing frame:', error);
        }
    }
    
    isProcessingQueue = false;
}

function updateVideoStateDisplay(repInfo) {
    if (videoRepCounterOverlay) {
        const repLabel = videoRepCounterOverlay.querySelector('.rep-label');
        if (repLabel) {
             const stateEmojis = {
                    'standing': '🧍 READY',
                    'descending': '⬇️ DOWN',
                    'bottom': '🔄 HOLD',
                    'ascending': '⬆️ UP',
                    'lifting': '💪 LIFT',
                    'up': '🔝 READY',
                    'down': '⬇️ ARMS DOWN',      // Bicep curl
                    'curling': '💪 CURLING',      // Bicep curl
                    'lowering': '⬇️ LOWERING',    // Bicep curl
                };
            repLabel.textContent = stateEmojis[repInfo.state] || 'REPS';
        }
    }
}

function updateVideoRepCount() {
    if (statReps) {
        statReps.textContent = videoRepCount;
    }
    
    if (videoRepCounterOverlay) {
        const repNumber = videoRepCounterOverlay.querySelector('.rep-number');
        if (repNumber) {
            repNumber.textContent = videoRepCount;
            repNumber.style.transform = 'scale(1.3)';
            setTimeout(() => {
                repNumber.style.transform = 'scale(1)';
            }, 300);
        }
    }
}

function displayVideoResults(data) {
    videoResultsDiv.classList.remove('hidden');
    
    const correctFrames = data.correct_frames;
    const totalFrames = data.total_frames;
    const accuracy = data.accuracy.toFixed(1);
    const reps = data.reps || 0;
    
    videoResultsDiv.innerHTML = `
        <h3>📊 Video Analysis Complete</h3>
        
        <div class="result-summary">
            <div class="result-item">
                <div class="result-item-value" style="background: var(--gradient-1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${reps}</div>
                <div class="result-item-label">Total Reps</div>
            </div>
            <div class="result-item">
                <div class="result-item-value">${totalFrames}</div>
                <div class="result-item-label">Frames Analyzed</div>
            </div>
            <div class="result-item">
                <div class="result-item-value" style="background: linear-gradient(135deg, #10B981, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${correctFrames}</div>
                <div class="result-item-label">Correct Form</div>
            </div>
            <div class="result-item">
                <div class="result-item-value" style="background: linear-gradient(135deg, #EF4444, #DC2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${totalFrames - correctFrames}</div>
                <div class="result-item-label">Incorrect Form</div>
            </div>
            <div class="result-item">
                <div class="result-item-value">${accuracy}%</div>
                <div class="result-item-label">Accuracy</div>
            </div>
        </div>
        
        <div class="result-details">
            <h4>📋 Frame-by-Frame Breakdown</h4>
            <div style="max-height: 300px; overflow-y: auto;">
                ${data.results.slice(0, 100).map(r => `
                    <div class="result-row">
                        <span class="result-frame">Frame ${r.frame} (${r.time}s)</span>
                        <span class="result-prediction ${r.prediction === 'none' ? 'correct' : 'incorrect'}">
                            ${r.prediction === 'none' ? '✓ Correct' : '✗ ' + r.prediction.replace(/_/g, ' ')}
                        </span>
                    </div>
                `).join('')}
                ${data.results.length > 100 ? '<p style="text-align: center; color: var(--text-secondary); margin-top: 15px;">Showing first 100 results</p>' : ''}
            </div>
        </div>
    `;
    
    // Show upload another button after results
    uploadAnotherVideoBtn.classList.remove('hidden');
    
    videoResultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// NEW: Image Upload Mode
imageUploadArea.addEventListener('click', () => {
    imageInput.click();
});

imageUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    imageUploadArea.style.borderColor = 'var(--primary-color)';
});

imageUploadArea.addEventListener('dragleave', () => {
    imageUploadArea.style.borderColor = 'var(--border-color)';
});

imageUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    imageUploadArea.style.borderColor = 'var(--border-color)';
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageFile(file);
    }
});

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImageFile(file);
    }
});

function handleImageFile(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        originalImage.src = e.target.result;
        originalImage.imageData = e.target.result;
        
        imageUploadArea.classList.add('hidden');
        imagePreview.classList.remove('hidden');
        imageResults.classList.add('hidden');
        
        // Clear previous analysis
        analyzedImage.width = 0;
        analyzedImage.height = 0;
    };
    
    reader.readAsDataURL(file);
}

analyzeImageBtn.addEventListener('click', async () => {
    if (!originalImage.imageData) {
        alert('No image loaded');
        return;
    }
    
    showLoading('Analyzing posture...');
    
    try {
        const response = await fetch('/api/process_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: originalImage.imageData })
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.success) {
            // Display analyzed image with landmarks
            const img = new Image();
            img.onload = () => {
                analyzedImage.width = img.width;
                analyzedImage.height = img.height;
                const ctx = analyzedImage.getContext('2d');
                ctx.drawImage(img, 0, 0);
            };
            img.src = data.processed_frame;
            
            // Update status panel
            updateStatusPanel(data.prediction, data.confidence, data.feedback);
            
            // Display detailed analysis
            displayImageAnalysis(data);
            
            imageResults.classList.remove('hidden');
            imageResults.scrollIntoView({ behavior: 'smooth' });
            
        } else {
            alert('Error: ' + (data.message || data.error));
        }
        
    } catch (error) {
        hideLoading();
        alert('Error analyzing image: ' + error.message);
    }
});

uploadAnotherBtn.addEventListener('click', () => {
    imagePreview.classList.add('hidden');
    imageResults.classList.add('hidden');
    imageUploadArea.classList.remove('hidden');
    originalImage.src = '';
    originalImage.imageData = null;
    analyzedImage.width = 0;
    analyzedImage.height = 0;
    imageInput.value = '';
});


function displayImageAnalysis(data) {
    const details = data.analysis_details;
    
    // Check if it's squat or lunge based on available details
    const isLunge = details.hasOwnProperty('front_knee_angle');
    
    if (isLunge) {
        // LUNGE ANALYSIS
        document.getElementById('knee-angle-value').textContent = details.front_knee_angle + '°';
        document.getElementById('left-knee').textContent = 'Front: ' + details.front_knee_angle + '°';
        document.getElementById('right-knee').textContent = 'Back: ' + details.back_knee_angle + '°';
        
        // Hip Angle
        document.getElementById('hip-angle-value').textContent = details.front_hip_angle + '°';
        
        // Lunge Depth
        const depthElement = document.getElementById('depth-value');
        if (details.depth_achieved) {
            depthElement.textContent = 'Good ✓';
            depthElement.style.color = '#10B981';
        } else {
            depthElement.textContent = 'Shallow ✗';
            depthElement.style.color = '#EF4444';
        }
        
        // Stance Width
        const stanceElement = document.getElementById('stance-value');
        stanceElement.textContent = details.stance_width;
        if (details.stance_width === 'Good') {
            stanceElement.style.color = '#10B981';
        } else {
            stanceElement.style.color = '#F59E0B';
        }
        
        // Back Position
        const backElement = document.getElementById('back-lean-value');
        backElement.textContent = details.back_lean;
        if (details.back_lean === 'Neutral') {
            backElement.style.color = '#10B981';
        } else {
            backElement.style.color = '#F59E0B';
        }
        document.getElementById('lean-amount').textContent = details.lean_amount;
        
    } else {
        // SQUAT ANALYSIS (existing code)
        document.getElementById('knee-angle-value').textContent = details.knee_angle + '°';
        document.getElementById('left-knee').textContent = details.left_knee_angle;
        document.getElementById('right-knee').textContent = details.right_knee_angle;
        
        // Hip Angle
        document.getElementById('hip-angle-value').textContent = details.hip_angle + '°';
        
        // Squat Depth
        const depthElement = document.getElementById('depth-value');
        if (details.depth_achieved) {
            depthElement.textContent = 'Good ✓';
            depthElement.style.color = '#10B981';
        } else {
            depthElement.textContent = 'Shallow ✗';
            depthElement.style.color = '#EF4444';
        }
        
        // Stance Width
        const stanceElement = document.getElementById('stance-value');
        stanceElement.textContent = details.stance_width;
        if (details.stance_width === 'Good') {
            stanceElement.style.color = '#10B981';
        } else {
            stanceElement.style.color = '#F59E0B';
        }
        
        // Back Position
        const backElement = document.getElementById('back-lean-value');
        backElement.textContent = details.back_lean;
        if (details.back_lean === 'Neutral') {
            backElement.style.color = '#10B981';
        } else {
            backElement.style.color = '#F59E0B';
        }
        document.getElementById('lean-amount').textContent = details.lean_amount;
    }
    
    // Overall Form (same for both)
    const formElement = document.getElementById('overall-form');
    const prediction = data.prediction;
    if (prediction === 'none') {
        formElement.textContent = 'Excellent ✓';
        formElement.style.color = '#10B981';
    } else {
        formElement.textContent = prediction.replace(/_/g, ' ').toUpperCase();
        formElement.style.color = '#EF4444';
    }
    document.getElementById('form-confidence').textContent = Math.round(data.confidence * 100);
}

function updateStatsDisplay() {
    statGood.textContent = stats.good;
    statBad.textContent = stats.bad;
    statTotal.textContent = stats.total;
    
    const accuracy = stats.total > 0 ? Math.round((stats.good / stats.total) * 100) : 0;
    statAccuracy.textContent = accuracy + '%';
}

function resetStats() {
    stats = { good: 0, bad: 0, total: 0 };
    updateStatsDisplay();
}

function updateRepCount() {
    if (statReps) {
        statReps.textContent = repCount;
    }
    if (repCounterOverlay) {
        const repNumber = repCounterOverlay.querySelector('.rep-number');
        if (repNumber) {
            repNumber.textContent = repCount;
        }
    }
}

function resetRepCount() {
    repCount = 0;
    updateRepCount();
}

function flashRepCounter() {
    if (statReps) {
        statReps.style.transform = 'scale(1.5)';
        statReps.style.color = '#10B981';
        setTimeout(() => {
            statReps.style.transform = 'scale(1)';
            statReps.style.color = '';
        }, 300);
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'success' ? '#10B981' : type === 'error' ? '#EF4444' : '#3B82F6'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function showLoading(message) {
    loadingMessage.textContent = message;
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopWebcam();
    isVideoProcessing = false;
});

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);