// ================================
// GLOBAL VARIABLES
// ================================
let emotionChart = null;
let webcamChart = null;
let videoStream = null;
let webcamInterval = null;
let fpsInterval = null;
let webcamRequestInFlight = false;
let lastFrameTime = Date.now();
let frameCount = 0;

const MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024;
const ALLOWED_IMAGE_TYPES = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp']);

const EMOTIONS_VI = {
    'T\u1ee9c gi\u1eadn': '😠',
    'Gh\u00ea t\u1edfm': '🤢',
    'S\u1ee3 h\u00e3i': '😨',
    'Vui v\u1ebb': '😊',
    'Bu\u1ed3n b\u00e3': '😢',
    'Ng\u1ea1c nhi\u00ean': '😲',
    'B\u00ecnh th\u01b0\u1eddng': '😐'
};

const EMOTION_COLORS = {
    'T\u1ee9c gi\u1eadn': '#ef4444',
    'Gh\u00ea t\u1edfm': '#8b5cf6',
    'S\u1ee3 h\u00e3i': '#f97316',
    'Vui v\u1ebb': '#22c55e',
    'Bu\u1ed3n b\u00e3': '#3b82f6',
    'Ng\u1ea1c nhi\u00ean': '#eab308',
    'B\u00ecnh th\u01b0\u1eddng': '#6b7280'
};

// ================================
// DOM ELEMENTS
// ================================
const elements = {
    // Upload
    imageUpload: document.getElementById('imageUpload'),
    uploadArea: document.getElementById('uploadArea'),
    previewImage: document.getElementById('previewImage'),
    detectBtn: document.getElementById('detectBtn'),
    resultsCard: document.getElementById('resultsCard'),
    resultStats: document.getElementById('resultStats'),

    // Webcam
    video: document.getElementById('video'),
    canvas: document.getElementById('canvas'),
    startWebcam: document.getElementById('startWebcam'),
    stopWebcam: document.getElementById('stopWebcam'),
    webcamPlaceholder: document.getElementById('webcamPlaceholder'),
    cameraStatus: document.getElementById('cameraStatus'),
    fpsCounter: document.getElementById('fpsCounter'),
    faceCount: document.getElementById('faceCount'),
    emotionList: document.getElementById('emotionList'),

    // Loading
    loadingOverlay: document.getElementById('loadingOverlay')
};

// ================================
// INITIALIZATION
// ================================
document.addEventListener('DOMContentLoaded', () => {
    initializeUpload();
    initializeWebcam();
    smoothScroll();
});

// ================================
// UPLOAD IMAGE FUNCTIONALITY
// ================================
function initializeUpload() {
    elements.imageUpload.addEventListener('change', handleFileSelect);
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    elements.detectBtn.addEventListener('click', detectEmotion);
}

function validateImageFile(file) {
    if (!file) {
        return 'Vui long chon anh truoc.';
    }

    if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
        return 'Dinh dang anh khong hop le. Chi ho tro JPG, PNG, WEBP.';
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
        return 'Anh vuot qua 5MB. Vui long chon anh nho hon.';
    }

    return null;
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    const error = validateImageFile(file);
    if (error) {
        showNotification(error, 'warning');
        elements.imageUpload.value = '';
        return;
    }

    previewFile(file);
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    const error = validateImageFile(file);
    if (error) {
        showNotification(error, 'warning');
        return;
    }

    previewFile(file);

    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    elements.imageUpload.files = dataTransfer.files;
}

function previewFile(file) {
    const reader = new FileReader();

    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.previewImage.style.display = 'block';
        document.querySelector('.preview-placeholder').style.display = 'none';
        elements.detectBtn.style.display = 'block';
        elements.resultsCard.style.display = 'none';
    };

    reader.readAsDataURL(file);
}

async function detectEmotion() {
    const file = elements.imageUpload.files[0];
    const error = validateImageFile(file);
    if (error) {
        showNotification(error, 'warning');
        return;
    }

    showLoading(true);

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Khong the phan tich anh');
        }

        displayResults(data);
    } catch (fetchError) {
        console.error('Error:', fetchError);
        showNotification(fetchError.message || 'Khong the ket noi den server', 'error');
    } finally {
        showLoading(false);
    }
}

function displayResults(data) {
    elements.resultsCard.style.display = 'block';

    if (data.image) {
        elements.previewImage.src = data.image;
    }

    let statsHTML = `
        <div class="alert alert-success">
            <i class="fas fa-check-circle me-2"></i>
            <strong>Phat hien ${data.faces_count} khuon mat</strong>
        </div>
    `;

    if (data.message) {
        statsHTML += `<p class="text-muted small mb-2">${data.message}</p>`;
    }

    if (Array.isArray(data.faces) && data.faces.length > 0) {
        statsHTML += '<div class="emotion-list">';
        data.faces.forEach((face, index) => {
            const emoji = EMOTIONS_VI[face.emotion] || '😊';
            statsHTML += `
                <div class="emotion-item">
                    <span class="emotion-name">
                        ${emoji} Khuon mat ${index + 1}: ${face.emotion}
                    </span>
                    <span class="emotion-value">${Number(face.confidence).toFixed(2)}%</span>
                </div>
            `;
        });
        statsHTML += '</div>';
    }

    elements.resultStats.innerHTML = statsHTML;

    if (Array.isArray(data.faces) && data.faces.length > 0 && data.faces[0].probabilities) {
        updateEmotionChart(data.faces[0].probabilities);
    } else if (emotionChart) {
        emotionChart.destroy();
        emotionChart = null;
    }

    elements.resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateEmotionChart(probabilities) {
    const ctx = document.getElementById('emotionChart').getContext('2d');

    if (emotionChart) {
        emotionChart.destroy();
    }

    const labels = [];
    const values = [];
    const colors = [];

    Object.entries(probabilities).forEach(([emotion, value]) => {
        labels.push(emotion);
        values.push(Number(value.toFixed(2)));
        colors.push(EMOTION_COLORS[emotion] || '#6b7280');
    });

    emotionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Xac suat (%)',
                data: values,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Phan bo xac suat cac cam xuc',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    callbacks: {
                        label(context) {
                            return `${context.parsed.y.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback(value) {
                            return `${value}%`;
                        }
                    }
                }
            }
        }
    });
}

// ================================
// WEBCAM FUNCTIONALITY
// ================================
function initializeWebcam() {
    elements.startWebcam.addEventListener('click', startWebcam);
    elements.stopWebcam.addEventListener('click', stopWebcam);
}

async function startWebcam() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480
            }
        });

        elements.video.srcObject = videoStream;
        elements.video.style.display = 'block';
        elements.webcamPlaceholder.style.display = 'none';
        elements.startWebcam.style.display = 'none';
        elements.stopWebcam.style.display = 'inline-block';
        elements.cameraStatus.classList.add('online');
        elements.cameraStatus.querySelector('.status-text').textContent = 'Online';

        frameCount = 0;
        lastFrameTime = Date.now();

        elements.video.onloadedmetadata = () => {
            elements.video.play();
            startDetectionLoop();
            startFPSCounter();
        };
    } catch (error) {
        console.error('Error accessing webcam:', error);
        showNotification('Khong the truy cap camera. Vui long kiem tra quyen truy cap.', 'error');
    }
}

function stopWebcam() {
    if (videoStream) {
        videoStream.getTracks().forEach((track) => track.stop());
        videoStream = null;
    }

    if (webcamInterval) {
        clearInterval(webcamInterval);
        webcamInterval = null;
    }

    if (fpsInterval) {
        clearInterval(fpsInterval);
        fpsInterval = null;
    }

    webcamRequestInFlight = false;

    elements.video.style.display = 'none';
    elements.webcamPlaceholder.style.display = 'block';
    elements.startWebcam.style.display = 'inline-block';
    elements.stopWebcam.style.display = 'none';
    elements.cameraStatus.classList.remove('online');
    elements.cameraStatus.querySelector('.status-text').textContent = 'Offline';
    elements.fpsCounter.textContent = 'FPS: 0';

    const ctx = elements.canvas.getContext('2d');
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

    clearWebcamStats();
}

function startDetectionLoop() {
    webcamInterval = setInterval(() => {
        if (!webcamRequestInFlight) {
            detectWebcamEmotion();
        }
    }, 200);
}

function startFPSCounter() {
    fpsInterval = setInterval(() => {
        const now = Date.now();
        const elapsed = (now - lastFrameTime) / 1000;
        const fps = elapsed > 0 ? Math.round(frameCount / elapsed) : 0;
        elements.fpsCounter.textContent = `FPS: ${fps}`;
        frameCount = 0;
        lastFrameTime = now;
    }, 1000);
}

async function detectWebcamEmotion() {
    const canvas = elements.canvas;
    const video = elements.video;
    const ctx = canvas.getContext('2d');

    if (!video.videoWidth || !video.videoHeight) {
        return;
    }

    webcamRequestInFlight = true;

    try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        const response = await fetch('/detect_base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Khong the phan tich webcam frame');
        }

        if (Array.isArray(data.faces) && data.faces.length > 0) {
            drawWebcamResults(ctx, data.faces);
            updateWebcamStats(data.faces);
        } else {
            ctx.drawImage(video, 0, 0);
            clearWebcamStats();
        }

        frameCount += 1;
    } catch (error) {
        console.error('Error detecting emotion:', error);
    } finally {
        webcamRequestInFlight = false;
    }
}

function drawWebcamResults(ctx, faces) {
    ctx.drawImage(elements.video, 0, 0);

    faces.forEach((face) => {
        const { position, emotion, confidence } = face;
        const { x, y, w, h } = position;

        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        const text = `${emotion}: ${Number(confidence).toFixed(1)}%`;
        ctx.font = 'bold 16px Poppins';
        const textMetrics = ctx.measureText(text);
        const textHeight = 20;

        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.fillRect(x, y - textHeight - 5, textMetrics.width + 10, textHeight + 5);

        ctx.fillStyle = '#000';
        ctx.fillText(text, x + 5, y - 8);
    });
}

function updateWebcamStats(faces) {
    elements.faceCount.textContent = faces.length;

    let listHTML = '';
    faces.forEach((face, index) => {
        const emoji = EMOTIONS_VI[face.emotion] || '😊';
        listHTML += `
            <div class="emotion-item">
                <span class="emotion-name">
                    ${emoji} Khuon mat ${index + 1}: ${face.emotion}
                </span>
                <span class="emotion-value">${Number(face.confidence).toFixed(1)}%</span>
            </div>
        `;
    });
    elements.emotionList.innerHTML = listHTML;

    if (faces.length > 0 && faces[0].probabilities) {
        updateWebcamChart(faces[0].probabilities);
    } else if (webcamChart) {
        webcamChart.destroy();
        webcamChart = null;
    }
}

function clearWebcamStats() {
    elements.faceCount.textContent = '0';
    elements.emotionList.innerHTML = '<p class="text-muted small">Khong phat hien khuon mat</p>';

    if (webcamChart) {
        webcamChart.destroy();
        webcamChart = null;
    }
}

function updateWebcamChart(probabilities) {
    const ctx = document.getElementById('webcamChart').getContext('2d');

    if (webcamChart) {
        webcamChart.destroy();
    }

    const labels = [];
    const values = [];
    const colors = [];

    Object.entries(probabilities).forEach(([emotion, value]) => {
        labels.push(emotion);
        values.push(Number(value.toFixed(2)));
        colors.push(EMOTION_COLORS[emotion] || '#6b7280');
    });

    webcamChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label(context) {
                            return `${context.label}: ${context.parsed.toFixed(2)}%`;
                        }
                    }
                }
            }
        }
    });
}

// ================================
// UTILITY FUNCTIONS
// ================================
function showLoading(show) {
    elements.loadingOverlay.style.display = show ? 'flex' : 'none';
}

function showNotification(message, type = 'info') {
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };

    alert(`${icons[type]} ${message}`);
}

function smoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
        anchor.addEventListener('click', function scrollToTarget(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ================================
// CLEAN UP ON PAGE UNLOAD
// ================================
window.addEventListener('beforeunload', () => {
    stopWebcam();
});

// ================================
// KEYBOARD SHORTCUTS
// ================================
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && videoStream) {
        stopWebcam();
    }

    if (e.key === ' ' && document.querySelector('#upload.active') && elements.imageUpload.files[0]) {
        e.preventDefault();
        detectEmotion();
    }
});

// ================================
// CONSOLE MESSAGE
// ================================
console.log('%cFace Emotion Recognition System', 'color: #667eea; font-size: 20px; font-weight: bold');
console.log('%cPowered by TensorFlow & OpenCV', 'color: #764ba2; font-size: 14px');
console.log('%cPress ESC to stop webcam', 'color: #6c757d; font-size: 12px');
