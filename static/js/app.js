// ================================
// GLOBAL VARIABLES
// ================================
let emotionChart = null;
let webcamChart = null;
let videoStream = null;
let webcamInterval = null;
let fpsInterval = null;
let lastFrameTime = Date.now();
let frameCount = 0;

const EMOTIONS_VI = {
    'Tức giận': '😠',
    'Ghê tởm': '🤢',
    'Sợ hãi': '😨',
    'Vui vẻ': '😊',
    'Buồn bã': '😢',
    'Ngạc nhiên': '😲',
    'Bình thường': '😐'
};

const EMOTION_COLORS = {
    'Tức giận': '#ef4444',
    'Ghê tởm': '#8b5cf6',
    'Sợ hãi': '#f97316',
    'Vui vẻ': '#22c55e',
    'Buồn bã': '#3b82f6',
    'Ngạc nhiên': '#eab308',
    'Bình thường': '#6b7280'
};

// ================================
// DOM ELEMENTS
// ================================
const elements = {
    // Upload
    imageUpload: document.getElementById('imageUpload'),
    uploadArea: document.getElementById('uploadArea'),
    previewImage: document.getElementById('previewImage'),
    previewContainer: document.getElementById('previewContainer'),
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
document.addEventListener('DOMContentLoaded', function() {
    initializeUpload();
    initializeWebcam();
    smoothScroll();
});

// ================================
// UPLOAD IMAGE FUNCTIONALITY
// ================================
function initializeUpload() {
    // File input change
    elements.imageUpload.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    
    // Detect button
    elements.detectBtn.addEventListener('click', detectEmotion);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        previewFile(file);
    }
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
    if (file && file.type.startsWith('image/')) {
        previewFile(file);
        // Set the file to input element
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        elements.imageUpload.files = dataTransfer.files;
    }
}

function previewFile(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
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
    if (!file) {
        showNotification('Vui lòng chọn ảnh trước!', 'warning');
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
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showNotification(data.error || 'Có lỗi xảy ra', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Không thể kết nối đến server', 'error');
    } finally {
        showLoading(false);
    }
}

function displayResults(data) {
    elements.resultsCard.style.display = 'block';
    
    // Show detected image
    if (data.image) {
        elements.previewImage.src = data.image;
    }
    
    // Show stats
    let statsHTML = `
        <div class="alert alert-success">
            <i class="fas fa-check-circle me-2"></i>
            <strong>Phát hiện ${data.faces_count} khuôn mặt</strong>
        </div>
    `;
    
    if (data.faces && data.faces.length > 0) {
        statsHTML += '<div class="emotion-list">';
        data.faces.forEach((face, index) => {
            const emoji = EMOTIONS_VI[face.emotion] || '😊';
            statsHTML += `
                <div class="emotion-item">
                    <span class="emotion-name">
                        ${emoji} Khuôn mặt ${index + 1}: ${face.emotion}
                    </span>
                    <span class="emotion-value">${face.confidence}%</span>
                </div>
            `;
        });
        statsHTML += '</div>';
    }
    
    elements.resultStats.innerHTML = statsHTML;
    
    // Update chart
    if (data.faces && data.faces.length > 0) {
        updateEmotionChart(data.faces[0]);
    }
    
    // Smooth scroll to results
    elements.resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateEmotionChart(face) {
    const ctx = document.getElementById('emotionChart').getContext('2d');
    
    // Destroy existing chart
    if (emotionChart) {
        emotionChart.destroy();
    }
    
    // Get probabilities
    let labels = [];
    let values = [];
    let colors = [];
    
    if (face.probabilities) {
        for (const [emotion, value] of Object.entries(face.probabilities)) {
            labels.push(emotion);
            values.push(value.toFixed(2));
            colors.push(EMOTION_COLORS[emotion] || '#6b7280');
        }
    }
    
    emotionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Xác suất (%)',
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
                    text: 'Phân bố xác suất các cảm xúc',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
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
        
        // Wait for video to load
        elements.video.onloadedmetadata = () => {
            elements.video.play();
            startDetectionLoop();
            startFPSCounter();
        };
        
    } catch (error) {
        console.error('Error accessing webcam:', error);
        showNotification('Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập.', 'error');
    }
}

function stopWebcam() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
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
    
    elements.video.style.display = 'none';
    elements.webcamPlaceholder.style.display = 'block';
    elements.startWebcam.style.display = 'inline-block';
    elements.stopWebcam.style.display = 'none';
    elements.cameraStatus.classList.remove('online');
    elements.cameraStatus.querySelector('.status-text').textContent = 'Offline';
    elements.fpsCounter.textContent = 'FPS: 0';
    
    // Clear canvas
    const ctx = elements.canvas.getContext('2d');
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
}

function startDetectionLoop() {
    webcamInterval = setInterval(async () => {
        await detectWebcamEmotion();
        frameCount++;
    }, 200); // Detect every 200ms (~5 FPS for API calls)
}

function startFPSCounter() {
    fpsInterval = setInterval(() => {
        const now = Date.now();
        const elapsed = (now - lastFrameTime) / 1000;
        const fps = Math.round(frameCount / elapsed);
        elements.fpsCounter.textContent = `FPS: ${fps}`;
        frameCount = 0;
        lastFrameTime = now;
    }, 1000);
}

async function detectWebcamEmotion() {
    const canvas = elements.canvas;
    const video = elements.video;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0);
    
    // Convert canvas to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    try {
        const response = await fetch('/detect_base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        
        if (data.success && data.faces && data.faces.length > 0) {
            drawWebcamResults(ctx, data.faces);
            updateWebcamStats(data.faces);
        } else {
            // Just redraw the video frame
            ctx.drawImage(video, 0, 0);
            elements.faceCount.textContent = '0';
            elements.emotionList.innerHTML = '<p class="text-muted small">Không phát hiện khuôn mặt</p>';
        }
    } catch (error) {
        console.error('Error detecting emotion:', error);
    }
}

function drawWebcamResults(ctx, faces) {
    // Redraw video frame
    ctx.drawImage(elements.video, 0, 0);
    
    // Draw bounding boxes and labels
    faces.forEach(face => {
        const { position, emotion, confidence } = face;
        const { x, y, w, h } = position;
        
        // Draw rectangle
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        
        // Draw label background
        const text = `${emotion}: ${confidence.toFixed(1)}%`;
        ctx.font = 'bold 16px Poppins';
        const textMetrics = ctx.measureText(text);
        const textHeight = 20;
        
        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.fillRect(x, y - textHeight - 5, textMetrics.width + 10, textHeight + 5);
        
        // Draw text
        ctx.fillStyle = '#000';
        ctx.fillText(text, x + 5, y - 8);
    });
}

function updateWebcamStats(faces) {
    // Update face count
    elements.faceCount.textContent = faces.length;
    
    // Update emotion list
    let listHTML = '';
    faces.forEach((face, index) => {
        const emoji = EMOTIONS_VI[face.emotion] || '😊';
        listHTML += `
            <div class="emotion-item">
                <span class="emotion-name">
                    ${emoji} Face ${index + 1}: ${face.emotion}
                </span>
                <span class="emotion-value">${face.confidence.toFixed(1)}%</span>
            </div>
        `;
    });
    elements.emotionList.innerHTML = listHTML;
    
    // Update chart with first face
    if (faces.length > 0 && faces[0].probabilities) {
        updateWebcamChart(faces[0].probabilities);
    }
}

function updateWebcamChart(probabilities) {
    const ctx = document.getElementById('webcamChart').getContext('2d');
    
    // Destroy existing chart
    if (webcamChart) {
        webcamChart.destroy();
    }
    
    let labels = [];
    let values = [];
    let colors = [];
    
    for (const [emotion, value] of Object.entries(probabilities)) {
        labels.push(emotion);
        values.push(value.toFixed(2));
        colors.push(EMOTION_COLORS[emotion] || '#6b7280');
    }
    
    webcamChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
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
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(2) + '%';
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
    // Simple alert for now, can be replaced with toast notification
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    alert(`${icons[type]} ${message}`);
}

function smoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
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
    // ESC to stop webcam
    if (e.key === 'Escape' && videoStream) {
        stopWebcam();
    }
    
    // Space to detect (when in upload tab)
    if (e.key === ' ' && document.querySelector('#upload.active') && elements.imageUpload.files[0]) {
        e.preventDefault();
        detectEmotion();
    }
});

// ================================
// CONSOLE MESSAGE
// ================================
console.log('%c🎭 Face Emotion Recognition System', 'color: #667eea; font-size: 20px; font-weight: bold');
console.log('%cPowered by TensorFlow & OpenCV', 'color: #764ba2; font-size: 14px');
console.log('%c💡 Press ESC to stop webcam', 'color: #6c757d; font-size: 12px');

