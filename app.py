"""
Flask Web Application cho Emotion Detection
"""
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.emotion_detector import EmotionDetector

app = Flask(__name__)
CORS(app)

# Global detector
detector = None
MODEL_PATH = 'models/emotion_model.h5'

def init_detector():
    """Khởi tạo detector"""
    global detector
    if detector is None:
        if os.path.exists(MODEL_PATH):
            detector = EmotionDetector(MODEL_PATH, use_vietnamese=True)
            print("✓ Đã load model")
        else:
            print(f"❌ Không tìm thấy model tại: {MODEL_PATH}")
            print("Vui lòng chạy training trước: python src/train.py")

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """
    API nhận diện cảm xúc từ ảnh upload
    """
    if detector is None:
        return jsonify({'error': 'Model chưa được load'}), 500
    
    try:
        # Nhận ảnh từ request
        if 'image' not in request.files:
            return jsonify({'error': 'Không có file ảnh'}), 400
        
        file = request.files['image']
        
        # Đọc ảnh
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Không thể đọc ảnh'}), 400
        
        # Detect emotions
        results = detector.detect_emotion_from_image(image)
        
        if not results:
            return jsonify({
                'success': True,
                'faces_count': 0,
                'message': 'Không phát hiện khuôn mặt nào'
            })
        
        # Vẽ kết quả
        output_image = detector.draw_results(image, results)
        
        # Chuyển sang base64
        _, buffer = cv2.imencode('.jpg', output_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Tạo response
        faces = []
        for (x, y, w, h, emotion, confidence) in results:
            faces.append({
                'emotion': emotion,
                'confidence': f'{confidence:.2f}',
                'position': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })
        
        return jsonify({
            'success': True,
            'faces_count': len(results),
            'faces': faces,
            'image': f'data:image/jpeg;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_base64', methods=['POST'])
def detect_base64():
    """
    API nhận diện từ base64 (cho webcam)
    """
    if detector is None:
        return jsonify({'error': 'Model chưa được load'}), 500
    
    try:
        data = request.get_json()
        img_data = data['image'].split(',')[1]
        
        # Decode base64
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect
        results = detector.detect_emotion_from_image(image)
        
        if not results:
            return jsonify({
                'success': True,
                'faces_count': 0,
                'faces': []
            })
        
        # Response
        faces = []
        for (x, y, w, h, emotion, confidence) in results:
            # Lấy xác suất tất cả cảm xúc
            face_img = image[y:y+h, x:x+w]
            _, _, probabilities = detector.predict_emotion(face_img)
            
            faces.append({
                'emotion': emotion,
                'confidence': float(confidence),
                'position': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'probabilities': probabilities
            })
        
        return jsonify({
            'success': True,
            'faces_count': len(results),
            'faces': faces
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': detector is not None
    })

if __name__ == '__main__':
    print("="*60)
    print("EMOTION DETECTION WEB APP")
    print("="*60)
    
    # Khởi tạo detector
    init_detector()
    
    if detector is None:
        print("\n⚠️  Cảnh báo: Model chưa được load!")
        print("App vẫn chạy nhưng chức năng nhận diện sẽ không hoạt động.")
        print("Vui lòng chạy training trước: python src/train.py")
    
    print("\n✓ Khởi động server...")
    print("✓ Truy cập: http://localhost:5000")
    print("\nNhấn Ctrl+C để dừng server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
