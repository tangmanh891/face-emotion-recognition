import base64
import binascii
import logging
import os
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover - fallback for minimal environments
    CORS = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
if CORS is not None:
    CORS(app)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
# Prefer the modern .keras format; fall back to legacy .h5 if a user still
# has the previous artifact on disk.
_KERAS_MODEL = MODEL_DIR / 'emotion_model.keras'
_H5_MODEL = MODEL_DIR / 'emotion_model.h5'
MODEL_PATH = _KERAS_MODEL if _KERAS_MODEL.exists() else _H5_MODEL
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_MIME_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp'}

app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE_BYTES

# Global detector
# Lazy-loaded to support `flask run` and WSGI servers.
detector = None
detector_lock = Lock()


def init_detector():
    """Initialize detector once per process."""
    global detector

    with detector_lock:
        if detector is not None:
            return detector

        if not MODEL_PATH.exists():
            logger.warning('Model not found at: %s', MODEL_PATH)
            logger.warning('Run training first: python src/train.py')
            return None

        try:
            from src.emotion_detector import EmotionDetector

            detector = EmotionDetector(str(MODEL_PATH), use_vietnamese=True)
            logger.info('Model loaded successfully from %s', MODEL_PATH)
        except Exception:
            detector = None
            logger.exception('Failed to load model')

        return detector


def get_detector():
    """Get detector and initialize lazily if needed."""
    return detector if detector is not None else init_detector()


def decode_uploaded_image(file_storage):
    """Decode uploaded image from multipart form-data."""
    if file_storage is None or not file_storage.filename:
        return None, ('Khong co file anh', 400)

    mimetype = (file_storage.mimetype or '').lower()
    if mimetype and mimetype not in ALLOWED_IMAGE_MIME_TYPES:
        return None, ('Dinh dang anh khong duoc ho tro', 400)

    image_bytes = file_storage.read()
    if not image_bytes:
        return None, ('File anh rong', 400)

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None, ('Khong the doc du lieu anh', 400)

    return image, None


def decode_base64_image(payload):
    """Decode base64 image from JSON payload."""
    if not isinstance(payload, dict):
        return None, ('Du lieu yeu cau khong hop le', 400)

    image_data = payload.get('image')
    if not isinstance(image_data, str) or not image_data:
        return None, ('Thieu truong image', 400)

    encoded_data = image_data.split(',', 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded_data, validate=True)
    except (binascii.Error, ValueError):
        return None, ('Du lieu base64 khong hop le', 400)

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None, ('Khong the doc du lieu anh', 400)

    return image, None


def build_faces_response(face_analyses):
    """Create serializable face payload and draw tuples."""
    faces = []
    draw_results = []

    for face in face_analyses:
        draw_results.append(
            (
                face['x'],
                face['y'],
                face['w'],
                face['h'],
                face['emotion'],
                face['confidence'],
            )
        )
        faces.append(
            {
                'emotion': face['emotion'],
                'confidence': round(face['confidence'], 2),
                'position': {
                    'x': face['x'],
                    'y': face['y'],
                    'w': face['w'],
                    'h': face['h'],
                },
                'probabilities': face['probabilities'],
            }
        )

    return faces, draw_results


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_):
    return jsonify({'error': 'Kich thuoc anh vuot qua 5MB'}), 413


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """Detect emotions from uploaded image."""
    current_detector = get_detector()
    if current_detector is None:
        return jsonify({'error': 'Model chua duoc load'}), 500

    try:
        image, error = decode_uploaded_image(request.files.get('image'))
        if error:
            message, status_code = error
            return jsonify({'error': message}), status_code

        face_analyses = current_detector.analyze_faces(image)
        if not face_analyses:
            return jsonify(
                {
                    'success': True,
                    'faces_count': 0,
                    'faces': [],
                    'message': 'Khong phat hien khuon mat nao',
                }
            )

        faces, draw_results = build_faces_response(face_analyses)

        output_image = current_detector.draw_results(image, draw_results)
        ok, buffer = cv2.imencode('.jpg', output_image)
        if not ok:
            raise RuntimeError('Khong the ma hoa anh ket qua')

        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify(
            {
                'success': True,
                'faces_count': len(faces),
                'faces': faces,
                'image': f'data:image/jpeg;base64,{img_base64}',
            }
        )

    except RequestEntityTooLarge:
        raise  # delegate to the registered errorhandler -> 413
    except Exception:
        logger.exception('Unexpected error in /detect')
        return jsonify({'error': 'Loi xu ly anh tren server'}), 500


@app.route('/detect_base64', methods=['POST'])
def detect_base64():
    """Detect emotions from webcam base64 frame."""
    current_detector = get_detector()
    if current_detector is None:
        return jsonify({'error': 'Model chua duoc load'}), 500

    try:
        image, error = decode_base64_image(request.get_json(silent=True))
        if error:
            message, status_code = error
            return jsonify({'error': message}), status_code

        face_analyses = current_detector.analyze_faces(image)
        if not face_analyses:
            return jsonify({'success': True, 'faces_count': 0, 'faces': []})

        faces, _ = build_faces_response(face_analyses)

        return jsonify({'success': True, 'faces_count': len(faces), 'faces': faces})

    except RequestEntityTooLarge:
        raise  # delegate to the registered errorhandler -> 413
    except Exception:
        logger.exception('Unexpected error in /detect_base64')
        return jsonify({'error': 'Loi xu ly frame tren server'}), 500


@app.route('/health')
def health():
    """Health check."""
    return jsonify(
        {
            'status': 'ok',
            'model_loaded': detector is not None,
            'model_path': str(MODEL_PATH),
            'model_found': MODEL_PATH.exists(),
        }
    )


if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('EMOTION DETECTION WEB APP')
    logger.info('=' * 60)

    init_detector()

    if detector is None:
        logger.warning('Model not loaded — detection endpoints will return errors.')
        logger.warning('Run training first: python src/train.py')

    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', '0').lower() in {'1', 'true', 'yes'}

    logger.info('Starting server at http://%s:%d (debug=%s)', host, port, debug)
    app.run(debug=debug, host=host, port=port)
