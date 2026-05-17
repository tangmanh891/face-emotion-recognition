"""Emotion detector utilities."""

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

try:
    from src.face_detectors import FaceDetectorBase, create_face_detector
except ImportError:  # pragma: no cover - allow `from emotion_detector import ...`
    from face_detectors import FaceDetectorBase, create_face_detector

logger = logging.getLogger(__name__)


# Canonical English labels — used as fallback when class_indices.json is missing.
# Order matches the alphabetical sort that Keras flow_from_directory applies.
DEFAULT_EMOTIONS_EN = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Vietnamese display labels mapped from the English folder name.
EMOTIONS_VI_MAP = {
    'Angry': 'Tức giận',
    'Disgust': 'Ghê tởm',
    'Fear': 'Sợ hãi',
    'Happy': 'Vui vẻ',
    'Sad': 'Buồn bã',
    'Surprise': 'Ngạc nhiên',
    'Neutral': 'Bình thường',
}


class EmotionDetector:
    """Detect facial emotions from images."""

    def __init__(
        self,
        model_path,
        use_vietnamese=False,
        class_indices_path=None,
        face_detector: FaceDetectorBase | str | None = None,
    ):
        """Load the classifier and a face detector.

        face_detector:
            - None  → use env var EMOTION_FACE_BACKEND (default 'mediapipe')
            - str   → 'mediapipe' | 'haar'
            - FaceDetectorBase instance → use as-is
        """
        self.model = keras.models.load_model(model_path)
        self.face_detector = self._build_face_detector(face_detector)

        class_names_en = self._load_class_names(model_path, class_indices_path)
        if use_vietnamese:
            self.emotions = [EMOTIONS_VI_MAP.get(name, name) for name in class_names_en]
        else:
            self.emotions = class_names_en

    @staticmethod
    def _build_face_detector(spec):
        if isinstance(spec, FaceDetectorBase):
            return spec
        backend = spec if isinstance(spec, str) else os.environ.get(
            'EMOTION_FACE_BACKEND', 'mediapipe'
        )
        try:
            detector = create_face_detector(backend)
            logger.info('Using face detector backend: %s', detector.backend_name)
            return detector
        except Exception:
            if backend == 'mediapipe':
                logger.warning(
                    'MediaPipe backend failed to initialize — falling back to Haar. '
                    'Run with EMOTION_FACE_BACKEND=haar to silence this.',
                    exc_info=True,
                )
                return create_face_detector('haar')
            raise

    @staticmethod
    def _load_class_names(model_path, class_indices_path):
        """Load class names in the exact order the model was trained on.

        Why: training uses ImageDataGenerator.class_indices which sorts
        folders alphabetically. Reading this mapping from disk prevents
        silent mislabeling if folders are renamed or reordered.
        """
        if class_indices_path is None:
            class_indices_path = Path(model_path).parent / 'class_indices.json'

        class_indices_path = Path(class_indices_path)
        if not class_indices_path.exists():
            logger.warning(
                'class_indices.json not found at %s — falling back to default order. '
                'Re-train with src/train.py to generate it.',
                class_indices_path,
            )
            return list(DEFAULT_EMOTIONS_EN)

        with class_indices_path.open('r', encoding='utf-8') as f:
            mapping = json.load(f)

        # mapping is {class_name: index}; invert and sort by index.
        ordered = sorted(mapping.items(), key=lambda item: item[1])
        return [name.capitalize() for name, _ in ordered]

    def detect_faces(self, image):
        """Detect face regions (x, y, w, h) from a BGR image."""
        return self.face_detector.detect(image)

    def preprocess_face(self, face_img):
        """Prepare face image for model input (1, 48, 48, 1)."""
        face_img = cv2.resize(face_img, (48, 48))

        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        face_img = face_img / 255.0
        return face_img.reshape(1, 48, 48, 1)

    def predict_emotion(self, face_img):
        """Predict emotion, confidence (%), and class probabilities (%)."""
        processed = self.preprocess_face(face_img)
        predictions = self.model.predict(processed, verbose=0)[0]

        emotion_idx = int(np.argmax(predictions))
        emotion = self.emotions[emotion_idx]
        confidence = float(predictions[emotion_idx] * 100)

        probabilities = {
            self.emotions[i]: float(predictions[i] * 100)
            for i in range(len(self.emotions))
        }

        return emotion, confidence, probabilities

    def analyze_faces(self, image):
        """Return detailed analysis for each face including probabilities."""
        analyses = []

        for (x, y, w, h) in self.detect_faces(image):
            face_img = image[y : y + h, x : x + w]
            if face_img.size == 0:
                continue
            emotion, confidence, probabilities = self.predict_emotion(face_img)

            analyses.append(
                {
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h),
                    'emotion': emotion,
                    'confidence': confidence,
                    'probabilities': probabilities,
                }
            )

        return analyses

    def detect_emotion_from_image(self, image):
        """Backward-compatible tuple output for existing callers."""
        analyses = self.analyze_faces(image)
        return [
            (
                face['x'],
                face['y'],
                face['w'],
                face['h'],
                face['emotion'],
                face['confidence'],
            )
            for face in analyses
        ]

    def draw_results(self, image, results):
        """Draw face bounding boxes and emotion labels."""
        image_copy = image.copy()

        for (x, y, w, h, emotion, confidence) in results:
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = f"{emotion}: {confidence:.1f}%"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                image_copy,
                (x, y - text_height - 10),
                (x + text_width, y),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                image_copy,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return image_copy


if __name__ == '__main__':
    import os

    logging.basicConfig(level=logging.INFO)

    model_path = 'models/emotion_model.keras'
    if not os.path.exists(model_path):
        model_path = 'models/emotion_model.h5'

    if not os.path.exists(model_path):
        logger.error('Model not found. Run training first: python src/train.py')
    else:
        detector = EmotionDetector(model_path, use_vietnamese=True)
        logger.info('Detector ready. Emotions: %s', ', '.join(detector.emotions))
