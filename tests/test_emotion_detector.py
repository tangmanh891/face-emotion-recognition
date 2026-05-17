"""Tests for EmotionDetector that don't require the trained Keras model."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the module (not the class) so we can patch keras.models.load_model.
import src.emotion_detector as emotion_detector
from src.emotion_detector import (
    DEFAULT_EMOTIONS_EN,
    EMOTIONS_VI_MAP,
    EmotionDetector,
)
from src.face_detectors import FaceDetectorBase


class _FakeFaceDetector(FaceDetectorBase):
    """Concrete subclass so isinstance() in _build_face_detector matches."""

    backend_name = 'fake'

    def __init__(self, boxes=None):
        self.boxes = list(boxes or [])
        self.detect_calls = 0

    def detect(self, image):
        self.detect_calls += 1
        return list(self.boxes)


def _new_detector(class_indices=None, use_vietnamese=False, predictions=None, boxes=None):
    """Build an EmotionDetector with a mocked keras model + fake face detector."""
    fake_model = MagicMock()
    if predictions is not None:
        fake_model.predict.return_value = np.array([predictions])

    with patch.object(
        emotion_detector.keras.models, 'load_model', return_value=fake_model
    ), patch.object(
        EmotionDetector, '_load_class_names',
        return_value=class_indices or DEFAULT_EMOTIONS_EN,
    ):
        detector = EmotionDetector(
            model_path='ignored.keras',
            use_vietnamese=use_vietnamese,
            face_detector=_FakeFaceDetector(boxes),
        )
    detector._fake_model = fake_model  # exposed for assertions
    return detector


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.detector = _new_detector()

    def test_color_face_is_grayscaled(self):
        face = np.full((96, 96, 3), 200, dtype=np.uint8)
        out = self.detector.preprocess_face(face)
        self.assertEqual(out.shape, (1, 48, 48, 1))
        # All pixels were 200/255 ≈ 0.784
        np.testing.assert_allclose(out.mean(), 200.0 / 255.0, atol=0.01)

    def test_grayscale_face_kept(self):
        face = np.full((30, 30), 100, dtype=np.uint8)
        out = self.detector.preprocess_face(face)
        self.assertEqual(out.shape, (1, 48, 48, 1))
        np.testing.assert_allclose(out.mean(), 100.0 / 255.0, atol=0.01)

    def test_resizes_non_square_face(self):
        face = np.zeros((10, 80, 3), dtype=np.uint8)
        out = self.detector.preprocess_face(face)
        self.assertEqual(out.shape, (1, 48, 48, 1))


class PredictEmotionTests(unittest.TestCase):
    def test_returns_top_class_and_full_probabilities(self):
        # softmax favoring class 3 (= 'Happy' in default order)
        detector = _new_detector(predictions=[0.05, 0.02, 0.03, 0.80, 0.05, 0.03, 0.02])
        face = np.zeros((48, 48), dtype=np.uint8)

        emotion, confidence, probs = detector.predict_emotion(face)

        self.assertEqual(emotion, 'Happy')
        self.assertAlmostEqual(confidence, 80.0, places=4)
        self.assertEqual(set(probs.keys()), set(DEFAULT_EMOTIONS_EN))
        self.assertAlmostEqual(sum(probs.values()), 100.0, places=2)

    def test_vietnamese_labels_used_when_enabled(self):
        detector = _new_detector(
            use_vietnamese=True,
            predictions=[0.9, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02],
        )
        face = np.zeros((48, 48), dtype=np.uint8)

        emotion, _, probs = detector.predict_emotion(face)

        self.assertEqual(emotion, EMOTIONS_VI_MAP['Angry'])
        self.assertIn(EMOTIONS_VI_MAP['Happy'], probs)


class ClassIndicesLoadingTests(unittest.TestCase):
    def test_loads_alphabetical_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            ci_path = Path(tmp) / 'class_indices.json'
            ci_path.write_text(
                json.dumps({'happy': 2, 'angry': 0, 'sad': 1}), encoding='utf-8'
            )
            names = EmotionDetector._load_class_names('ignored.keras', ci_path)
        # Sorted by index value, then capitalized to match EMOTIONS_VI_MAP keys.
        self.assertEqual(names, ['Angry', 'Sad', 'Happy'])

    def test_falls_back_when_json_missing(self):
        # Point at a path that genuinely does not exist.
        missing = Path(tempfile.gettempdir()) / 'definitely-missing-class-indices.json'
        if missing.exists():
            missing.unlink()
        names = EmotionDetector._load_class_names('ignored.keras', missing)
        self.assertEqual(names, list(DEFAULT_EMOTIONS_EN))


class AnalyzeFacesIntegrationTests(unittest.TestCase):
    def test_returns_per_face_payload(self):
        detector = _new_detector(
            predictions=[0.05, 0.02, 0.03, 0.80, 0.05, 0.03, 0.02],
            boxes=[(10, 20, 30, 40), (60, 70, 25, 25)],
        )
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detector.analyze_faces(img)

        self.assertEqual(len(result), 2)
        for face, expected_box in zip(
            result, [(10, 20, 30, 40), (60, 70, 25, 25)], strict=True
        ):
            self.assertEqual(
                (face['x'], face['y'], face['w'], face['h']), expected_box
            )
            self.assertEqual(face['emotion'], 'Happy')
            self.assertIn('probabilities', face)

    def test_skips_zero_size_face_crop(self):
        # Box outside image bounds → crop is empty.
        detector = _new_detector(boxes=[(500, 500, 10, 10)])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual(detector.analyze_faces(img), [])
        detector._fake_model.predict.assert_not_called()


if __name__ == '__main__':
    unittest.main()
