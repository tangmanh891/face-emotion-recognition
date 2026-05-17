"""Tests for the pluggable face detector backends."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_detectors import (
    HaarFaceDetector,
    MediaPipeFaceDetector,
    create_face_detector,
)


class FactoryTests(unittest.TestCase):
    def test_factory_returns_haar(self):
        detector = create_face_detector('haar')
        self.assertIsInstance(detector, HaarFaceDetector)
        self.assertEqual(detector.backend_name, 'haar')

    def test_factory_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_face_detector('definitely-not-a-backend')

    def test_factory_default_is_mediapipe(self):
        # Lazy import: only check the type if mediapipe + model file present.
        try:
            detector = create_face_detector()
        except (ImportError, FileNotFoundError) as exc:
            self.skipTest(f'mediapipe unavailable in this env: {exc}')
        self.assertEqual(detector.backend_name, 'mediapipe')


class HaarDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = HaarFaceDetector()

    def test_blank_image_returns_no_faces(self):
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = self.detector.detect(blank)
        self.assertEqual(faces, [])

    def test_returns_list_of_int_tuples(self):
        blank = np.full((50, 50, 3), 128, dtype=np.uint8)
        faces = self.detector.detect(blank)
        self.assertIsInstance(faces, list)
        for face in faces:
            self.assertEqual(len(face), 4)
            for v in face:
                self.assertIsInstance(v, int)


class MediaPipeDetectorMissingModelTests(unittest.TestCase):
    def test_missing_model_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            MediaPipeFaceDetector(model_path='/nonexistent/blaze.tflite')


class MediaPipeDetectorBoundingBoxTests(unittest.TestCase):
    """Verify the MediaPipe wrapper clamps boxes to image bounds.

    Uses a mocked underlying detector to keep the test runnable without the
    mediapipe model file installed.
    """

    def _build_detector_with_mock(self, raw_detections):
        with patch.dict('sys.modules', {
            'mediapipe': MagicMock(),
            'mediapipe.tasks': MagicMock(),
            'mediapipe.tasks.python': MagicMock(),
            'mediapipe.tasks.python.vision': MagicMock(),
        }):
            detector = MediaPipeFaceDetector.__new__(MediaPipeFaceDetector)
            detector._mp = MagicMock()
            detector._mp.Image = MagicMock(return_value='mp_image_stub')
            detector._mp.ImageFormat.SRGB = 'srgb'
            mock_result = MagicMock()
            mock_result.detections = raw_detections
            detector._detector = MagicMock()
            detector._detector.detect.return_value = mock_result
        return detector

    def _make_detection(self, x, y, w, h):
        det = MagicMock()
        det.bounding_box.origin_x = x
        det.bounding_box.origin_y = y
        det.bounding_box.width = w
        det.bounding_box.height = h
        return det

    def test_clamps_negative_origin(self):
        detector = self._build_detector_with_mock([self._make_detection(-5, -3, 50, 50)])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect(img)
        self.assertEqual(result, [(0, 0, 50, 50)])

    def test_clamps_width_overflow(self):
        detector = self._build_detector_with_mock([self._make_detection(80, 80, 50, 50)])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect(img)
        # 80+50 > 100, so width clamped to 20.
        self.assertEqual(result, [(80, 80, 20, 20)])

    def test_drops_zero_size_boxes(self):
        detector = self._build_detector_with_mock([self._make_detection(100, 100, 50, 50)])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect(img)
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
