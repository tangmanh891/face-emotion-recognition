"""Pluggable face detection backends.

Two backends with a common interface:
- HaarFaceDetector: OpenCV's Viola-Jones cascade (legacy, fast, low recall on
  side profiles / low light).
- MediaPipeFaceDetector: Google BlazeFace via MediaPipe Tasks API
  (CNN-based, higher recall on diverse poses).

Both return face bounding boxes as `[(x, y, w, h), ...]` so they are
drop-in interchangeable for downstream emotion classification.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FaceBox = tuple[int, int, int, int]  # (x, y, w, h)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MEDIAPIPE_MODEL = BASE_DIR / 'models' / 'blaze_face_short_range.tflite'


class FaceDetectorBase(ABC):
    """Common interface for all face detectors."""

    backend_name: str = 'base'

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        """Return face boxes for a BGR image."""

    def close(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release any backend resources. Override if the backend needs cleanup."""


class HaarFaceDetector(FaceDetectorBase):
    """OpenCV Haar Cascade face detector (frontal faces only)."""

    backend_name = 'haar'

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
    ):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f'Failed to load Haar cascade at {cascade_path}')
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]


class MediaPipeFaceDetector(FaceDetectorBase):
    """Google BlazeFace via MediaPipe Tasks API.

    Uses the short-range model (~2m optimal). For far-away faces, use the
    full-range model — same interface, different .tflite asset.
    """

    backend_name = 'mediapipe'

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MEDIAPIPE_MODEL,
        min_detection_confidence: float = 0.5,
    ):
        # Lazy import so the module is usable without mediapipe installed
        # as long as the user only picks the Haar backend.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f'MediaPipe face model not found at {model_path}. '
                'Download with:\n'
                '  curl -L -o models/blaze_face_short_range.tflite '
                'https://storage.googleapis.com/mediapipe-models/face_detector/'
                'blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'
            )

        self._mp = mp
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        # MediaPipe expects RGB.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)

        boxes: list[FaceBox] = []
        h_img, w_img = image_bgr.shape[:2]
        for detection in result.detections:
            bbox = detection.bounding_box
            # MediaPipe sometimes returns slightly negative origins; clamp.
            x = max(0, int(bbox.origin_x))
            y = max(0, int(bbox.origin_y))
            w = min(int(bbox.width), w_img - x)
            h = min(int(bbox.height), h_img - y)
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))
        return boxes

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                logger.debug('MediaPipe detector close() raised', exc_info=True)
            self._detector = None


def create_face_detector(backend: str = 'mediapipe', **kwargs) -> FaceDetectorBase:
    """Factory: build a detector by backend name.

    backend: 'mediapipe' (default) | 'haar'
    """
    backend = (backend or 'mediapipe').lower()
    if backend == 'mediapipe':
        return MediaPipeFaceDetector(**kwargs)
    if backend == 'haar':
        return HaarFaceDetector(**kwargs)
    raise ValueError(f'Unknown face detector backend: {backend!r}')
