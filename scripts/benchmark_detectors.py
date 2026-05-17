"""Benchmark face detector backends on a directory of test images.

Compares Haar Cascade vs MediaPipe BlazeFace on two axes:
1. Detection coverage — how many images contain at least one detected face,
   and the total number of faces found.
2. Latency — per-image inference time (ms).

Usage:
    python scripts/benchmark_detectors.py --images <dir>
    python scripts/benchmark_detectors.py --images <dir> --save-visualizations

By default scans `tests/face_samples/` for *.jpg, *.jpeg, *.png, *.webp.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_detectors import create_face_detector  # noqa: E402

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
DEFAULT_IMAGE_DIR = PROJECT_ROOT / 'tests' / 'face_samples'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'models' / 'detector_benchmark'

logger = logging.getLogger(__name__)


def collect_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(p for p in image_dir.rglob('*') if p.suffix.lower() in IMAGE_EXTS)


def draw_boxes(image: np.ndarray, boxes, color, label: str) -> np.ndarray:
    out = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        out,
        f'{label}: {len(boxes)} face(s)',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    return out


def benchmark_backend(
    backend: str,
    images: list[Path],
    save_dir: Path | None,
) -> dict:
    detector = create_face_detector(backend)
    color = (0, 255, 0) if backend == 'mediapipe' else (0, 165, 255)

    per_image = []
    total_faces = 0
    images_with_faces = 0
    latencies_ms: list[float] = []

    try:
        # Warmup — first call has graph compilation overhead.
        if images:
            warm = cv2.imread(str(images[0]))
            if warm is not None:
                for _ in range(2):
                    detector.detect(warm)

        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                logger.warning('Could not read %s', path)
                continue

            t0 = time.perf_counter()
            boxes = detector.detect(image)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            latencies_ms.append(latency_ms)
            total_faces += len(boxes)
            if boxes:
                images_with_faces += 1

            per_image.append(
                {
                    'image': path.name,
                    'faces': len(boxes),
                    'latency_ms': round(latency_ms, 2),
                }
            )

            if save_dir is not None:
                annotated = draw_boxes(image, boxes, color, backend)
                out_path = save_dir / f'{path.stem}__{backend}.jpg'
                cv2.imwrite(str(out_path), annotated)
    finally:
        detector.close()

    summary = {
        'backend': backend,
        'images_evaluated': len(per_image),
        'images_with_faces': images_with_faces,
        'total_faces_detected': total_faces,
        'latency_ms': {
            'mean': round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
            'median': round(float(np.median(latencies_ms)), 2) if latencies_ms else 0.0,
            'p95': round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
        },
        'per_image': per_image,
    }
    return summary


def print_human_summary(results: list[dict]) -> None:
    print()
    print('=' * 72)
    print(f"{'Backend':<12} {'Images':>8} {'WithFace':>10} {'TotalFaces':>12} "
          f"{'Mean(ms)':>10} {'p95(ms)':>10}")
    print('-' * 72)
    for r in results:
        lat = r['latency_ms']
        print(
            f"{r['backend']:<12} {r['images_evaluated']:>8d} "
            f"{r['images_with_faces']:>10d} {r['total_faces_detected']:>12d} "
            f"{lat['mean']:>10.2f} {lat['p95']:>10.2f}"
        )
    print('=' * 72)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--images',
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f'Directory of test images (default: {DEFAULT_IMAGE_DIR})',
    )
    parser.add_argument(
        '--backends',
        nargs='+',
        default=['haar', 'mediapipe'],
        choices=['haar', 'mediapipe'],
    )
    parser.add_argument(
        '--save-visualizations',
        action='store_true',
        help='Save annotated images to <output-dir>/visualizations/',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Where to write results (default: {DEFAULT_OUTPUT_DIR})',
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    args = parse_args()

    images = collect_images(args.images)
    if not images:
        logger.error(
            'No images found in %s. Add some JPG/PNG portraits there and re-run. '
            'Tip: try a mix of frontal, side-profile, low-light, and group photos.',
            args.images,
        )
        sys.exit(1)

    logger.info('Found %d test images', len(images))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = args.output_dir / 'visualizations' if args.save_visualizations else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for backend in args.backends:
        logger.info('Benchmarking backend: %s', backend)
        results.append(benchmark_backend(backend, images, viz_dir))

    summary_path = args.output_dir / 'benchmark.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump({'image_count': len(images), 'results': results}, f, indent=2)
    logger.info('Saved benchmark JSON to %s', summary_path)

    print_human_summary(results)


if __name__ == '__main__':
    main()
