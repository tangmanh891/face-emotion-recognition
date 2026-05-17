# Face detection benchmark samples

Drop JPG/PNG images here, then run:

```bash
python scripts/benchmark_detectors.py --images tests/face_samples --save-visualizations
```

For a meaningful Haar vs MediaPipe comparison, try to include a mix of:

- **Frontal portraits** — both detectors should find these.
- **Side profiles** — Haar typically fails; MediaPipe should still detect.
- **Low-light / shadows** — stress test for Haar.
- **Group photos / multi-face** — measures recall on multiple faces.
- **Occlusion** — masks, sunglasses, hands near face.
- **Different scales** — tiny faces vs close-ups.

Sources for public-domain photos: Wikipedia Commons, Pexels (CC0), Unsplash.

This folder is checked in so the benchmark can run in CI, but the bundled
sample set is intentionally small. Add more locally for better coverage.
