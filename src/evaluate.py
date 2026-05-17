"""Evaluate a trained emotion model on the test set.

Outputs:
- Per-class precision/recall/F1 (classification report)
- Confusion matrix as both PNG and JSON
- Inference latency benchmark (ms per face)
- Summary JSON suitable for embedding in the README
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
TEST_DIR = DATA_DIR / 'test'
MODEL_DIR = BASE_DIR / 'models'
DEFAULT_MODEL_PATH = MODEL_DIR / 'emotion_model.keras'
CLASS_INDICES_PATH = MODEL_DIR / 'class_indices.json'

IMG_SIZE = 48
BATCH_SIZE = 64
BENCHMARK_RUNS = 100

logger = logging.getLogger(__name__)


def resolve_model_path(user_path):
    if user_path is not None:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f'Model not found: {path}')
        return path

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    legacy = MODEL_DIR / 'emotion_model.h5'
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        f'No model found at {DEFAULT_MODEL_PATH} or {legacy}. '
        'Run training first: python src/train.py'
    )


def load_class_names():
    """Load class names from the file written by train.py."""
    if not CLASS_INDICES_PATH.exists():
        logger.warning(
            'class_indices.json not found — falling back to alphabetical '
            'subdirectory names from the test set.'
        )
        return sorted(p.name for p in TEST_DIR.iterdir() if p.is_dir())

    with CLASS_INDICES_PATH.open('r', encoding='utf-8') as f:
        mapping = json.load(f)
    ordered = sorted(mapping.items(), key=lambda item: item[1])
    return [name for name, _ in ordered]


def build_test_generator():
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return test_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False,
    )


def evaluate_model(model, generator, class_names):
    logger.info('Predicting on %d test samples...', generator.samples)
    probabilities = model.predict(generator, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = generator.classes[: len(y_pred)]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'classification_report': report,
        'confusion_matrix': cm,
    }


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot a normalized confusion matrix (row-wise = recall per class)."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Recall'},
        ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (normalized by true class)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info('Saved confusion matrix to %s', output_path)


def benchmark_inference(model, runs=BENCHMARK_RUNS):
    """Measure single-face inference latency."""
    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
    # Warm-up to exclude one-time graph compilation cost.
    for _ in range(5):
        model.predict(dummy, verbose=0)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(dummy, verbose=0)
        timings.append((time.perf_counter() - start) * 1000.0)

    return {
        'runs': runs,
        'mean_ms': float(np.mean(timings)),
        'median_ms': float(np.median(timings)),
        'p95_ms': float(np.percentile(timings, 95)),
    }


def write_summary(results, latency, class_names, output_path):
    report = results['classification_report']
    per_class = {
        name: {
            'precision': round(report[name]['precision'], 4),
            'recall': round(report[name]['recall'], 4),
            'f1': round(report[name]['f1-score'], 4),
            'support': int(report[name]['support']),
        }
        for name in class_names
        if name in report
    }

    summary = {
        'accuracy': round(results['accuracy'], 4),
        'macro_f1': round(results['macro_f1'], 4),
        'weighted_f1': round(results['weighted_f1'], 4),
        'per_class': per_class,
        'inference_latency_ms': {
            'mean': round(latency['mean_ms'], 3),
            'median': round(latency['median_ms'], 3),
            'p95': round(latency['p95_ms'], 3),
            'runs': latency['runs'],
        },
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'class_names': class_names,
    }

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info('Saved evaluation summary to %s', output_path)
    return summary


def print_human_summary(summary):
    print()
    print('=' * 60)
    print('EVALUATION SUMMARY')
    print('=' * 60)
    print(f"Test accuracy : {summary['accuracy'] * 100:.2f}%")
    print(f"Macro F1      : {summary['macro_f1']:.4f}")
    print(f"Weighted F1   : {summary['weighted_f1']:.4f}")
    print()
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print('-' * 56)
    for name, scores in summary['per_class'].items():
        print(
            f'{name:<12} {scores["precision"]:>10.4f} {scores["recall"]:>10.4f} '
            f'{scores["f1"]:>10.4f} {scores["support"]:>10d}'
        )
    print()
    lat = summary['inference_latency_ms']
    print(
        f"Inference     : mean={lat['mean']:.2f}ms  median={lat['median']:.2f}ms  "
        f"p95={lat['p95']:.2f}ms  (n={lat['runs']})"
    )
    print('=' * 60)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Path to a .keras or .h5 model file')
    parser.add_argument(
        '--output-dir',
        default=str(MODEL_DIR),
        help='Directory to write evaluation artifacts',
    )
    parser.add_argument(
        '--no-benchmark',
        action='store_true',
        help='Skip the inference latency benchmark',
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    args = parse_args()

    if not TEST_DIR.exists():
        logger.error('Test data not found at %s', TEST_DIR)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model)
    logger.info('Loading model: %s', model_path)
    model = keras.models.load_model(model_path)

    class_names = load_class_names()
    logger.info('Class names: %s', class_names)

    generator = build_test_generator()
    results = evaluate_model(model, generator, class_names)

    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names,
        output_dir / 'confusion_matrix.png',
    )

    if args.no_benchmark:
        latency = {'runs': 0, 'mean_ms': 0.0, 'median_ms': 0.0, 'p95_ms': 0.0}
    else:
        logger.info('Benchmarking inference latency...')
        latency = benchmark_inference(model)

    summary = write_summary(
        results, latency, class_names, output_dir / 'evaluation.json'
    )
    print_human_summary(summary)


if __name__ == '__main__':
    main()
