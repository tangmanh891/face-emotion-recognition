"""Training script for the emotion recognition CNN model."""

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import create_emotion_model

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('FER_DATA_DIR', BASE_DIR / 'data'))
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
MODEL_DIR = BASE_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'emotion_model.keras'
CLASS_INDICES_PATH = MODEL_DIR / 'class_indices.json'

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50

logger = logging.getLogger(__name__)


def prepare_data():
    """Prepare train/validation/test generators."""
    logger.info('Preparing data generators...')

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2,
    )

    # Validation must be deterministic — no augmentation.
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training',
    )

    validation_generator = validation_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
    )

    test_generator = test_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False,
    )

    logger.info('Training samples: %d', train_generator.samples)
    logger.info('Validation samples: %d', validation_generator.samples)
    logger.info('Test samples: %d', test_generator.samples)
    logger.info('Classes: %s', train_generator.class_indices)

    return train_generator, validation_generator, test_generator


def save_class_indices(class_indices):
    """Persist generator's class_indices mapping so inference can load it.

    Why: Keras flow_from_directory orders classes alphabetically by folder
    name. Hardcoding the label list at inference time is a latent bug —
    rename one folder and predictions silently mislabel.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with CLASS_INDICES_PATH.open('w', encoding='utf-8') as f:
        json.dump(class_indices, f, ensure_ascii=False, indent=2)
    logger.info('Saved class_indices to %s', CLASS_INDICES_PATH)


def train_model():
    """Train and evaluate model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, test_gen = prepare_data()
    save_class_indices(train_gen.class_indices)

    logger.info('Building model...')
    model = create_emotion_model()
    model.summary(print_fn=logger.info)

    callbacks = [
        ModelCheckpoint(
            str(MODEL_PATH),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    logger.info('Starting training for %d epochs...', EPOCHS)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info('Evaluating on test set...')
    test_loss, test_acc = model.evaluate(test_gen)
    logger.info('Test accuracy: %.2f%%', test_acc * 100)
    logger.info('Test loss: %.4f', test_loss)

    plot_history(history)

    return model, history


def plot_history(history):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    output_path = MODEL_DIR / 'training_history.png'
    plt.tight_layout()
    plt.savefig(str(output_path))
    logger.info('Saved training history plot to %s', output_path)

    if os.environ.get('SHOW_PLOTS', '0') == '1':
        plt.show()
    else:
        plt.close(fig)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    logger.info('=' * 60)
    logger.info('TRAINING EMOTION RECOGNITION MODEL')
    logger.info('=' * 60)

    missing_dirs = [path for path in (TRAIN_DIR, TEST_DIR) if not path.exists()]
    if missing_dirs:
        logger.error('Dataset directories not found:')
        for missing_path in missing_dirs:
            logger.error(' - %s', missing_path)
        logger.error('Download FER-2013 and place it in data/train and data/test.')
        sys.exit(1)

    train_model()

    logger.info('=' * 60)
    logger.info('DONE. Model saved at: %s', MODEL_PATH)
    logger.info('Class indices saved at: %s', CLASS_INDICES_PATH)
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
