"""
Script huấn luyện model nhận diện cảm xúc
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import create_emotion_model

# Constants
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_PATH = 'models/emotion_model.h5'

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def prepare_data():
    """
    Chuẩn bị data generators với data augmentation
    """
    print("Chuẩn bị dữ liệu...")
    
    # Data augmentation cho training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% cho validation
    )
    
    # Chỉ rescale cho test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training'
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
    )
    
    print(f"✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {validation_generator.samples}")
    print(f"✓ Test samples: {test_generator.samples}")
    print(f"✓ Classes: {train_generator.class_indices}")
    
    return train_generator, validation_generator, test_generator

def train_model():
    """
    Huấn luyện model
    """
    # Tạo thư mục models nếu chưa có
    os.makedirs('models', exist_ok=True)
    
    # Chuẩn bị dữ liệu
    train_gen, val_gen, test_gen = prepare_data()
    
    # Tạo model
    print("\nTạo model...")
    model = create_emotion_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        # Lưu model tốt nhất
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Giảm learning rate khi plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Training
    print("\nBắt đầu training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate trên test set
    print("\nĐánh giá trên test set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history
    plot_history(history)
    
    return model, history

def plot_history(history):
    """
    Vẽ đồ thị training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("\n✓ Đồ thị lưu tại: models/training_history.png")
    plt.show()

if __name__ == '__main__':
    print("=" * 60)
    print("HUẤN LUYỆN MODEL NHẬN DIỆN CẢM XÚC")
    print("=" * 60)
    
    # Kiểm tra dataset
    if not os.path.exists(TRAIN_DIR):
        print(f"\n❌ Không tìm thấy dataset tại: {TRAIN_DIR}")
        print("Vui lòng chạy: python src/download_data.py")
        exit(1)
    
    # Train
    model, history = train_model()
    
    print("\n" + "=" * 60)
    print("✓ HOÀN THÀNH!")
    print(f"✓ Model đã lưu tại: {MODEL_PATH}")
    print("=" * 60)
