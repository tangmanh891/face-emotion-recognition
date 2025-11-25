# Nhận diện Cảm xúc Khuôn mặt (Face Emotion Recognition)

Dự án sử dụng Deep Learning để nhận diện 7 loại cảm xúc từ khuôn mặt người: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

## 🎯 Tính năng

- ✅ Huấn luyện mô hình CNN trên dataset FER-2013
- ✅ Nhận diện cảm xúc real-time qua webcam
- ✅ Nhận diện từ ảnh/video
- ✅ Web interface để demo
- ✅ Hiển thị biểu đồ phân bố cảm xúc

## 📊 Dataset

Sử dụng **FER-2013** từ Kaggle:
- 35,887 ảnh khuôn mặt grayscale 48x48 pixels
- 7 loại cảm xúc
- Link: https://www.kaggle.com/datasets/msambare/fer2013

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/yourusername/face-emotion-recognition.git
cd face-emotion-recognition
```

### 2. Tạo môi trường ảo
```bash
# Tạo môi trường conda với Python 3.10
conda create -n emotion-recognition python=3.10 -y

# Kích hoạt môi trường
conda activate emotion-recognition
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```


### 4. Tải dataset
Có 2 cách:

**Cách 1: Sử dụng Kaggle API**
```bash
# Đăng ký tài khoản Kaggle và tải kaggle.json
# Đặt kaggle.json vào: ~/.kaggle/ (Linux/Mac) hoặc C:\Users\<username>\.kaggle\ (Windows)
python src/download_data.py
```

**Cách 2: Tải thủ công**
- Truy cập: https://www.kaggle.com/datasets/msambare/fer2013
- Tải về và giải nén vào thư mục `data/`

## 📚 Sử dụng

### 1. Huấn luyện model

**Cách 1: Local (CPU/GPU)**
```bash
python src/train.py
```

**Cách 2: Google Colab**
1. Mở [train_colab.ipynb](train_colab.ipynb) trong Google Colab
2. Runtime → Change runtime type → **GPU (T4)**
3. Chạy từng cell theo thứ tự
4. Tải model về sau khi train xong

### 2. Nhận diện real-time (webcam)
```bash
python src/realtime_detection.py
```

### 3. Chạy web interface
```bash
python app.py
```
Truy cập: http://localhost:5000

### 4. Nhận diện từ ảnh
```python
from src.emotion_detector import EmotionDetector

detector = EmotionDetector('models/emotion_model.h5')
emotion, confidence = detector.predict_emotion('path/to/image.jpg')
print(f"Cảm xúc: {emotion}, Độ tin cậy: {confidence:.2f}%")
```

## 🏗️ Cấu trúc dự án

```
face-emotion-recognition/
│
├── data/                       # Dataset
├── models/                     # Trained models
├── src/                        # Source code
│   ├── model.py               # CNN model architecture
│   ├── train.py               # Training script
│   ├── emotion_detector.py    # Detection class
│   ├── realtime_detection.py  # Webcam detection
│   └── download_data.py       # Download dataset
├── notebooks/                  # Jupyter notebooks
│   └── emotion_recognition.ipynb
├── static/                     # Web static files
│   ├── css/
│   └── js/
├── templates/                  # HTML templates
│   └── index.html
├── app.py                      # Flask web app
├── requirements.txt
└── README.md
```

## 🧠 Kiến trúc Model

Mô hình CNN gồm:
- 4 Convolutional blocks (Conv2D + BatchNorm + MaxPooling + Dropout)
- 2 Fully Connected layers
- Output: 7 classes (softmax activation)

## 📈 Kết quả

- Training Accuracy: ~65%
- Validation Accuracy: ~60%
- Test Accuracy: ~58%

## 🛠️ Công nghệ sử dụng

- **TensorFlow/Keras**: Xây dựng và huấn luyện model
- **OpenCV**: Xử lý ảnh và video
- **Flask**: Web framework
- **NumPy, Pandas**: Xử lý dữ liệu
- **Matplotlib, Seaborn**: Visualization

## 📝 Ghi chú

- Model hoạt động tốt nhất với ảnh có ánh sáng tốt
- Khuôn mặt nên ở chính giữa và rõ ràng
- Có thể cải thiện độ chính xác bằng cách:
  - Tăng kích thước dataset
  - Data augmentation
  - Transfer learning (VGG, ResNet)
  - Ensemble methods

## 📄 License

MIT License

## 👨‍💻 Tác giả

[Your Name]

## 🤝 Đóng góp

Pull requests are welcome! For major changes, please open an issue first.
