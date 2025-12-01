# 🎭 Face Emotion Recognition

Hệ thống nhận diện cảm xúc khuôn mặt sử dụng Deep Learning (CNN) với giao diện web tương tác. Dự án hỗ trợ nhận diện 7 loại cảm xúc cơ bản từ ảnh và video real-time.

## 🎬 Demo

<div align="center">
  <img src="templates/demo.png" alt="Demo Web Interface" width="800"/>
  <p><i>Giao diện web nhận diện cảm xúc khuôn mặt</i></p>
</div>

## ✨ Tính năng

- 🤖 **Nhận diện 7 cảm xúc**: Tức giận, Ghê tởm, Sợ hãi, Vui vẻ, Buồn bã, Ngạc nhiên, Bình thường
- 📷 **Xử lý ảnh**: Upload và nhận diện cảm xúc từ ảnh
- 🎥 **Real-time Detection**: Nhận diện cảm xúc trực tiếp qua webcam
- 🌐 **Web Interface**: Giao diện web đẹp mắt, dễ sử dụng với Bootstrap 5
- 📊 **Visualization**: Hiển thị biểu đồ phân tích độ tin cậy của từng cảm xúc

## 🏗️ Kiến trúc

### Model CNN
- **Input**: Ảnh grayscale 48x48 pixels
- **Architecture**: 4 Conv blocks với BatchNormalization và Dropout
- **Output**: 7 classes (softmax activation)
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss**: Categorical Crossentropy

### Tech Stack
- **Backend**: Flask (Python web framework)
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Visualization**: Chart.js, Matplotlib

## 📁 Cấu trúc dự án

```
face-emotion-recognition/
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── data/                       # Dataset directory
│
├── models/                     # Trained models
│   └── emotion_model.h5
│
├── src/                        # Source code
│   ├── emotion_detector.py    # Emotion detection class
│   ├── model.py               # CNN model architecture
│   ├── train.py               # Training script
│   └── realtime_detection.py  # Webcam detection
│
├── static/                     # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
└── templates/                  # HTML templates
    └── index.html
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/tangmanh891/face-emotion-recognition.git
cd face-emotion-recognition
```

### 2. Tạo môi trường ảo

```bash
conda create -n emotion-recognition python=3.9 
conda activate emotion-recognition 
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị dữ liệu

Tải dataset và đặt vào thư mục `data/` theo cấu trúc:
- `data/train/` - Dữ liệu training
- `data/test/` - Dữ liệu testing

Mỗi thư mục chứa 7 thư mục con tương ứng với 7 loại cảm xúc.

**Dataset**: 
- [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

## 🎯 Sử dụng

### Training Model

```bash
python src/train.py
```

Các tùy chọn trong `train.py`:
- `BATCH_SIZE`: Kích thước batch (mặc định: 64)
- `EPOCHS`: Số epoch (mặc định: 50)
- `IMG_SIZE`: Kích thước ảnh (mặc định: 48x48)

Model được lưu tại: `models/emotion_model.h5`

Tải model đã huấn luyện tại: https://drive.google.com/file/d/16FKwNZKq9PG3DrbBfC4oAhXrrFuqQSyF/view?usp=sharing 

### Chạy Web Application

```bash
python app.py
```

Truy cập: `http://localhost:5000`

### Nhận diện Real-time qua Webcam

```bash
python src/realtime_detection.py
```

Nhấn:
- `q`: Thoát
- `s`: Chụp ảnh và lưu kết quả
- `p`: Hiển thị biểu đồ phân tích

## 🎨 Giao diện Web

### Các chức năng chính:

1. **Upload ảnh**: 
   - Kéo thả hoặc chọn file ảnh
   - Hỗ trợ: JPG, PNG, JPEG
   
2. **Chụp ảnh từ webcam**:
   - Bật camera trực tiếp trên trình duyệt
   - Chụp và phân tích ngay lập tức

3. **Kết quả**:
   - Hiển thị ảnh với khung đánh dấu khuôn mặt
   - Tên cảm xúc và độ tin cậy (%)
   - Biểu đồ phân bố xác suất các cảm xúc

## 🙏 Acknowledgments

- Dataset: FER-2013
- OpenCV for face detection
- TensorFlow/Keras team
- Flask framework
- Bootstrap team

## 📞 Contact

Nếu có câu hỏi hoặc đóng góp ý tưởng, vui lòng tạo issue trên GitHub.

---

⭐ **Star this repo if you find it helpful!** ⭐
