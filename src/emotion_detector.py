"""
Class để nhận diện cảm xúc từ ảnh
"""
import cv2
import numpy as np
from tensorflow import keras

class EmotionDetector:
    """
    Class nhận diện cảm xúc từ khuôn mặt
    """
    
    # 7 loại cảm xúc
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Tiếng Việt
    EMOTIONS_VI = ['Tức giận', 'Ghê tởm', 'Sợ hãi', 'Vui vẻ', 'Buồn bã', 'Ngạc nhiên', 'Bình thường']
    
    def __init__(self, model_path, use_vietnamese=False):
        """
        Khởi tạo detector
        
        Args:
            model_path: Đường dẫn đến file model (.h5)
            use_vietnamese: Sử dụng tên cảm xúc tiếng Việt
        """
        self.model = keras.models.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotions = self.EMOTIONS_VI if use_vietnamese else self.EMOTIONS
        
    def detect_faces(self, image):
        """
        Phát hiện khuôn mặt trong ảnh
        
        Args:
            image: Ảnh BGR từ OpenCV
            
        Returns:
            faces: List các vùng khuôn mặt (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def preprocess_face(self, face_img):
        """
        Tiền xử lý ảnh khuôn mặt
        
        Args:
            face_img: Ảnh khuôn mặt
            
        Returns:
            processed: Ảnh đã xử lý, shape (1, 48, 48, 1)
        """
        # Resize về 48x48
        face_img = cv2.resize(face_img, (48, 48))
        
        # Chuyển sang grayscale nếu cần
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        face_img = face_img / 255.0
        
        # Reshape
        face_img = face_img.reshape(1, 48, 48, 1)
        
        return face_img
    
    def predict_emotion(self, face_img):
        """
        Dự đoán cảm xúc từ ảnh khuôn mặt
        
        Args:
            face_img: Ảnh khuôn mặt
            
        Returns:
            emotion: Tên cảm xúc
            confidence: Độ tin cậy (0-100)
            probabilities: Xác suất của tất cả các cảm xúc
        """
        # Tiền xử lý
        processed = self.preprocess_face(face_img)
        
        # Dự đoán
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Lấy kết quả
        emotion_idx = np.argmax(predictions)
        emotion = self.emotions[emotion_idx]
        confidence = predictions[emotion_idx] * 100
        
        # Tạo dict xác suất
        probabilities = {
            self.emotions[i]: predictions[i] * 100 
            for i in range(len(self.emotions))
        }
        
        return emotion, confidence, probabilities
    
    def detect_emotion_from_image(self, image):
        """
        Nhận diện cảm xúc từ ảnh có thể có nhiều khuôn mặt
        
        Args:
            image: Ảnh BGR từ OpenCV
            
        Returns:
            results: List các kết quả [(x, y, w, h, emotion, confidence), ...]
        """
        faces = self.detect_faces(image)
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face
            face_img = image[y:y+h, x:x+w]
            
            # Predict emotion
            emotion, confidence, _ = self.predict_emotion(face_img)
            
            results.append((x, y, w, h, emotion, confidence))
        
        return results
    
    def draw_results(self, image, results):
        """
        Vẽ kết quả lên ảnh
        
        Args:
            image: Ảnh gốc
            results: Kết quả từ detect_emotion_from_image
            
        Returns:
            image: Ảnh đã vẽ kết quả
        """
        image_copy = image.copy()
        
        for (x, y, w, h, emotion, confidence) in results:
            # Vẽ khung khuôn mặt
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Vẽ text
            text = f"{emotion}: {confidence:.1f}%"
            
            # Background cho text
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                image_copy,
                (x, y - text_height - 10),
                (x + text_width, y),
                (0, 255, 0),
                -1
            )
            
            # Text
            cv2.putText(
                image_copy,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return image_copy

if __name__ == '__main__':
    # Test detector
    import os
    
    model_path = 'models/emotion_model.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model tại: {model_path}")
        print("Vui lòng chạy training trước: python src/train.py")
    else:
        detector = EmotionDetector(model_path, use_vietnamese=True)
        print("✓ Detector đã sẵn sàng!")
        print(f"✓ Các cảm xúc: {', '.join(detector.emotions)}")
