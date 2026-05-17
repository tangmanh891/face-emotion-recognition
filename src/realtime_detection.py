"""
Nhận diện cảm xúc real-time qua webcam
"""
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np

from emotion_detector import EmotionDetector


def plot_emotion_bar(probabilities, emotions):
    """
    Vẽ biểu đồ cảm xúc
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red', 'purple', 'orange', 'green', 'blue', 'yellow', 'gray']

    y_pos = np.arange(len(emotions))
    values = [probabilities.get(emotion, 0) for emotion in emotions]

    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(emotions)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Emotion Distribution')
    ax.set_xlim([0, 100])

    for i, v in enumerate(values):
        ax.text(v + 2, i, f'{v:.1f}%', va='center')

    plt.tight_layout()
    return fig

def realtime_detection(model_path='models/emotion_model.keras', use_vietnamese=True):
    """Realtime emotion detection from webcam."""
    print("Đang khởi tạo...")

    # Load detector
    detector = EmotionDetector(model_path, use_vietnamese=use_vietnamese)
    print("✓ Đã load model")

    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return

    print("✓ Đã mở webcam")
    print("\nHướng dẫn:")
    print("  - Nhấn 'q' để thoát")
    print("  - Nhấn 's' để chụp ảnh và lưu")
    print("  - Nhấn 'p' để xem biểu đồ phân bố cảm xúc")
    print("\n" + "="*50)

    # Để lưu lịch sử cảm xúc
    emotion_history = deque(maxlen=30)  # 30 frames
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không đọc được frame!")
            break

        frame_count += 1

        # Flip frame (mirror)
        frame = cv2.flip(frame, 1)

        # Detect emotions
        results = detector.detect_emotion_from_image(frame)

        # Draw results
        output = detector.draw_results(frame, results)

        # Lưu lịch sử cảm xúc (chỉ lấy cảm xúc đầu tiên nếu có)
        if results:
            emotion_history.append(results[0][4])  # Cảm xúc

        # Hiển thị số khuôn mặt
        cv2.putText(
            output,
            f"Faces: {len(results)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Hiển thị FPS (mỗi 10 frames)
        if frame_count % 10 == 0:
            cv2.putText(
                output,
                f"Frame: {frame_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Hiển thị
        cv2.imshow('Emotion Detection - Press Q to quit', output)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nThoát...")
            break
        elif key == ord('s'):
            # Lưu ảnh
            filename = f'captured_{frame_count}.jpg'
            cv2.imwrite(filename, output)
            print(f"✓ Đã lưu: {filename}")
        elif key == ord('p') and results:
            # Hiển thị biểu đồ
            _, _, probabilities = detector.predict_emotion(
                frame[results[0][1]:results[0][1]+results[0][3],
                      results[0][0]:results[0][0]+results[0][2]]
            )
            plot_emotion_bar(probabilities, detector.emotions)
            plt.show()

    # Thống kê
    if emotion_history:
        print("\n" + "="*50)
        print("THỐNG KÊ CẢM XÚC:")
        from collections import Counter
        emotion_counts = Counter(emotion_history)
        for emotion, count in emotion_counts.most_common():
            percentage = (count / len(emotion_history)) * 100
            print(f"  {emotion}: {count} lần ({percentage:.1f}%)")
        print("="*50)

    # Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Đã đóng webcam")

if __name__ == '__main__':
    import os

    model_path = 'models/emotion_model.keras'
    if not os.path.exists(model_path):
        model_path = 'models/emotion_model.h5'

    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model tại: {model_path}")
        print("Vui lòng chạy training trước: python src/train.py")
    else:
        realtime_detection(model_path, use_vietnamese=True)
