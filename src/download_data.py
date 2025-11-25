"""
Script để tải dataset FER-2013 từ Kaggle
"""
import os
import zipfile
import kaggle

def download_fer2013():
    """
    Tải dataset FER-2013 từ Kaggle
    Yêu cầu: kaggle.json trong ~/.kaggle/
    """
    print("Đang tải dataset FER-2013 từ Kaggle...")
    
    # Tạo thư mục data nếu chưa có
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Tải dataset
    kaggle.api.dataset_download_files(
        'msambare/fer2013',
        path='data/',
        unzip=True
    )
    
    print("✓ Tải dataset thành công!")
    print(f"Dataset được lưu tại: {os.path.abspath('data/')}")
    
    # Kiểm tra cấu trúc
    if os.path.exists('data/train'):
        print("\nCấu trúc dataset:")
        for emotion in os.listdir('data/train'):
            emotion_path = os.path.join('data/train', emotion)
            if os.path.isdir(emotion_path):
                count = len(os.listdir(emotion_path))
                print(f"  - {emotion}: {count} ảnh")

if __name__ == '__main__':
    try:
        download_fer2013()
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        print("\nHướng dẫn:")
        print("1. Tạo tài khoản Kaggle tại https://www.kaggle.com/")
        print("2. Vào Account Settings > API > Create New API Token")
        print("3. Tải file kaggle.json")
        print("4. Đặt file vào:")
        print("   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("   - Linux/Mac: ~/.kaggle/kaggle.json")
        print("\nHoặc tải dataset thủ công từ:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
