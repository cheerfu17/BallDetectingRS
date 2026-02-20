import cv2

# Проверить первые 10 индексов (обычно 0 и 1 - ваши камеры)
for i in range(1000):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Камера с ID {i} доступна")
        cap.release()