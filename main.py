import os
from threading import Thread
from src.classes.DefaultCam import *
from src.classes.DepthCam import *
from src.helpers.state.ThreadSafeSingleton import *


def main():
    state = ThreadSafeSingleton()

    # Для Depth Cam (RealSense)
    depth_cam_config = {
        'distance_min': 0.8,
        'distance_max': 2.44,
        'use_live_camera': True,  # Флаг для использования живой камеры
        'camera_serial': None,  # Серийный номер камеры RealSense (если несколько)
        'record_to_bag': 'data/output/live_recording.bag'  # Опционально: запись в bag
    }

    # Для Default Cam (обычная USB камера)
    default_cam_config = {
        'min_area': 20,
        'max_area': 500,
        'min_speed': 25.0,
        'max_speed': 300.0,
        'use_live_camera': True,  # Флаг для использования живой камеры
        'camera_id': 0,  # ID камеры (0 - обычно встроенная/первая USB)
        'record_output': True  # Записывать ли выходное видео
    }

    # Пути для выходных файлов (если запись включена)
    video_output_path = 'data/output/videos/result_optimized.mp4'
    mask_output_path = 'data/output/videos/result_mask.mp4'

    # Создаем процессоры
    depth_processor = DepthCamProcessor(config=depth_cam_config)
    default_processor = DefaultCamProcessor(
        output_path=video_output_path,
        mask_output_path=mask_output_path,
        config=default_cam_config
    )

    # Инициализация
    depth_processor.initialize()
    default_processor.initialize()

    # Создаем и запускаем потоки
    thread1 = Thread(target=default_processor.run, args=([state]))
    thread2 = Thread(target=depth_processor.run, args=([state]))

    thread1.start()
    thread2.start()

    # Ожидаем завершения
    thread1.join()
    thread2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()