import os
from threading import Thread
from src.classes.DefaultCam import *
from src.classes.DepthCam import *
from src.helpers.state.ThreadSafeSingleton import *
global test
def main():
    state = ThreadSafeSingleton()
    # Проверяем наличие файла

    bag_file_path = "data/input/bag/test.bag"
    if not os.path.exists(bag_file_path):
        logger.error(f"Файл {bag_file_path} не найден!")
        return

    # Создаем процессор с кастомной конфигурацией
    config = {
        'distance_min': 0.8,
        'distance_max': 2.44
    }

    processor = BagFileProcessor(bag_file_path, config=config)
    processor.initialize()

    video_path = 'data/input/videos/default_cam.mp4'
    output_path = 'data/output/videos/result_optimized.mp4'
    mask_output_path = 'data/output/videos/result_mask.mp4'

    # Кастомная конфигурация (опционально)
    config_default_cam_process = {
        'min_area': 20,
        'max_area': 500,
        'min_speed': 25.0,
        'max_speed': 300.0
    }

    default_cam_process = DefaultCamProcessor(
        video_path,
        output_path,
        mask_output_path,
        config=config_default_cam_process
    )
    default_cam_process.initialize()

    thread2 = Thread(target=processor.run, args=([state]))
    thread1 = Thread(target=default_cam_process.run, args=([state]))


    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()