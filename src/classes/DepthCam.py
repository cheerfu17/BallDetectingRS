import numpy as np
import cv2
from os import path
import logging

from src.classes.depth_cam.RealsensePipeline import RealsensePipeline
from src.classes.general.VideoWriterManager import VideoWriterManager
from src.classes.depth_cam.CSVWriter import CSVWriter
from src.classes.depth_cam.DetectionProcessor import DetectionProcessor
from src.classes.depth_cam.VisualizationOverlay import VisualizationOverlay
from src.default_configs.depth_cam_config import DEFAULT_CONFIG

parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BagFileProcessor:
    """Основной класс для обработки bag-файлов"""

    def __init__(self, bag_file_path: str, output_video_name: str = None,
                 output_csv_name: str = None, config: dict = None):
        self.bag_file_path = bag_file_path
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Обновление путей если предоставлены пользовательские имена
        if output_video_name:
            self.config['output_video'] = output_video_name
        if output_csv_name:
            self.config['csv_file'] = output_csv_name

        # ROI полигон (можно вынести в конфиг)
        self.roi_polygon = np.array([
            [93, 298],
            [306, 270],
            [575, 270],
            [785, 293]
        ], dtype=np.int32)

        # Инициализация компонентов
        self.pipeline = RealsensePipeline(bag_file_path)
        self.detection_processor = DetectionProcessor(
            self.config['distance_min'],
            self.config['distance_max'],
            self.config['min_contour_area'],
            self.config['min_valid_depth_points']
        )

        # Будут инициализированы позже
        self.video_writer = None
        self.csv_writer = None
        self.visualization = None
        self.camera_config = None

        self.frame_count = 0
        self.total_detections = 0

    def initialize(self):
        """Инициализация всех компонентов"""
        logger.info(f"Начинаю обработку {self.bag_file_path}...")

        # Инициализация конвейера
        self.camera_config = self.pipeline.initialize()

        # Инициализация видеозаписи
        self.video_writer = VideoWriterManager(
            self.config['output_video'],
            self.config['debug_video'],
            self.camera_config
        )
        self.video_writer.initialize()

        # Инициализация CSV записи
        self.csv_writer = CSVWriter(self.config['csv_file'])
        self.csv_writer.initialize()

        # Инициализация визуализации
        self.visualization = VisualizationOverlay(
            self.camera_config.width,
            self.camera_config.height,
            self.roi_polygon
        )

        logger.info("Инициализация завершена")

    def process_frame(self, state) -> bool:
        """Обработка одного кадра"""
        try:
            # Получение кадров
            depth_frame, color_frame, timestamp = self.pipeline.get_frames()

            if not depth_frame or not color_frame:
                logger.warning("Пропускаю кадр: отсутствуют данные глубины или цвета")
                return True

            # Конвертация кадров
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_meters = depth_image.astype(float) * self.camera_config.depth_scale

            # Обработка детекций
            processed_frame, detections, debug_frame = self.detection_processor.process(
                color_image, depth_meters, self.roi_polygon,
                self.frame_count, timestamp
            )

            # Визуализация
            processed_frame = self.visualization.add_roi_overlay(processed_frame)

            # Обновление состояния
            self._update_state(state, timestamp)

            # Добавление информационной панели
            info = {
                "Frame": self.frame_count,
                "Time": f"{timestamp:.0f} ms",
                "Detections": len(detections),
                f"Range ({self.config['distance_min']}-{self.config['distance_max']}m)": "",
                "Frame diff": f"{timestamp - state.get_timestamp_default_cam():.0f} ms",
                "Default cam state": state.get_paused_default_cam()
            }
            processed_frame = self.visualization.add_info_panel(processed_frame, info)

            # Запись детекций
            for detection in detections:
                self.csv_writer.write_detection(detection)
                self.total_detections += 1

            # Запись видео
            self.video_writer.write(processed_frame, debug_frame)

            # Отображение (для отладки)
            self._display_frames(processed_frame, debug_frame)

            # Логирование прогресса
            if self.frame_count % 30 == 0 and self.frame_count > 0:
                logger.info(f"Кадр {self.frame_count} | Обнаружено: {len(detections)} объектов")

            self.frame_count += 1
            return True

        except RuntimeError as e:
            if "frame didn't arrive" in str(e):
                logger.info("Обработка завершена (конец файла)")
                return False
            else:
                logger.error(f"Ошибка при обработке кадра: {e}")
                raise

    def _update_state(self, state, timestamp):
        """Обновление состояния синхронизации"""
        state.set_timestamp_depth_cam(timestamp)

        if (timestamp - state.get_timestamp_default_cam() < 0):
            state.pause_default_cam()
        else:
            state.resume_default_cam()

    def _display_frames(self, processed_frame, debug_frame):
        """Отображение кадров для отладки"""
        cv2.imshow('Processed', processed_frame)
        cv2.imshow('Debug Mask', debug_frame)

        # Проверка нажатия клавиши для выхода
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt()

    def run(self, state):
        """Основной цикл обработки"""
        try:
            logger.info("Запуск обработки...")
            while True:
                # state.get_event_depth_cam().wait()
                if not self.process_frame(state):
                    break


        except KeyboardInterrupt:
            logger.info("Обработка прервана пользователем")
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Очистка ресурсов"""
        logger.info("Очистка ресурсов...")

        if self.pipeline:
            self.pipeline.stop()

        if self.video_writer:
            self.video_writer.release()

        if self.csv_writer:
            self.csv_writer.close()

        # cv2.destroyAllWindows()

        # Вывод статистики
        self._print_statistics()

    def _print_statistics(self):
        """Вывод статистики обработки"""
        print("\n" + "=" * 50)
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"Всего кадров: {self.frame_count}")
        print(f"Всего обнаружено объектов: {self.total_detections}")
        print(f"Результаты сохранены:")
        print(f"  Видео: {self.config['output_video']}")
        print(f"  Отладочное видео: {self.config['debug_video']}")
        print(f"  Данные: {self.config['csv_file']}")