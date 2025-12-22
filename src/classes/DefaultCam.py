import cv2
import numpy as np
from os import path
import logging


from src.classes.general.VideoWriterManager import VideoWriterManager
from src.classes.default_cam.VisualizationManager import VisualizationManager
from src.classes.general.data.CameraConfig import CameraConfig
from src.classes.default_cam.DetectionFilter import DetectionFilter
from src.classes.default_cam.MotionDetector import MotionDetector
from src.classes.default_cam.TimestampReader import TimestampReader
from src.classes.default_cam.Tracker import Tracker
from src.classes.default_cam.VideoProcessor import VideoProcessor
from src.default_configs.default_cam_config import DEFAULT_CONFIG

parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefaultCamProcessor:
    """Основной класс для обработки видео с обычной камеры"""

    def __init__(self, video_path: str, output_path: str, mask_output_path: str,
                 config: dict = None):
        self.video_path = video_path
        self.output_path = output_path
        self.mask_output_path = mask_output_path
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Инициализация компонентов
        self.video_processor = VideoProcessor(video_path)
        config = CameraConfig(
            width=self.video_processor.width,
            height=self.video_processor.height,
            fps=self.video_processor.fps,
            depth_scale=0.0
        )

        self.video_writer = VideoWriterManager(
            output_path, mask_output_path,
            config
        )
        self.timestamp_reader = TimestampReader(self.config['csv_file'])
        self.motion_detector = MotionDetector(self.config)
        self.detection_filter = DetectionFilter(self.config)
        self.tracker = Tracker(self.config)
        self.visualization = VisualizationManager(
            self.video_processor.width, self.video_processor.height
        )

        self.paused = False
        logger.info(f"DefaultCamProcessor инициализирован для видео: {video_path}")

    def initialize(self):
        """Инициализация всех компонентов"""
        self.video_writer.initialize()
        logger.info("Все компоненты инициализированы")

    def process_frame(self, state) -> bool:
        """Обработка одного кадра"""
        # Чтение кадра
        ret, frame = self.video_processor.read_frame()
        if not ret:
            return False

        # Получение временной метки
        timestamp = self.timestamp_reader.get_timestamp(self.video_processor.frame_count - 1)

        # Обновление состояния
        self._update_state(state, timestamp)

        # Запуск таймера для измерения производительности
        self.visualization.start_frame_timer()

        # Детекция движения
        motion_mask = self.motion_detector.process_frame(frame)

        # Поиск контуров
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Фильтрация детекций
        detections = self.detection_filter.filter_contours(contours)

        # Трекинг
        trajectories = self.tracker.update(detections)

        # Визуализация
        debug_frame = self.visualization.draw_detections(frame, detections)
        debug_frame = self.visualization.draw_trajectories(
            debug_frame, trajectories, self.tracker.colors
        )

        # Измерение производительности
        frame_time, current_fps, avg_fps, avg_time = self.visualization.end_frame_timer()

        # Добавление информационной панели
        debug_frame = self.visualization.draw_info_panel(
            debug_frame, self.video_processor.frame_count, frame_time,
            current_fps, avg_fps, timestamp, state
        )

        # Запись результатов
        self.video_writer.write(debug_frame, motion_mask)

        # Отображение
        cv2.imshow('Tracking', debug_frame)
        cv2.imshow('Mask', motion_mask)

        return True

    def _update_state(self, state, timestamp: float):
        """Обновление состояния синхронизации"""
        state.set_timestamp_default_cam(timestamp)

        # if hasattr(state, 'get_timestamp_depth_cam'):
        #     if (timestamp - state.get_timestamp_depth_cam() < 0):
        #         if hasattr(state, 'pause_depth_cam'):
        #             state.pause_depth_cam()
        #     else:
        #         if hasattr(state, 'resume_depth_cam'):
        #             state.resume_depth_cam()

    def handle_keyboard(self) -> bool:
        """Обработка клавиатуры"""
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            return False
        elif key == ord('p') or key == ord('P'):
            self.paused = not self.paused
            logger.info(f"Пауза: {'включена' if self.paused else 'выключена'}")

        return True

    def run(self, state):
        """Основной цикл обработки"""
        try:
            self.initialize()
            logger.info("Запуск обработки видео...")

            while True:
                state.get_event_default_cam().wait()

                if not self.paused:
                    if not self.process_frame(state):
                        break

                if not self.handle_keyboard():
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

        self.video_processor.release()
        self.video_writer.release()
        # cv2.destroyAllWindows()

        # Вывод статистики
        self._print_statistics()

    def _print_statistics(self):
        """Вывод статистики обработки"""
        if self.visualization.frame_times:
            avg_frame_time = np.mean(self.visualization.frame_times) * 1000
            avg_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0

            print(f"\n{'=' * 50}")
            print("ОБРАБОТКА ЗАВЕРШЕНА")
            print(f"Обработано кадров: {self.video_processor.frame_count}")
            print(f"Среднее время на кадр: {avg_frame_time:.2f} ms")
            print(f"Средний FPS обработки: {avg_fps:.1f}")
            print(f"Создано траекторий: {self.tracker.next_id}")
        else:
            print("\nОбработка завершена (нет данных о производительности)")