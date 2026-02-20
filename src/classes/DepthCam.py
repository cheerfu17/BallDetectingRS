import numpy as np
import cv2
from os import path
import logging
import time
import pyrealsense2 as rs
from src.classes.depth_cam.RealsensePipeline import RealsensePipeline
from src.classes.depth_cam.LiveRealsenseCapture import LiveRealsenseCapture
from src.classes.general.VideoWriterManager import VideoWriterManager
from src.classes.depth_cam.CSVWriter import CSVWriter
from src.classes.depth_cam.DetectionProcessor import DetectionProcessor
from src.classes.depth_cam.VisualizationOverlay import VisualizationOverlay
from src.default_configs.depth_cam_config import DEFAULT_CONFIG

parent_dir = path.dirname(path.abspath(__file__))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepthCamProcessor:
    """Основной класс для обработки с камеры глубины RealSense"""

    def __init__(self, bag_file_path: str = None, output_video_name: str = None,
                 output_csv_name: str = None, config: dict = None):
        self.bag_file_path = bag_file_path
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        if output_video_name:
            self.config['output_video'] = output_video_name
        if output_csv_name:
            self.config['csv_file'] = output_csv_name

        self.roi_polygon = np.array([
            [93, 298], [306, 270], [575, 270], [785, 293]
        ], dtype=np.int32)

        # Выбор источника: файл или живая камера
        self.use_live_camera = self.config.get('use_live_camera', True)

        if self.use_live_camera:
            self.pipeline = LiveRealsenseCapture(
                camera_serial=self.config.get('camera_serial'),
                record_to_bag=self.config.get('record_to_bag')
            )
        else:
            if not bag_file_path:
                raise ValueError("bag_file_path должен быть указан при использовании файла")
            self.pipeline = RealsensePipeline(bag_file_path)

        self.detection_processor = DetectionProcessor(
            self.config['distance_min'],
            self.config['distance_max'],
            self.config['min_contour_area'],
            self.config['min_valid_depth_points']
        )

        self.video_writer = None
        self.csv_writer = None
        self.visualization = None
        self.camera_config = None
        self.frame_count = 0
        self.total_detections = 0
        self._is_initialized = False

    def initialize(self):
        """Инициализация всех компонентов"""
        if self._is_initialized:
            logger.warning("Попытка повторной инициализации DepthCam. Пропуск.")
            return

        source_type = "живая камера" if self.use_live_camera else f"bag файл {self.bag_file_path}"
        logger.info(f"Начинаю обработку с {source_type}...")

        self.camera_config = self.pipeline.initialize()

        self.video_writer = VideoWriterManager(
            self.config['output_video'],
            self.config['debug_video'],
            self.camera_config
        )
        self.video_writer.initialize()

        self.csv_writer = CSVWriter(self.config['csv_file'])
        self.csv_writer.initialize()

        self.visualization = VisualizationOverlay(
            self.camera_config.width,
            self.camera_config.height,
            self.roi_polygon
        )
        self._is_initialized = True
        logger.info("Инициализация DepthCam завершена")

    def process_frame(self, state) -> bool:
        """Обработка одного кадра"""
        try:
            depth_frame, color_frame, timestamp = self.pipeline.get_frames()
            if not depth_frame or not color_frame:
                logger.warning("Пропускаю кадр: отсутствуют данные глубины или цвета")
                return True

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_meters = depth_image.astype(float) * self.camera_config.depth_scale

            processed_frame, detections, debug_frame = self.detection_processor.process(
                color_image, depth_meters, self.roi_polygon,
                self.frame_count, timestamp
            )

            # Проверка попадания в полигон
            is_hit_in_polygon = len(detections) > 0

            processed_frame = self.visualization.add_roi_overlay(processed_frame)

            info = {
                "Frame": self.frame_count,
                "Time": f"{timestamp:.0f} ms",
                "Detections": len(detections),
                "Hit": "YES" if is_hit_in_polygon else "NO"
            }
            processed_frame = self.visualization.add_info_panel(processed_frame, info)

            for detection in detections:
                self.csv_writer.write_detection(detection)
            self.total_detections += 1

            self.video_writer.write(processed_frame, debug_frame)

            # Отображение кадров
            self._display_frames(processed_frame, debug_frame)

            if self.frame_count % 30 == 0 and self.frame_count > 0:
                logger.info(f"DepthCam Frame {self.frame_count} | Hit: {is_hit_in_polygon}")

            self.frame_count += 1

            # state.sync_depth_cam(timestamp, is_in_polygon=is_hit_in_polygon)

            return True

        except RuntimeError as e:
            if "frame didn't arrive" in str(e):
                logger.info("Обработка завершена (конец потока)")
                return False
            else:
                logger.error(f"Ошибка при обработке кадра: {e}")
                raise

    def _display_frames(self, processed_frame, debug_frame):
        """Отображение кадров"""
        cv2.imshow('Depth Processed', processed_frame)
        cv2.imshow('Depth Debug', debug_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt()

    def run(self, state):
        """Основной цикл обработки"""
        try:
            if not self._is_initialized:
                self.initialize()

            logger.info("Запуск обработки DepthCam (живой режим)...")
            while not state.should_stop():
                if not self.process_frame(state):
                    logger.info("DepthCam: Поток закончился.")
                    state.request_stop()
                    break
        except KeyboardInterrupt:
            logger.info("Обработка прервана пользователем")
        except Exception as e:
            logger.error(f"Критическая ошибка DepthCam: {e}")
            state.request_stop()
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Очистка ресурсов"""
        logger.info("Очистка ресурсов DepthCam...")
        if self.pipeline:
            self.pipeline.stop()
        if self.video_writer:
            self.video_writer.release()
        if self.csv_writer:
            self.csv_writer.close()
        cv2.destroyAllWindows()
        self._print_statistics()

    def _print_statistics(self):
        print("\n" + "=" * 50)
        print("DEPTH CAM ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"Всего кадров: {self.frame_count}")
        print(f"Всего обнаружено объектов: {self.total_detections}")