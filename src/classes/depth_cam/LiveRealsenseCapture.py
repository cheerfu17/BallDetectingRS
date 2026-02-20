import pyrealsense2 as rs
import numpy as np
from os import path
import logging
from src.classes.general.data.CameraConfig import CameraConfig

parent_dir = path.dirname(path.abspath(__file__))
logger = logging.getLogger(__name__)


class LiveRealsenseCapture:
    """Класс для захвата видео с живой камеры RealSense"""

    def __init__(self, camera_serial: str = None, record_to_bag: str = None):
        self.camera_serial = camera_serial
        self.record_to_bag = record_to_bag
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.recording = None
        self._is_running = False

    def initialize(self) -> CameraConfig:
        """Инициализация конвейера для живой камеры"""

        # Настройка потоков
        if self.camera_serial:
            self.config.enable_device(self.camera_serial)

        # Настройка разрешения и FPS (можно сделать настраиваемым)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Опционально: запись в bag файл
        if self.record_to_bag:
            self.config.enable_record_to_file(self.record_to_bag)
            logger.info(f"Запись потока в файл: {self.record_to_bag}")

        # Запуск конвейера
        profile = self.pipeline.start(self.config)

        # Получение параметров камеры
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

        # Масштаб глубины
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        config = CameraConfig(
            width=color_profile.get_intrinsics().width,
            height=color_profile.get_intrinsics().height,
            fps=color_profile.fps(),
            depth_scale=depth_scale
        )

        self._is_running = True
        logger.info(f"Живая камера RealSense инициализирована. "
                    f"Разрешение: {config.width}x{config.height}, FPS: {config.fps}")

        return config

    def get_frames(self):
        """Получение кадров из живой камеры"""
        if not self._is_running:
            raise RuntimeError("Камера не инициализирована")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        timestamp = frames.get_timestamp()

        return depth_frame, color_frame, timestamp

    def stop(self):
        """Остановка конвейера"""
        if self._is_running:
            self.pipeline.stop()
            self._is_running = False
            logger.info("Конвейер живой камеры остановлен")