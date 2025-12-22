import pyrealsense2 as rs
from os import path
import logging
from src.classes.general.data.CameraConfig import CameraConfig

parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealsensePipeline:
    """Класс для управления конвейером RealSense"""

    def __init__(self, bag_file_path: str):
        self.bag_file_path = bag_file_path
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)

    def initialize(self) -> CameraConfig:
        """Инициализация конвейера"""
        # Настройка для чтения из bag-файла
        self.config.enable_device_from_file(self.bag_file_path, repeat_playback=False)
        self.config.enable_stream(rs.stream.depth)
        self.config.enable_stream(rs.stream.color)

        # Запуск конвейера
        profile = self.pipeline.start(self.config)

        # Получение конфигурации камеры
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

        logger.info(f"Конвейер инициализирован. Разрешение: {config.width}x{config.height}, FPS: {config.fps}")

        return config

    def get_frames(self):
        """Получение кадров из конвейера"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        timestamp = frames.get_timestamp()

        return depth_frame, color_frame, timestamp

    def stop(self):
        """Остановка конвейера"""
        self.pipeline.stop()
        logger.info("Конвейер остановлен")