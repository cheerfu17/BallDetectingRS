import cv2
import logging

from src.classes.general.data.CameraConfig import CameraConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoWriterManager:
    """Менеджер для записи видео"""

    def __init__(self, output_path: str, debug_path: str, config: CameraConfig):
        self.output_path = output_path
        self.debug_path = debug_path
        self.config = config
        self.output_writer = None
        self.debug_writer = None

    def initialize(self):
        """Инициализация видеозаписывающих устройств"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.output_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.config.fps,
            (self.config.width, self.config.height)
        )

        self.debug_writer = cv2.VideoWriter(
            self.debug_path,
            fourcc,
            self.config.fps,
            (self.config.width, self.config.height)
        )

        if not self.output_writer.isOpened():
            raise RuntimeError(f"Не удалось открыть видеовывод: {self.output_path}")

        logger.info(f"Видеовыводы инициализированы: {self.output_path}, {self.debug_path}")

    def write(self, frame, debug_frame):
        """Запись кадров"""
        if len(debug_frame.shape) == 2:
            debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
        self.output_writer.write(frame)
        self.debug_writer.write(debug_frame)

    def release(self):
        """Освобождение ресурсов"""
        if self.output_writer:
            self.output_writer.release()
        if self.debug_writer:
            self.debug_writer.release()
        logger.info("Ресурсы видеозаписи освобождены")