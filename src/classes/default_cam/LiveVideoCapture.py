import cv2
import numpy as np
from os import path
import logging
import time
from src.classes.general.data.CameraConfig import CameraConfig

parent_dir = path.dirname(path.abspath(__file__))
logger = logging.getLogger(__name__)


class LiveVideoCapture:
    """Класс для захвата видео с обычной USB камеры в реальном времени"""

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame_count = 0
        self.start_time = time.time()
        self._is_opened = False

    def initialize(self) -> CameraConfig:
        """Инициализация камеры"""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть камеру с ID: {self.camera_id}")

        # Устанавливаем параметры захвата
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Получаем реальные параметры
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        config = CameraConfig(
            width=actual_width,
            height=actual_height,
            fps=actual_fps if actual_fps > 0 else self.fps,
            depth_scale=0.0
        )

        self.start_time = time.time()
        self._is_opened = True

        logger.info(f"Камера {self.camera_id} инициализирована. "
                    f"Разрешение: {config.width}x{config.height}, FPS: {config.fps}")

        return config

    def read_frame(self):
        """Чтение кадра с камеры"""
        if not self._is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1

        return ret, frame if ret else None

    def get_timestamp(self):
        """Получение временной метки текущего кадра"""
        return int((time.time() - self.start_time) * 1000)  # в миллисекундах

    def release(self):
        """Освобождение ресурсов камеры"""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False
            logger.info(f"Камера {self.camera_id} освобождена")