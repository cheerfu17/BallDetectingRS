import cv2
import numpy as np
from os import path
from typing import Tuple, Optional
import logging

parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Базовый класс для обработки видео"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть видео файл: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = 0

        logger.info(f"Видео загружено: {video_path}")
        logger.info(f"Разрешение: {self.width}x{self.height}, FPS: {self.fps}")

    def read_frame(self) -> Optional[Tuple[bool, np.ndarray]]:
        """Чтение кадра из видео"""
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame if ret else None

    def release(self):
        """Освобождение ресурсов видео"""
        if self.cap:
            self.cap.release()
            logger.info("Видеоресурсы освобождены")