import cv2
import numpy as np
from os import path
from typing import Optional
import logging

from src.default_configs.default_cam_config import DEFAULT_CONFIG

parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MotionDetector:
    """Детектор движения"""

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Инициализация вычитателя фона
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config['background_history'],
            varThreshold=self.config['background_threshold'],
            detectShadows=False
        )

        # Буферы для frame differencing
        self.prev_gray = None
        self.prev_prev_gray = None

        # Структурные элементы для морфологических операций
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            self.config['dilation_kernel_size']
        )

        logger.info("Детектор движения инициализирован")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Обработка кадра для выделения движения"""
        # Преобразование в градации серого и размытие
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.config['gaussian_blur_size'], 0)

        # Вычитание фона (MOG2)
        fg_mask_mog = self.background_subtractor.apply(frame)
        _, fg_mask_mog = cv2.threshold(fg_mask_mog, 200, 255, cv2.THRESH_BINARY)

        # Frame differencing
        motion_mask = self._frame_differencing(gray)

        # Объединение масок
        if motion_mask is not None:
            fg_mask = cv2.bitwise_or(fg_mask_mog, motion_mask)
        else:
            fg_mask = fg_mask_mog

        # Удаление шума
        fg_mask = cv2.erode(fg_mask, self.kernel, iterations=1)

        # Обновление буферов
        self._update_buffers(gray)

        return fg_mask

    def _frame_differencing(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Frame differencing метод"""
        if self.prev_gray is not None and self.prev_prev_gray is not None:
            # Разница: Текущий - Прошлый
            diff1 = cv2.absdiff(gray, self.prev_gray)
            # Разница: Прошлый - Позапрошлый
            diff2 = cv2.absdiff(self.prev_gray, self.prev_prev_gray)
            # Логическое И для устранения "призраков"
            motion_mask = cv2.bitwise_and(diff1, diff2)
            # Бинаризация
            _, motion_mask = cv2.threshold(
                motion_mask,
                self.config['motion_threshold'],
                255,
                cv2.THRESH_BINARY
            )
            return motion_mask
        return None

    def _update_buffers(self, gray: np.ndarray):
        """Обновление буферов предыдущих кадров"""
        self.prev_prev_gray = self.prev_gray
        self.prev_gray = gray.copy()