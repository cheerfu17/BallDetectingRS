import cv2
import numpy as np
from os import path
from typing import List, Tuple, Optional
import logging

from src.classes.default_cam.data.Detection import Detection
from src.default_configs.default_cam_config import DEFAULT_CONFIG

parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionFilter:
    """Фильтр детекций"""

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def filter_contours(self, contours: List[np.ndarray]) -> List[Detection]:
        """Фильтрация контуров по различным критериям"""
        filtered_detections = []

        for contour in contours:
            detection = self._process_contour(contour)
            if detection:
                filtered_detections.append(detection)

        return filtered_detections

    def _process_contour(self, contour: np.ndarray) -> Optional[Detection]:
        """Обработка отдельного контура"""
        # Фильтр по площади
        area = cv2.contourArea(contour)
        if not (self.config['min_area'] <= area <= self.config['max_area']):
            return None

        # Получение ограничивающего прямоугольника
        x, y, w, h = cv2.boundingRect(contour)

        # Фильтр по форме (соотношение сторон)
        aspect_ratio = self._calculate_aspect_ratio(w, h)
        if aspect_ratio > self.config['max_aspect_ratio']:
            return None

        # Вычисление центра масс
        center = self._calculate_center(contour)
        if center is None:
            return None

        return Detection(
            center=center,
            contour=contour,
            area=area,
            bounding_box=(x, y, w, h),
            aspect_ratio=aspect_ratio
        )

    def _calculate_aspect_ratio(self, width: int, height: int) -> float:
        """Вычисление соотношения сторон"""
        if min(width, height) == 0:
            return float('inf')
        return max(width, height) / min(width, height)

    def _calculate_center(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """Вычисление центра масс контура"""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
