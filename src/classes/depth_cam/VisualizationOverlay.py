import numpy as np
import cv2
from os import path
import logging

parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationOverlay:
    """Класс для управления оверлеями визуализации"""

    def __init__(self, width: int, height: int, roi_polygon: np.ndarray):
        self.width = width
        self.height = height
        self.roi_polygon = roi_polygon
        self.roi_overlay = self._create_roi_overlay()

    def _create_roi_overlay(self) -> np.ndarray:
        """Создание оверлея для ROI"""
        overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.fillPoly(overlay, [self.roi_polygon], (0, 100, 0))
        return overlay

    def add_roi_overlay(self, image: np.ndarray) -> np.ndarray:
        """Добавление ROI оверлея на изображение"""
        result = cv2.addWeighted(image, 1.0, self.roi_overlay, 0.2, 0)
        cv2.polylines(result, [self.roi_polygon], True, (0, 255, 255), 2)
        return result

    def add_info_panel(self, image: np.ndarray, info: dict) -> np.ndarray:
        """Добавление информационной панели"""
        # Фон информационной панели
        cv2.rectangle(image, (0, 0), (400, 180), (0, 0, 0), -1)

        # Информация
        y_offset = 30
        for key, value in info.items():
            cv2.putText(image, f"{key}: {value}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        return image