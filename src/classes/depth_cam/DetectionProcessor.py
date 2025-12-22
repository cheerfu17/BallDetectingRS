import numpy as np
import cv2
from os import path
from datetime import datetime
from typing import List, Optional, Tuple
import logging
from src.classes.depth_cam.data.Detection import Detection
parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionProcessor:
    """Обработчик детекций"""

    def __init__(self, distance_min: float, distance_max: float,
                 min_contour_area: float = 0.0, min_valid_depth_points: int = 10):
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.min_contour_area = min_contour_area
        self.min_valid_depth_points = min_valid_depth_points

    def process(self, color_image: np.ndarray, depth_meters: np.ndarray,
                roi_polygon: np.ndarray, frame_number: int, timestamp: float) -> Tuple[
        np.ndarray, List[Detection], np.ndarray]:
        """
        Обработка кадра для обнаружения объектов

        Args:
            color_image: Цветное изображение
            depth_meters: Изображение глубины в метрах
            roi_polygon: Полигон области интереса
            frame_number: Номер кадра
            timestamp: Временная метка

        Returns:
            Tuple[обработанное изображение, список детекций, отладочное изображение]
        """
        display_image = color_image.copy()

        # Создание маски ROI
        roi_mask = np.zeros_like(depth_meters, dtype=np.uint8)
        cv2.fillPoly(roi_mask, [roi_polygon], 1)

        # Применение маски ROI
        roi_depth = depth_meters * roi_mask

        # Создание бинарной маски для диапазона расстояний
        distance_mask = self._create_distance_mask(roi_depth)

        # Поиск контуров
        contours, _ = cv2.findContours(distance_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        debug_mask = (distance_mask * 255).astype(np.uint8)
        debug_display = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            detection = self._process_contour(contour, roi_depth, display_image,
                                              debug_display, frame_number, timestamp)
            if detection:
                detections.append(detection)

        return display_image, detections, debug_display

    def _create_distance_mask(self, roi_depth: np.ndarray) -> np.ndarray:
        """Создание маски для заданного диапазона расстояний"""
        distance_mask = np.zeros_like(roi_depth, dtype=np.uint8)
        distance_mask[(roi_depth > self.distance_min) &
                      (roi_depth < self.distance_max) &
                      (roi_depth > 0)] = 255
        return distance_mask

    def _process_contour(self, contour: np.ndarray, roi_depth: np.ndarray,
                         display_image: np.ndarray, debug_display: np.ndarray,
                         frame_number: int, timestamp: float) -> Optional[Detection]:
        """Обработка отдельного контура"""
        # # Фильтрация по площади
        # area = cv2.contourArea(contour)
        # print(area)
        # if area < self.min_contour_area:
        #     return None

        # Получение ограничивающего прямоугольника
        x, y, w, h = cv2.boundingRect(contour)

        # Создание маски для контура
        contour_mask = np.zeros_like(roi_depth, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        # Вычисление средней глубины
        contour_pixels = roi_depth[contour_mask == 255]
        valid_depths = contour_pixels[contour_pixels > 0]

        if len(valid_depths) < self.min_valid_depth_points:
            return None

        avg_depth = np.mean(valid_depths)

        if not (self.distance_min < avg_depth < self.distance_max):
            return None

        # Визуализация
        self._visualize_detection(display_image, debug_display,
                                  contour, x, y, w, h, avg_depth)

        # Создание объекта детекции
        return Detection(
            frame_number=frame_number,
            timestamp=timestamp,
            x=x, y=y, width=w, height=h,
            distance=avg_depth,
            datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        )

    def _visualize_detection(self, display_image: np.ndarray, debug_display: np.ndarray,
                             contour: np.ndarray, x: int, y: int, w: int, h: int,
                             avg_depth: float):
        """Визуализация детекции на изображениях"""
        # Рисование прямоугольника
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Отображение расстояния
        cv2.putText(display_image, f"{avg_depth:.2f}m",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # Рисование контура на отладочном изображении
        cv2.drawContours(debug_display, [contour], -1, (0, 0, 255), 2)