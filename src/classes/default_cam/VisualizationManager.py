import cv2
import numpy as np
import time
from collections import deque
from os import path
from typing import List, Tuple, Dict, Deque, Any
import logging

from numpy import floating

from src.classes.default_cam.data.Trajectory import Trajectory
from src.classes.default_cam.data.Detection import Detection
from src.default_configs.default_cam_config import DEFAULT_CONFIG

parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationManager:
    """Менеджер визуализации"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Статистика производительности
        self.frame_times: Deque[float] = deque(maxlen=DEFAULT_CONFIG['frame_time_buffer_size'])
        self.frame_start_time = 0

    def start_frame_timer(self):
        """Запуск таймера для измерения времени обработки кадра"""
        self.frame_start_time = time.perf_counter()

    def end_frame_timer(self) -> tuple[float, float | int, float | int, floating[Any] | int]:
        """Завершение измерения времени обработки кадра"""
        frame_time = time.perf_counter() - self.frame_start_time
        self.frame_times.append(frame_time)

        avg_time = np.mean(self.frame_times) if self.frame_times else 0
        current_fps = 1 / frame_time if frame_time > 0 else 0
        avg_fps = 1 / avg_time if avg_time > 0 else 0

        return frame_time, current_fps, avg_fps, avg_time

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Отрисовка детекций"""
        debug_frame = frame.copy()

        for detection in detections:
            # Отрисовка ограничивающего прямоугольника (серый, для отладки)
            rect = cv2.minAreaRect(detection.contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(debug_frame, [box], 0, (150, 150, 150), 1)

        return debug_frame

    def draw_trajectories(self, frame: np.ndarray, trajectories: Dict[int, Trajectory],
                          colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
        """Отрисовка траекторий"""
        for traj_id, trajectory in trajectories.items():
            if not trajectory.is_active or len(trajectory.points) < 2:
                continue

            # Проверка скорости
            if not self._check_speed(trajectory):
                continue

            color = colors.get(traj_id, (0, 255, 0))

            # Отрисовка траектории
            pts = np.array(trajectory.points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, color, 3)

            # Отрисовка последнего контура
            if trajectory.contours:
                last_contour = trajectory.contours[-1]
                rect = cv2.minAreaRect(last_contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(frame, [box], 0, color, 2)

            # Отрисовка центра
            if trajectory.last_point:
                cv2.circle(frame, trajectory.last_point, 5, color, -1)

                # Отображение скорости
                avg_speed = trajectory.average_speed
                cv2.putText(frame, f"{avg_speed:.1f}",
                            (trajectory.last_point[0] + 10, trajectory.last_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def _check_speed(self, trajectory: Trajectory) -> bool:
        """Проверка скорости объекта"""
        if len(trajectory.speeds) < 1:
            return False

        avg_speed = trajectory.average_speed
        return (DEFAULT_CONFIG['min_speed'] <= avg_speed <= DEFAULT_CONFIG['max_speed'])

    def draw_info_panel(self, frame: np.ndarray, frame_number: int,
                        frame_time: float, current_fps: float, avg_fps: float,
                        timestamp: float, state) -> np.ndarray:
        """Добавление информационной панели"""
        # Информация о кадре
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Время обработки и FPS
        cv2.putText(frame, f"Time: {frame_time * 1000:.1f} ms | FPS: {current_fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Средние значения
        cv2.putText(frame, f"Avg: {np.mean(self.frame_times) * 1000:.1f} ms | Avg FPS: {avg_fps:.1f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Временные метки
        cv2.putText(frame, f"Timestamp: {timestamp:.0f} ms",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Разница временных меток
        if hasattr(state, 'get_timestamp_depth_cam'):
            time_diff = timestamp - state.get_timestamp_depth_cam()
            cv2.putText(frame, f"Frame differents: {time_diff:.0f} ms",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame