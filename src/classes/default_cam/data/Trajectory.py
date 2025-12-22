from dataclasses import dataclass
from typing import Tuple, Optional, Deque
import numpy as np
from src.default_configs.default_cam_config import DEFAULT_CONFIG


@dataclass
class Trajectory:
    """Класс для хранения траектории объекта"""
    id: int
    points: Deque[Tuple[int, int]]
    speeds: Deque[float]
    contours: Deque[np.ndarray]
    missed_frames: int = 0
    color: Optional[Tuple[int, int, int]] = None

    @property
    def last_point(self) -> Optional[Tuple[int, int]]:
        return self.points[-1] if self.points else None

    @property
    def average_speed(self) -> float:
        return np.mean(self.speeds) if self.speeds else 0.0

    @property
    def is_active(self) -> bool:
        return self.missed_frames <= 0

    def add_point(self, center: Tuple[int, int], contour: np.ndarray, speed: float):
        """Добавление новой точки в траекторию"""
        self.points.append(center)
        self.speeds.append(speed)
        self.contours.append(contour)

        # Поддерживаем максимальную длину
        if len(self.points) > DEFAULT_CONFIG['trajectory_length']:
            self.points.popleft()
            self.speeds.popleft()
            self.contours.popleft()

        self.missed_frames = 0

    def increment_missed(self):
        """Увеличение счетчика пропущенных кадров"""
        self.missed_frames += 1