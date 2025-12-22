from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Detection:
    """Класс для хранения информации о детекции"""
    center: Tuple[int, int]
    contour: np.ndarray
    area: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    aspect_ratio: float