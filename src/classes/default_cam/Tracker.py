import numpy as np
from collections import deque
from os import path
from typing import List, Tuple, Dict
import logging

from src.classes.default_cam.data.Trajectory import Trajectory
from src.classes.default_cam.data.Detection import Detection
from src.default_configs.default_cam_config import DEFAULT_CONFIG

parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tracker:
    """Трекер объектов"""

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.trajectories: Dict[int, Trajectory] = {}
        self.next_id = 0
        self.colors: Dict[int, Tuple[int, int, int]] = {}

        logger.info("Трекер инициализирован")

    def update(self, detections: List[Detection]) -> Dict[int, Trajectory]:
        """Обновление траекторий на основе новых детекций"""
        matched_ids = set()

        # Обработка каждой детекции
        for detection in detections:
            self._process_detection(detection, matched_ids)

        # Увеличение счетчика пропущенных кадров для несовпавших треков
        self._increment_missed_frames(matched_ids)

        # Удаление старых траекторий
        self._remove_old_trajectories()

        return self.trajectories

    def _process_detection(self, detection: Detection, matched_ids: set):
        """Обработка отдельной детекции"""
        best_id, min_dist = self._find_best_match(detection.center)

        if best_id != -1:
            # Обновление существующей траектории
            self._update_trajectory(best_id, detection)
            matched_ids.add(best_id)
        else:
            # Создание новой траектории
            self._create_new_trajectory(detection)
            matched_ids.add(self.next_id - 1)

    def _find_best_match(self, center: Tuple[int, int]) -> Tuple[int, float]:
        """Поиск ближайшей траектории"""
        best_id = -1
        min_dist = float('inf')

        for traj_id, trajectory in self.trajectories.items():
            if not trajectory.is_active or trajectory.last_point is None:
                continue

            dist = np.linalg.norm(np.array(center) - np.array(trajectory.last_point))

            if dist < self.config['track_distance'] and dist < min_dist:
                min_dist = dist
                best_id = traj_id

        return best_id, min_dist

    def _update_trajectory(self, traj_id: int, detection: Detection):
        """Обновление существующей траектории"""
        trajectory = self.trajectories[traj_id]

        # Вычисление скорости
        speed = 0.0
        if trajectory.last_point:
            speed = np.linalg.norm(
                np.array(detection.center) - np.array(trajectory.last_point)
            )

        # Обновление траектории
        trajectory.add_point(detection.center, detection.contour, speed)

    def _create_new_trajectory(self, detection: Detection):
        """Создание новой траектории"""
        color = tuple(np.random.randint(0, 255, 3).tolist())

        trajectory = Trajectory(
            id=self.next_id,
            points=deque([detection.center], maxlen=DEFAULT_CONFIG['trajectory_length']),
            speeds=deque(maxlen=DEFAULT_CONFIG['trajectory_length']),
            contours=deque([detection.contour], maxlen=DEFAULT_CONFIG['trajectory_length']),
            color=color
        )

        self.trajectories[self.next_id] = trajectory
        self.colors[self.next_id] = color
        self.next_id += 1

        logger.debug(f"Создана новая траектория с ID: {trajectory.id}")

    def _increment_missed_frames(self, matched_ids: set):
        """Увеличение счетчика пропущенных кадров"""
        for traj_id in list(self.trajectories.keys()):
            if traj_id not in matched_ids and self.trajectories[traj_id].is_active:
                self.trajectories[traj_id].increment_missed()

    def _remove_old_trajectories(self):
        """Удаление старых траекторий"""
        to_remove = []
        for traj_id, trajectory in self.trajectories.items():
            if trajectory.missed_frames > self.config['max_missed_frames']:
                to_remove.append(traj_id)

        for traj_id in to_remove:
            del self.trajectories[traj_id]
            logger.debug(f"Удалена траектория с ID: {traj_id}")