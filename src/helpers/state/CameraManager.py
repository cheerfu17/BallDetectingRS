from src.helpers.state.CameraState import CameraState
from threading import Lock, Event

class CameraManager:
    """Менеджер для управления состоянием камер"""

    def __init__(self):
        self.depth_cam = CameraState(Event())
        self.default_cam = CameraState(Event())
        self._lock = Lock()

        # Инициализируем события как установленные
        self.depth_cam.resume()
        self.default_cam.resume()

    def pause_depth(self):
        with self._lock:
            self.depth_cam.pause()

    def resume_depth(self):
        with self._lock:
            self.depth_cam.resume()

    def pause_default(self):
        with self._lock:
            self.default_cam.pause()

    def resume_default(self):
        with self._lock:
            self.default_cam.resume()