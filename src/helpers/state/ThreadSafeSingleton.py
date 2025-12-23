from threading import Lock, Event
from src.helpers.state.CameraManager import CameraManager
from typing import List, Any

class ThreadSafeSingleton:
    """Потокобезопасный синглтон с управлением камерами"""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                print("Создаю новый экземпляр синглтона")
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Приватная инициализация экземпляра"""
        print("Инициализирую синглтон...")

        # Данные с камер
        self.data_from_depth_cam: List[Any] = []

        # Менеджер камер
        self.cameras = CameraManager()

        # Общие атрибуты
        self.counter = 0
        self.pause_event = Event()
        self.pause_event.set()

        # Флаг инициализации
        self._initialized = True

    # Делегированные методы для работы с камерами
    # Глубинная камера
    def pause_depth_cam(self):
        self.cameras.pause_depth()

    def resume_depth_cam(self):
        self.cameras.resume_depth()

    def get_event_depth_cam(self) -> Event:
        return self.cameras.depth_cam.event

    def get_paused_depth_cam(self) -> bool:
        return self.cameras.depth_cam.paused

    def set_timestamp_depth_cam(self, timestamp: int):
        with self.cameras._lock:
            self.cameras.depth_cam.timestamp = timestamp

    def get_timestamp_depth_cam(self) -> int:
        with self.cameras._lock:
            return self.cameras.depth_cam.timestamp

    # Обычная камера
    def pause_default_cam(self):
        self.cameras.pause_default()

    def resume_default_cam(self):
        self.cameras.resume_default()

    def get_event_default_cam(self) -> Event:
        return self.cameras.default_cam.event

    def get_paused_default_cam(self) -> bool:
        return self.cameras.default_cam.paused

    def set_timestamp_default_cam(self, timestamp: int):
        with self.cameras._lock:
            self.cameras.default_cam.timestamp = timestamp

    def get_timestamp_default_cam(self) -> int:
        with self.cameras._lock:
            return self.cameras.default_cam.timestamp

    def set_touched_depth_cam(self, touched: bool):
        with self.cameras._lock:
            self.cameras.depth_cam.set_touched(touched)

    def get_touched_state_depth_cam(self) -> bool:
        with self.cameras._lock:
            return self.cameras.depth_cam.touched_state