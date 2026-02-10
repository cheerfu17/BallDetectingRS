from threading import Event
from dataclasses import dataclass

@dataclass
class CameraState:
    """Состояние одной камеры"""
    event: Event
    timestamp: int = 0
    paused: bool = False

    def pause(self):
        self.event.clear()
        self.paused = True

    def resume(self):
        self.event.set()
        self.paused = False

    def is_set(self) -> bool:
        return self.event.is_set()

    def wait(self, timeout: float = None) -> bool:
        return self.event.wait(timeout)