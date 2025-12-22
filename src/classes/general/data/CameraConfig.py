from dataclasses import dataclass

@dataclass
class CameraConfig:
    """Конфигурация камеры"""
    width: int
    height: int
    fps: int
    depth_scale: float