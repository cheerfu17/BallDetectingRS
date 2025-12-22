from dataclasses import dataclass

@dataclass
class Detection:
    """Класс для хранения информации о детекции"""
    frame_number: int
    timestamp: float
    x: int
    y: int
    width: int
    height: int
    distance: float
    datetime: str

    def to_dict(self):
        return {
            'frame': self.frame_number,
            'timestamp': self.timestamp,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'distance': self.distance,
            'datetime': self.datetime
        }