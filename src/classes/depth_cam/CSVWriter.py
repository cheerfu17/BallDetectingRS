import csv
from os import path
import logging
from src.classes.depth_cam.data.Detection import Detection
parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVWriter:
    """Класс для записи данных в CSV"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.writer = None

    def initialize(self):
        """Инициализация CSV файла"""
        self.file = open(self.filepath, 'w', newline='', encoding='utf-8')
        fieldnames = ['frame', 'timestamp', 'x', 'y', 'width', 'height',
                      'distance', 'datetime']
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        logger.info(f"CSV файл инициализирован: {self.filepath}")

    def write_detection(self, detection: Detection):
        """Запись детекции в CSV"""
        self.writer.writerow(detection.to_dict())

    def close(self):
        """Закрытие CSV файла"""
        if self.file:
            self.file.close()
            logger.info("CSV файл закрыт")
