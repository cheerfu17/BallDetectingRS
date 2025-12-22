import csv
import os
from os import path
import logging

parent_dir = path.dirname(path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimestampReader:
    """Чтение временных меток из CSV"""

    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.timestamps = []
        self._load_timestamps()

    def _load_timestamps(self):
        """Загрузка временных меток из CSV"""
        if not os.path.exists(self.csv_file):
            logger.warning(f"CSV файл не найден: {self.csv_file}")
            return

        try:
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if 'rs_hw_time' in row:
                        self.timestamps.append(float(row['rs_hw_time']))

            logger.info(f"Загружено {len(self.timestamps)} временных меток")
        except Exception as e:
            logger.error(f"Ошибка загрузки CSV: {e}")

    def get_timestamp(self, frame_number: int) -> float:
        """Получение временной метки для кадра"""
        if 0 <= frame_number < len(self.timestamps):
            return self.timestamps[frame_number]
        return 0.0