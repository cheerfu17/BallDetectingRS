from threading import Lock, Condition, Event
from typing import List, Any
import csv
import os
from datetime import datetime

class ThreadSafeSingleton:
    """
    Синхронизатор с диагностикой.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        print("Инициализация системы синхронизации...")
        
        self._sync_lock = Lock()
        self._sync_condition = Condition(self._sync_lock)

        self.depth_timestamp = -1.0
        self.default_timestamp = -1.0
        
        self.default_coords = []
        self.default_frame_num = 0
        self.depth_in_polygon = False

        # Порог синхронизации 35мс (чуть больше одного кадра 30fps)
        self.SYNC_THRESHOLD_MS = 100.0 
        self.MAX_AHEAD_MS = 100.0 # Увеличим буфер

        self._init_results_file()
        
        # Флаг для остановки всех потоков, если один завершился
        self.stop_event = Event()

    def _init_results_file(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        output_dir = os.path.join(root_dir, 'data', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"valid_hits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_path = os.path.join(output_dir, filename)
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame_Default', 'Timestamp_Default', 'Coords_Default', 'Timestamp_Realsense', 'Time_Diff'])
        
        print(f"ФАЙЛ РЕЗУЛЬТАТОВ: {self.csv_path}")

    def _write_hit(self, diff):
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                coords_str = ";".join([f"({x},{y})" for x, y in self.default_coords])
                
                writer.writerow([
                    self.default_frame_num,
                    f"{self.default_timestamp:.1f}",
                    coords_str,
                    f"{self.depth_timestamp:.1f}",
                    f"{diff:.1f}"
                ])
                print(f">>> ЗАПИСАНО! DefaultFrame: {self.default_frame_num} | Coords: {coords_str} | Diff: {diff:.1f}ms")
        except Exception as e:
            print(f"Ошибка записи: {e}")

    def request_stop(self):
        """Сигнал всем потокам остановиться"""
        self.stop_event.set()
        with self._sync_condition:
            self._sync_condition.notify_all()

    def should_stop(self):
        return self.stop_event.is_set()

    def sync_depth_cam(self, timestamp: float, is_in_polygon: bool):
        if self.stop_event.is_set(): return

        with self._sync_condition:
            self.depth_timestamp = timestamp
            self.depth_in_polygon = is_in_polygon
            
            # Ждем DefaultCam
            while (self.default_timestamp >= 0 and 
                   self.depth_timestamp > self.default_timestamp + self.MAX_AHEAD_MS and
                   not self.stop_event.is_set()):
                self._sync_condition.wait(timeout=0.1) # Таймаут чтобы проверять stop_event

            self._check_and_record()
            self._sync_condition.notify_all()

    def sync_default_cam(self, timestamp: float, coords: Any, frame_num: int):
        if self.stop_event.is_set(): return

        with self._sync_condition:
            self.default_timestamp = timestamp
            self.default_coords = coords
            self.default_frame_num = frame_num

            # Ждем DepthCam
            while (self.depth_timestamp >= 0 and 
                   self.default_timestamp > self.depth_timestamp + self.MAX_AHEAD_MS and
                   not self.stop_event.is_set()):
                self._sync_condition.wait(timeout=0.1)

            self._check_and_record()
            self._sync_condition.notify_all()

    def _check_and_record(self):
        if self.depth_timestamp < 0 or self.default_timestamp < 0:
            return

        diff = abs(self.depth_timestamp - self.default_timestamp)
        
        # ЛОГИКА ЗАПИСИ
        if self.depth_in_polygon:
            if diff <= self.SYNC_THRESHOLD_MS:
                if self.default_coords:
                    self._write_hit(diff)
                else:
                    # Попадание в полигон ЕСТЬ, синхронизация ЕСТЬ, но обычная камера НИЧЕГО НЕ НАШЛА
                    print(f"Missed Hit: Polygon YES, Sync YES ({diff:.1f}ms), but Default Coords EMPTY")
                    pass
            else:
                # Попадание в полигон ЕСТЬ, но РАССИНХРОН
                print(f"Missed Hit: Polygon YES, but Sync NO (Diff: {diff:.1f}ms > {self.SYNC_THRESHOLD_MS})")
                pass

    # Заглушки
    def get_paused_default_cam(self): return False
    def get_paused_depth_cam(self): return False
    def pause_default_cam(self): pass
    def resume_default_cam(self): pass
    def set_timestamp_depth_cam(self, t): pass
    def set_timestamp_default_cam(self, t): pass
    def get_event_default_cam(self):
        from threading import Event
        e = Event(); e.set(); return e
