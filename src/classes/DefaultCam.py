import cv2
import numpy as np
from os import path
import logging
from src.classes.general.VideoWriterManager import VideoWriterManager
from src.classes.default_cam.VisualizationManager import VisualizationManager
from src.classes.general.data.CameraConfig import CameraConfig
from src.classes.default_cam.DetectionFilter import DetectionFilter
from src.classes.default_cam.MotionDetector import MotionDetector
from src.classes.default_cam.TimestampReader import TimestampReader
from src.classes.default_cam.Tracker import Tracker
from src.classes.default_cam.VideoProcessor import VideoProcessor
from src.classes.default_cam.LiveVideoCapture import LiveVideoCapture
from src.default_configs.default_cam_config import DEFAULT_CONFIG

parent_dir = path.dirname(path.abspath(__file__))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefaultCamProcessor:
    def __init__(self, video_path: str = None, output_path: str = None,
                 mask_output_path: str = None, config: dict = None):
        self.video_path = video_path
        self.output_path = output_path
        self.mask_output_path = mask_output_path
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Выбор источника: файл или живая камера
        self.use_live_camera = self.config.get('use_live_camera', False)

        if self.use_live_camera:
            camera_id = self.config.get('camera_id', 0)
            self.video_processor = LiveVideoCapture(
                camera_id=camera_id,
                width=self.config.get('width', 640),
                height=self.config.get('height', 480),
                fps=self.config.get('fps', 30)
            )
            self.camera_config = self.video_processor.initialize()
        else:
            if not video_path:
                raise ValueError("video_path должен быть указан при использовании файла")
            self.video_processor = VideoProcessor(video_path)
            self.camera_config = CameraConfig(
                width=self.video_processor.width,
                height=self.video_processor.height,
                fps=self.video_processor.fps,
                depth_scale=0.0
            )

        # Инициализация компонентов
        self.video_writer = VideoWriterManager(output_path, mask_output_path, self.camera_config)
        self.timestamp_reader = TimestampReader(self.config['csv_file']) if not self.use_live_camera else None
        self.motion_detector = MotionDetector(self.config)
        self.detection_filter = DetectionFilter(self.config)
        self.tracker = Tracker(self.config)
        self.visualization = VisualizationManager(
            self.camera_config.width,
            self.camera_config.height
        )

        self.paused = False
        self._is_initialized = False
        source_type = "живая камера" if self.use_live_camera else f"файл {video_path}"
        logger.info(f"DefaultCamProcessor создан для: {source_type}")

    def initialize(self):
        if self._is_initialized:
            return
        self.video_writer.initialize()
        self._is_initialized = True
        logger.info("DefaultCam инициализирован")

    def process_frame(self, state) -> bool:
        # Получение кадра
        ret, frame = self.video_processor.read_frame()
        if not ret:
            return False

        # Получение временной метки
        if self.use_live_camera:
            timestamp = self.video_processor.get_timestamp()
        else:
            timestamp = self.timestamp_reader.get_timestamp(self.video_processor.frame_count - 1)

        self.visualization.start_frame_timer()

        # Обработка движения
        motion_mask = self.motion_detector.process_frame(frame)
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = self.detection_filter.filter_contours(contours)
        trajectories = self.tracker.update(detections)

        # Получение центров обнаружений
        detection_centers = []
        for det in detections:
            if hasattr(det, 'center'):
                detection_centers.append(det.center)
            elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 4:
                x, y, w, h = det[:4]
                center = (int(x + w / 2), int(y + h / 2))
                detection_centers.append(center)

        # Синхронизация состояния
        # state.sync_default_cam(timestamp, detection_centers, self.video_processor.frame_count)

        # Визуализация
        debug_frame = self.visualization.draw_detections(frame, detections)
        debug_frame = self.visualization.draw_trajectories(debug_frame, trajectories, self.tracker.colors)

        frame_time, current_fps, avg_fps, avg_time = self.visualization.end_frame_timer()
        debug_frame = self.visualization.draw_info_panel(
            debug_frame, self.video_processor.frame_count, frame_time,
            current_fps, avg_fps, timestamp, state
        )

        # Запись и отображение
        if self.config.get('record_output', True):
            self.video_writer.write(debug_frame, motion_mask)

        cv2.imshow('Default Tracking', debug_frame)
        cv2.imshow('Default Mask', motion_mask)

        return True

    def handle_keyboard(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return False
        elif key == ord('p') or key == ord('P'):
            self.paused = not self.paused
        elif key == ord('s'):  # Сохранить текущий кадр
            self._save_current_frame()
        return True

    def _save_current_frame(self):
        """Сохранение текущего кадра"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.png"
        cv2.imwrite(filename, self.last_frame)
        logger.info(f"Кадр сохранен: {filename}")

    def run(self, state):
        try:
            if not self._is_initialized:
                self.initialize()

            logger.info("Запуск обработки DefaultCam (живой режим)...")
            while not state.should_stop():
                if not self.paused:
                    if not self.process_frame(state):
                        logger.info("DefaultCam: Поток закончился.")
                        state.request_stop()
                        break
                if not self.handle_keyboard():
                    state.request_stop()
                    break
        except KeyboardInterrupt:
            logger.info("DefaultCam прерван пользователем")
        except Exception as e:
            logger.error(f"Ошибка DefaultCam: {e}")
            state.request_stop()
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Очистка ресурсов DefaultCam...")
        self.video_processor.release()
        self.video_writer.release()
        self._print_statistics()

    def _print_statistics(self):
        if self.visualization.frame_times:
            avg_frame_time = np.mean(self.visualization.frame_times) * 1000
            print(f"DEFAULT CAM: {self.video_processor.frame_count} кадров, "
                  f"{avg_frame_time:.2f} мс/кадр")