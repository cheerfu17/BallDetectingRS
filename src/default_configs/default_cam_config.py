from os import path

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    'csv_file': path.join(path.dirname(path.abspath(__file__)), '..\\..\\data\\input\\csv\\', 'timestamps.csv'),
    'min_area': 15,
    'max_area': 400,
    'min_speed': 20.0,
    'max_speed': 250.0,
    'track_distance': 350,
    'max_missed_frames': 3,
    'max_aspect_ratio': 6.0,
    'motion_threshold': 25,
    'background_history': 100,
    'background_threshold': 16,
    'trajectory_length': 30,
    'frame_time_buffer_size': 100,
    'dilation_kernel_size': (3, 3),
    'gaussian_blur_size': (5, 5)
}