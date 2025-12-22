from os import path
parent_dir = path.dirname(path.abspath(__file__))
# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    'output_video': path.join(parent_dir, '..\\..\\data\\output\\videos\\', 'output.mp4'),
    'debug_video': path.join(parent_dir, '..\\..\\data\\output\\videos\\', 'debug_output.mp4'),
    'csv_file': path.join(parent_dir, '..\\..\\data\\output\\csv\\', 'detections.csv'),
    'distance_min': 0.8,
    'distance_max': 2.45,
    'min_contour_area': 5,
    'min_valid_depth_points': 10
}