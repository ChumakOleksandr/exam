"""Типізовані налаштування та перевірка YAML перед запуском камери."""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'config.yaml'


@dataclass(frozen=True)
class PathsConfig:
    """Матеріали презентації."""

    slides: str
    extensions: list[str]


@dataclass(frozen=True)
class CameraConfig:
    """Джерело відео та обробка зображення."""

    index: int
    mirror: bool
    median_kernel: int
    swap_hands: bool


@dataclass(frozen=True)
class DetectionConfig:
    """Параметри моделі MediaPipe."""

    max_num_hands: int
    min_detection_confidence: float
    min_tracking_confidence: float
    handedness_confidence: float


@dataclass(frozen=True)
class GestureConfig:
    """Геометричні пороги, час підтвердження та індекси точок."""

    hold_seconds: float
    straight_cosine: float
    raised_margin: float
    thumb_distance: float
    folded_margin: float
    sideways_ratio: float
    min_palm_scale: float
    palm_scale_points: list[int]
    finger_tips: list[int]
    mcp_offset: int
    pip_offset: int
    thumb_points: list[int]
    thumb_reference: int


@dataclass(frozen=True)
class DrawingConfig:
    """Палітра, товщини та точки керування пензлем."""

    colors: list[list[int]]
    color_names: list[str]
    thickness: int
    eraser_thickness: int
    pointer_points: list[int]
    eraser_points: list[int]


@dataclass(frozen=True)
class DisplayConfig:
    """Вікна, курсор, підказки та клавіші."""

    presentation_title: str
    camera_title: str
    window_size: list[int]
    cursor_color: list[int]
    cursor_radius: int
    cursor_thickness: int
    landmark_color: list[int]
    landmark_radius: int
    text_color: list[int]
    outline_color: list[int]
    text_scale: float
    text_thickness: int
    outline_thickness: int
    text_origin: list[int]
    line_spacing: int
    welcome_origin: list[int]
    welcome_scale: float
    welcome_thickness: int
    wait_key_ms: int
    exit_keys: list[int]


@dataclass(frozen=True)
class DownloadConfig:
    """Адреса джерела і мережеві тайм-аути."""

    source_url: str
    listing_timeout: float
    file_timeout: float


@dataclass(frozen=True)
class AppConfig:
    """Перевірена конфігурація з каталогом для відносних шляхів."""

    paths: PathsConfig
    camera: CameraConfig
    detection: DetectionConfig
    gestures: GestureConfig
    drawing: DrawingConfig
    display: DisplayConfig
    downloads: DownloadConfig
    directory: Path

    @property
    def slides_dir(self) -> Path:
        """Повернути шлях слайдів відносно каталогу YAML."""
        return (self.directory / self.paths.slides).resolve()


def require(condition, message):
    """Підняти зрозумілу помилку для некоректного налаштування."""
    if not condition:
        raise ValueError(f'Config: {message}')


def matches_type(value, expected):
    """Перевірити скаляр або вкладений список за анотацією поля."""
    if getattr(expected, '__origin__', None) is list:
        return isinstance(value, list) and all(
            matches_type(item, expected.__args__[0]) for item in value
        )

    if expected is float:
        return type(value) in (float, int)

    return type(value) is expected


def make_section(cls, values, name):
    """Відхилити зайві/пропущені поля і створити типізовану секцію."""
    require(isinstance(values, dict), f'{name} must be a mapping')
    require(set(values) == {field.name for field in fields(cls)}, f'check fields in {name}')

    for field in fields(cls):
        require(matches_type(values[field.name], field.type),
                f'invalid type for {name}.{field.name}')

    return cls(**values)


def validate(cfg):
    """Перевірити діапазони, палітру, розміри та індекси MediaPipe."""
    camera, detection, gesture = cfg.camera, cfg.detection, cfg.gestures
    pen, ui = cfg.drawing, cfg.display

    require(camera.index >= 0, 'camera.index must be non-negative')
    require(camera.median_kernel > 0 and camera.median_kernel % 2 == 1,
            'camera.median_kernel must be positive and odd')
    require(detection.max_num_hands > 0, 'max_num_hands must be positive')

    for value in (detection.min_detection_confidence,
                  detection.min_tracking_confidence, detection.handedness_confidence):
        require(0 <= value <= 1, 'confidence must be between 0 and 1')

    require(gesture.hold_seconds >= 1, 'hold_seconds must be at least 1')
    require(-1 <= gesture.straight_cosine <= 1, 'straight_cosine must be between -1 and 1')
    require(all(v >= 0 for v in (gesture.raised_margin, gesture.thumb_distance,
                                gesture.folded_margin)), 'gesture margins must be non-negative')
    require(gesture.min_palm_scale > 0 and gesture.sideways_ratio > 0, 'scales must be positive')
    require(len(gesture.finger_tips) == 4 and len(gesture.thumb_points) == 3
            and len(gesture.palm_scale_points) == 2, 'invalid landmark list lengths')
    require(gesture.mcp_offset > gesture.pip_offset > 0, 'expected mcp_offset > pip_offset > 0')
    require(bool(pen.pointer_points) and bool(pen.eraser_points), 'empty pointer list')

    indices = (gesture.finger_tips + gesture.thumb_points + gesture.palm_scale_points
               + [gesture.thumb_reference] + pen.pointer_points + pen.eraser_points
               + [tip - offset for tip in gesture.finger_tips
                  for offset in (gesture.mcp_offset, gesture.pip_offset)])
    require(all(0 <= index < 21 for index in indices), 'landmark indices must be 0..20')

    require(bool(pen.colors) and len(pen.colors) == len(pen.color_names), 'palette and names must match')
    for color in pen.colors + [ui.cursor_color, ui.landmark_color, ui.text_color, ui.outline_color]:
        require(len(color) == 3 and all(0 <= v <= 255 for v in color), 'invalid BGR color')
    require(all(any(color) for color in pen.colors), 'black is reserved for erasing')

    require(len(ui.window_size) == 2 and all(v > 0 for v in ui.window_size), 'invalid window_size')
    require(len(ui.text_origin) == len(ui.welcome_origin) == 2, 'invalid text origins')
    require(bool(ui.presentation_title) and bool(ui.camera_title)
            and ui.presentation_title != ui.camera_title, 'window titles must differ')
    require(bool(ui.exit_keys) and all(0 <= key <= 255 for key in ui.exit_keys), 'invalid exit_keys')

    for value in (pen.thickness, pen.eraser_thickness, ui.cursor_radius, ui.cursor_thickness,
                  ui.landmark_radius, ui.text_scale, ui.text_thickness, ui.outline_thickness,
                  ui.line_spacing, ui.welcome_scale, ui.welcome_thickness, ui.wait_key_ms,
                  cfg.downloads.listing_timeout, cfg.downloads.file_timeout):
        require(value > 0, 'sizes and timeouts must be positive')

    require(bool(cfg.paths.slides), 'paths.slides must not be empty')
    require(bool(cfg.paths.extensions) and all(ext.startswith('.') for ext in cfg.paths.extensions),
            'extensions must begin with a dot')
    require(cfg.downloads.source_url.startswith('https://'), 'source_url must use HTTPS')


def load_config(path: Path = DEFAULT_CONFIG) -> AppConfig:
    """Прочитати повний YAML через safe_load та перевірити всі секції."""
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f'Invalid YAML: {exc}') from exc

    sections = dict(paths=PathsConfig, camera=CameraConfig, detection=DetectionConfig,
                    gestures=GestureConfig, drawing=DrawingConfig,
                    display=DisplayConfig, downloads=DownloadConfig)
    require(isinstance(data, dict) and set(data) == set(sections), 'check YAML section names')
    cfg = AppConfig(**{name: make_section(cls, data[name], name)
                       for name, cls in sections.items()}, directory=path.parent)
    validate(cfg)

    return cfg
