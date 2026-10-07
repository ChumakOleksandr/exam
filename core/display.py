"""Вікна OpenCV та візуальні підказки; полотна тут не змінюються."""

import cv2
import numpy as np

from core.gestures import Gesture
from core.presentation import composite


def create_windows(cfg):
    """Відкрити два вікна з розмірами з YAML.

    Args:
        cfg (DisplayConfig): Налаштування, потрібні для цієї операції.

    Returns:
        None: Функція не повертає значення.
    """
    for title in (cfg.presentation_title, cfg.camera_title):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, *cfg.window_size)


def render_slide(state, point, mode, cfg):
    """Побудувати слайд із малюнком або заставку очікування.

    Args:
        state (PresentationState): Змінюваний стан сеансу: слайди, полотна та поточні індекси.
        point (tuple[int, int] | None): Поточна позиція курсора (x, y) у пікселях або None.
        mode (Gesture | None): Режим DRAW, ERASE або відсутність малювання.
        cfg (AppConfig): Налаштування, потрібні для цієї операції.

    Returns:
        numpy.ndarray: BGR-зображення слайда або заставки.
    """
    ui = cfg.display

    if not state.started:
        screen = np.zeros_like(state.slides[0])
        text = f'Hold an open RIGHT palm > {cfg.gestures.hold_seconds:g} sec to start'
        cv2.putText(screen, text, tuple(ui.welcome_origin), cv2.FONT_HERSHEY_SIMPLEX,
                    ui.welcome_scale, tuple(ui.text_color), ui.welcome_thickness)
        return screen

    screen = composite(state.slides[state.index], state.canvases[state.index])
    if point is not None:
        radius = ui.cursor_radius if mode is Gesture.DRAW else max(1, cfg.drawing.eraser_thickness // 2)
        cv2.circle(screen, point, radius, tuple(ui.cursor_color), ui.cursor_thickness)

    return screen


def render_camera(frame, hand, gesture, gate, now, state, cfg):
    """Показати точки руки, прогрес підтвердження, номер слайда та колір.

    Args:
        frame (numpy.ndarray): Кадр камери uint8 форми (висота, ширина, 3) у BGR.
        hand (Sequence[NormalizedLandmark] | None): Точки вибраної руки або None, якщо руки не знайдено.
        gesture (Gesture | None): Розпізнаний жест або None, якщо його немає.
        gate (GestureGate): Стан таймера підтвердження поточного жесту.
        now (float): Поточний монотонний час у секундах.
        state (PresentationState): Змінюваний стан сеансу: слайди, полотна та поточні індекси.
        cfg (AppConfig): Налаштування, потрібні для цієї операції.

    Returns:
        numpy.ndarray: Вхідний кадр, доповнений точками й текстом на місці.
    """
    ui = cfg.display
    height, width = frame.shape[:2]
    if hand is not None:
        for landmark in hand:
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(frame, point, ui.landmark_radius, tuple(ui.landmark_color), cv2.FILLED)

    progress = min(1.0, (now - gate.since) / gate.hold) if gesture is not None else 0
    name = gesture.value if gesture is not None else 'none'
    lines = [
        f'RIGHT hand | Gesture: {name} | {progress:.0%}',
        f'Slide {state.index + 1}/{len(state.slides)} | Color: {cfg.drawing.color_names[state.color_index]}',
        'Palm: start/clear | Index: next | Thumb: previous',
        'V: draw | V+thumb: color | Fist: erase',
    ]

    for index, text in enumerate(lines):
        origin = (ui.text_origin[0], ui.text_origin[1] + ui.line_spacing * index)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    ui.text_scale, tuple(ui.outline_color), ui.outline_thickness)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    ui.text_scale, tuple(ui.text_color), ui.text_thickness)

    return frame


def should_exit(cfg):
    """Перевірити клавіші виходу та закриття будь-якого вікна.

    Args:
        cfg (DisplayConfig): Налаштування, потрібні для цієї операції.

    Returns:
        bool: Чи натиснута клавіша виходу або закрите вікно.
    """
    key = cv2.waitKey(cfg.wait_key_ms) & 0xFF
    closed = any(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
                 for title in (cfg.presentation_title, cfg.camera_title))

    return key in cfg.exit_keys or closed
