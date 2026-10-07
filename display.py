"""Вікна OpenCV та візуальні підказки; полотна тут не змінюються."""

import cv2
import numpy as np

from gestures import Gesture
from presentation import composite


def create_windows(cfg):
    """Відкрити два вікна з розмірами з YAML."""
    for title in (cfg.presentation_title, cfg.camera_title):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, *cfg.window_size)


def render_slide(state, point, mode, cfg):
    """Побудувати слайд із малюнком або заставку очікування."""
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
    """Показати точки руки, прогрес підтвердження, номер слайда та колір."""
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
    """Перевірити клавіші виходу та закриття будь-якого вікна."""
    key = cv2.waitKey(cfg.wait_key_ms) & 0xFF
    closed = any(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
                 for title in (cfg.presentation_title, cfg.camera_title))

    return key in cfg.exit_keys or closed
