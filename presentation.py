"""Завантаження слайдів, стан презентації та малювання на полотнах."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gestures import Gesture
from utils import landmark_point, natural_key


@dataclass
class PresentationState:
    """Стан сеансу, окремі полотна та попередня точка штриха."""

    slides: list
    canvases: list
    index: int = 0
    color_index: int = 0
    started: bool = False
    previous: tuple[int, int] | None = None
    previous_mode: Gesture | None = None

    @classmethod
    def create(cls, slides):
        """Створити чорне полотно розміру кожного слайда."""
        return cls(slides, [np.zeros_like(slide) for slide in slides])


def load_slides(folder: Path, extensions):
    """Прочитати зображення в природному порядку, підтримуючи Unicode-шляхи."""
    if not folder.is_dir():
        raise ValueError(f'Slides folder not found: {folder}')

    allowed = {ext.lower() for ext in extensions}
    paths = sorted((path for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in allowed), key=natural_key)
    slides = []
    for path in paths:
        slide = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if slide is None:
            raise ValueError(f'Cannot read slide: {path}')
        slides.append(slide)

    if not slides:
        raise ValueError(f'No slides in {folder}. Add images: {", ".join(extensions)}')

    return slides


def composite(slide, canvas):
    """Накласти нечорні пікселі полотна на копію початкового слайда."""
    result = slide.copy()
    mask = np.any(canvas != 0, axis=2)
    result[mask] = canvas[mask]

    return result


def stroke(canvas, point, previous, color, thickness):
    """З'єднати точки лінією; перша точка не створює окремого кола."""
    if previous is not None:
        cv2.line(canvas, point, previous, tuple(color), thickness)


def handle_gesture(state, event: Gesture | None, drawing):
    """Виконати одноразову команду: запуск, навігацію або зміну кольору."""
    if event is Gesture.START:
        state.started = True
        for canvas in state.canvases:
            canvas.fill(0)

    elif state.started:
        if event is Gesture.NEXT:
            state.index = min(state.index + 1, len(state.slides) - 1)
        elif event is Gesture.PREVIOUS:
            state.index = max(state.index - 1, 0)
        elif event is Gesture.COLOR:
            state.color_index = (state.color_index + 1) % len(drawing.colors)

    if event in (Gesture.START, Gesture.NEXT, Gesture.PREVIOUS, Gesture.COLOR):
        state.previous = None
        state.previous_mode = None


def update_drawing(state, active, hand, cfg):
    """Малювати або стирати; при втраті жесту розірвати поточний штрих."""
    mode = active if state.started and active in (Gesture.DRAW, Gesture.ERASE) else None
    point = None

    if mode is not None and hand is not None:
        canvas = state.canvases[state.index]
        height, width = canvas.shape[:2]
        indices = cfg.pointer_points if mode is Gesture.DRAW else cfg.eraser_points
        point = landmark_point(hand, indices, width, height)

        if state.previous_mode is not mode:
            state.previous = None

        # Чорний колір означає відсутність малюнка у масці.
        color = cfg.colors[state.color_index] if mode is Gesture.DRAW else (0, 0, 0)
        thickness = cfg.thickness if mode is Gesture.DRAW else cfg.eraser_thickness
        stroke(canvas, point, state.previous, color, thickness)

    state.previous, state.previous_mode = point, mode
    return point, mode
