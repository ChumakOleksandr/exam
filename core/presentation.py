"""Завантаження слайдів, стан презентації та малювання на полотнах."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.gestures import Gesture
from core.utils import landmark_point, natural_key


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
        """Створити чорне полотно розміру кожного слайда.

        Args:
            slides (list[numpy.ndarray]): Початкові BGR-зображення слайдів.

        Returns:
            PresentationState: Новий стан із чорними полотнами для кожного слайда.
        """
        return cls(slides, [np.zeros_like(slide) for slide in slides])


def load_slides(folder: Path, extensions):
    """Прочитати зображення в природному порядку, підтримуючи Unicode-шляхи.

    Args:
        folder (pathlib.Path): Каталог із зображеннями слайдів.
        extensions (Sequence[str]): Дозволені розширення з крапкою, наприклад .png.

    Returns:
        list[numpy.ndarray]: Прочитані BGR-слайди у природному порядку.
    """
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
    """Накласти нечорні пікселі полотна на копію початкового слайда.

    Args:
        slide (numpy.ndarray): Оригінальний слайд uint8 у BGR; не змінюється.
        canvas (numpy.ndarray): Полотно uint8 розміру слайда; чорний означає відсутність малюнка.

    Returns:
        numpy.ndarray: Нова копія слайда з накладеними нечорними пікселями полотна.
    """
    result = slide.copy()
    mask = np.any(canvas != 0, axis=2)
    result[mask] = canvas[mask]

    return result


def stroke(canvas, point, previous, color, thickness):
    """З'єднати точки лінією; перша точка не створює окремого кола.

    Args:
        canvas (numpy.ndarray): Полотно uint8 розміру слайда; чорний означає відсутність малюнка.
        point (tuple[int, int]): Поточна точка штриха (x, y) у пікселях.
        previous (tuple[int, int] | None): Попередня точка штриха; None починає новий штрих.
        color (Sequence[int]): Три компоненти кольору в порядку BGR, від 0 до 255.
        thickness (int): Товщина лінії в пікселях.

    Returns:
        None: Функція не повертає значення.
    """
    if previous is not None:
        cv2.line(canvas, point, previous, tuple(color), thickness)


def handle_gesture(state, event: Gesture | None, drawing):
    """Виконати одноразову команду: запуск, навігацію або зміну кольору.

    Args:
        state (PresentationState): Змінюваний стан сеансу: слайди, полотна та поточні індекси.
        event (Gesture | None): Нова підтверджена одноразова команда або None.
        drawing (DrawingConfig): Палітра, товщини та точки керування малюванням.

    Returns:
        None: Функція не повертає значення.
    """
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
    """Малювати або стирати; при втраті жесту розірвати поточний штрих.

    Args:
        state (PresentationState): Змінюваний стан сеансу: слайди, полотна та поточні індекси.
        active (Gesture | None): Жест, який уже пройшов часову перевірку.
        hand (Sequence[NormalizedLandmark] | None): Точки вибраної руки або None, якщо руки не знайдено.
        cfg (DrawingConfig): Налаштування, потрібні для цієї операції.

    Returns:
        tuple[tuple[int, int] | None, Gesture | None]: Позиція курсора та режим малювання.
    """
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
