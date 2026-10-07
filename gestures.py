"""Enum жестів, геометрична класифікація та часовий фільтр."""

from enum import Enum

from config import GestureConfig
from utils import distance, straight


class Gesture(Enum):
    """Команди презентації; значення рядків використовуються лише для показу."""

    START = 'start'
    NEXT = 'next'
    PREVIOUS = 'previous'
    DRAW = 'draw'
    COLOR = 'color'
    ERASE = 'erase'


class GestureGate:
    """Підтверджувати утриманий жест і видавати разову команду один раз."""

    def __init__(self, hold: float):
        """Ініціалізувати таймер заданою тривалістю утримання."""
        self.hold = hold
        self.gesture = None
        self.since = 0.0
        self.fired = False

    def update(self, gesture: Gesture | None, now: float):
        """Повернути (нова подія, активний жест); зміна скидає таймер."""
        if gesture != self.gesture:
            self.gesture, self.since, self.fired = gesture, now, False

        ready = gesture is not None and now - self.since > self.hold
        event = gesture if ready and not self.fired else None
        if ready:
            self.fired = True

        return event, gesture if ready else None


def classify(landmarks, width: int, height: int, cfg: GestureConfig) -> Gesture | None:
    """Визначити жест за кутами та відносними відстанями точок руки."""
    points = [(v.x * width, v.y * height) for v in landmarks]
    wrist, palm = cfg.palm_scale_points
    scale = max(distance(points[wrist], points[palm]), cfg.min_palm_scale)

    up = [straight(points[t - cfg.mcp_offset], points[t - cfg.pip_offset],
                   points[t], cfg.straight_cosine)
          and points[t][1] < points[t - cfg.pip_offset][1] - cfg.raised_margin * scale
          for t in cfg.finger_tips]

    base, joint, tip = cfg.thumb_points
    thumb = (straight(points[base], points[joint], points[tip], cfg.straight_cosine)
             and distance(points[tip], points[cfg.thumb_reference]) > cfg.thumb_distance * scale)
    sideways = thumb and (abs(points[tip][0] - points[base][0])
                          > cfg.sideways_ratio * abs(points[tip][1] - points[base][1]))
    folded = [distance(points[t], points[wrist])
              < distance(points[t - cfg.pip_offset], points[wrist]) + cfg.folded_margin * scale
              for t in cfg.finger_tips]

    if all(up) and thumb:
        return Gesture.START

    if all(up[:2]) and all(folded[2:]):
        return Gesture.COLOR if sideways else (Gesture.DRAW if not thumb else None)

    if up[0] and all(folded[1:]) and not thumb:
        return Gesture.NEXT

    if all(folded) and sideways:
        return Gesture.PREVIOUS

    if all(folded) and not thumb:
        return Gesture.ERASE

    return None
