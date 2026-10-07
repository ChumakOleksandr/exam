"""Керування презентацією правою рукою: OpenCV + MediaPipe Hands."""

import argparse
from dataclasses import replace
from pathlib import Path
import time

import cv2

from config.config import DEFAULT_CONFIG, load_config, validate
from core.display import create_windows, render_camera, render_slide, should_exit
from core.gestures import GestureGate, classify
from core.presentation import PresentationState, handle_gesture, load_slides, update_drawing
from core.vision import create_detector, detect_hand, prepare_frame


def parse_args(argv=None):
    """Прочитати параметри; None означає використати значення з YAML.

    Args:
        argv (list[str] | None): Аргументи командного рядка; None читає sys.argv.

    Returns:
        argparse.Namespace: Розібрані параметри командного рядка.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='Path to full YAML configuration')
    parser.add_argument('--camera', type=int, default=None, help='Override camera index from YAML')
    parser.add_argument('--slides', type=Path, default=None, help='Override slides folder (relative to working directory)')
    parser.add_argument('--swap-hands', action=argparse.BooleanOptionalAction, default=None,
                        help='Swap handedness labels; --no-swap-hands disables the swap')

    return parser.parse_args(argv)


def configure(args):
    """Завантажити YAML і застосувати явні параметри командного рядка.

    Args:
        args (argparse.Namespace): Параметри, отримані від parse_args.

    Returns:
        tuple[AppConfig, pathlib.Path]: Конфігурація з CLI-перевизначеннями та шлях слайдів.
    """
    cfg = load_config(args.config)
    camera = replace(
        cfg.camera,
        index=cfg.camera.index if args.camera is None else args.camera,
        swap_hands=cfg.camera.swap_hands if args.swap_hands is None else args.swap_hands,
    )
    cfg = replace(cfg, camera=camera)
    validate(cfg)
    folder = cfg.slides_dir if args.slides is None else args.slides.resolve()

    return cfg, folder


def process_frame(frame, detector, gate, state, cfg):
    """Обробити один кадр: розпізнати жест, змінити стан і підготувати вікна.

    Args:
        frame (numpy.ndarray): Кадр камери uint8 форми (висота, ширина, 3) у BGR.
        detector (mediapipe.solutions.hands.Hands): Ініціалізована модель пошуку точок руки.
        gate (GestureGate): Стан таймера підтвердження поточного жесту.
        state (PresentationState): Змінюваний стан сеансу: слайди, полотна та поточні індекси.
        cfg (AppConfig): Налаштування, потрібні для цієї операції.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Зображення слайда та камери для показу.
    """
    frame = prepare_frame(frame, cfg.camera)
    hand = detect_hand(frame, detector, cfg)
    height, width = frame.shape[:2]
    gesture = classify(hand, width, height, cfg.gestures) if hand is not None else None
    now = time.monotonic()
    event, active = gate.update(gesture, now)

    handle_gesture(state, event, cfg.drawing)
    point, mode = update_drawing(state, active, hand, cfg.drawing)
    screen = render_slide(state, point, mode, cfg)
    preview = render_camera(frame, hand, gesture, gate, now, state, cfg)

    return screen, preview


def run_presentation(cfg, state):
    """Запустити цикл камери та звільнити ресурси навіть після помилки.

    Args:
        cfg (AppConfig): Налаштування, потрібні для цієї операції.
        state (PresentationState): Змінюваний стан сеансу: слайди, полотна та поточні індекси.

    Returns:
        None: Функція не повертає значення.
    """
    capture = cv2.VideoCapture(cfg.camera.index)
    gate = GestureGate(cfg.gestures.hold_seconds)

    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open camera. Check access or try --camera 1')

        create_windows(cfg.display)
        with create_detector(cfg.detection) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('Camera stopped delivering frames')

                screen, preview = process_frame(frame, detector, gate, state, cfg)
                cv2.imshow(cfg.display.presentation_title, screen)
                cv2.imshow(cfg.display.camera_title, preview)
                if should_exit(cfg.display):
                    break

    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(argv=None):
    """Зібрати конфігурацію, слайди та запустити презентацію.

    Args:
        argv (list[str] | None): Аргументи командного рядка; None читає sys.argv.

    Returns:
        None: Функція не повертає значення.
    """
    cfg, folder = configure(parse_args(argv))
    state = PresentationState.create(load_slides(folder, cfg.paths.extensions))
    run_presentation(cfg, state)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(f'Error: {exc}')
