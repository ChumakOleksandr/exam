"""Підготовка кадру та вибір правої руки з результатів MediaPipe."""

import cv2
import mediapipe as mp


def create_detector(cfg):
    """Створити MediaPipe Hands з параметрами конфігурації.

    Args:
        cfg (DetectionConfig): Налаштування, потрібні для цієї операції.

    Returns:
        mediapipe.solutions.hands.Hands: Детектор, який потрібно закрити після роботи.
    """
    return mp.solutions.hands.Hands(
        max_num_hands=cfg.max_num_hands,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )


def prepare_frame(frame, cfg):
    """Віддзеркалити кадр та приглушити шум медіанним фільтром.

    Args:
        frame (numpy.ndarray): Кадр камери uint8 форми (висота, ширина, 3) у BGR.
        cfg (CameraConfig): Налаштування, потрібні для цієї операції.

    Returns:
        numpy.ndarray: Віддзеркалений за налаштуваннями та відфільтрований BGR-кадр.
    """
    if cfg.mirror:
        frame = cv2.flip(frame, 1)

    return cv2.medianBlur(frame, cfg.median_kernel)


def detect_hand(frame, detector, cfg):
    """Обробити RGB-кадр і повернути точки руки, дозволеної налаштуваннями.

    Args:
        frame (numpy.ndarray): Кадр камери uint8 форми (висота, ширина, 3) у BGR.
        detector (mediapipe.solutions.hands.Hands): Ініціалізована модель пошуку точок руки.
        cfg (AppConfig): Налаштування, потрібні для цієї операції.

    Returns:
        Sequence[NormalizedLandmark] | None: Точки дозволеної руки або None.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb_frame)
    target = 'Left' if cfg.camera.swap_hands else 'Right'

    if not result.multi_hand_landmarks:
        return None

    for landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness or []):
        label = handedness.classification[0]
        if label.label == target and label.score >= cfg.detection.handedness_confidence:
            return landmarks.landmark

    return None
