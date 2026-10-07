"""Геометрія та допоміжні перетворення без залежності від OpenCV."""

import math
import re


def distance(a, b):
    """Обчислити евклідову відстань між точками.

    Args:
        a (tuple[float, float]): Координати першої точки (x, y).
        b (tuple[float, float]): Координати другої точки (x, y); вершина кута для straight.

    Returns:
        float: Евклідова відстань між точками.
    """
    return math.dist(a, b)


def straight(a, b, c, cosine_threshold):
    """Перевірити випрямлення суглоба b за косинусом кута abc.

    Args:
        a (tuple[float, float]): Координати першої точки (x, y).
        b (tuple[float, float]): Координати другої точки (x, y); вершина кута для straight.
        c (tuple[float, float]): Координати третьої точки (x, y).
        cosine_threshold (float): Граничний косинус кута для випрямленого суглоба.

    Returns:
        bool: Чи є суглоб достатньо випрямленим.
    """
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    norm = math.hypot(*u) * math.hypot(*v)

    return norm > 0 and (u[0] * v[0] + u[1] * v[1]) / norm < cosine_threshold


def natural_key(path):
    """Сортувати назви так, щоб slide2 передував slide10.

    Args:
        path (pathlib.Path): Шлях до файлу.

    Returns:
        list[str | int]: Частини назви для природного сортування.
    """
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', path.name)]


def landmark_point(landmarks, indices, width, height):
    """Перенести середню позицію вибраних точок руки в межі полотна.

    Args:
        landmarks (Sequence[NormalizedLandmark]): Точки MediaPipe з нормалізованими координатами x і y.
        indices (Sequence[int]): Індекси точок руки, середнє положення яких використовується.
        width (int): Ширина цільового зображення у пікселях.
        height (int): Висота цільового зображення у пікселях.

    Returns:
        tuple[int, int]: Координати (x, y), обмежені розмірами полотна.
    """
    x = sum(landmarks[i].x for i in indices) / len(indices)
    y = sum(landmarks[i].y for i in indices) / len(indices)

    return (int(min(1, max(0, x)) * (width - 1)),
            int(min(1, max(0, y)) * (height - 1)))
