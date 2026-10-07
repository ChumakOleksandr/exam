"""Геометрія та допоміжні перетворення без залежності від OpenCV."""

import math
import re


def distance(a, b):
    """Обчислити евклідову відстань між точками."""
    return math.dist(a, b)


def straight(a, b, c, cosine_threshold):
    """Перевірити випрямлення суглоба b за косинусом кута abc."""
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    norm = math.hypot(*u) * math.hypot(*v)

    return norm > 0 and (u[0] * v[0] + u[1] * v[1]) / norm < cosine_threshold


def natural_key(path):
    """Сортувати назви так, щоб slide2 передував slide10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', path.name)]


def landmark_point(landmarks, indices, width, height):
    """Перенести середню позицію вибраних точок руки в межі полотна."""
    x = sum(landmarks[i].x for i in indices) / len(indices)
    y = sum(landmarks[i].y for i in indices) / len(indices)

    return (int(min(1, max(0, x)) * (width - 1)),
            int(min(1, max(0, y)) * (height - 1)))
