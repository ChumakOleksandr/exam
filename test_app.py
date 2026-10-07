"""Перевірки конфігурації, геометрії та поведінки без фізичної камери."""

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import yaml

from config import DEFAULT_CONFIG, load_config, validate
from gestures import Gesture, GestureGate, classify
from main import configure, parse_args, process_frame
from presentation import PresentationState, composite, handle_gesture, load_slides, stroke, update_drawing
from utils import distance, landmark_point, straight


class AppTests(unittest.TestCase):
    """Регресійні сценарії презентації та перевірка параметрів."""

    def setUp(self):
        """Завантажити стандартну конфігурацію для кожного тесту."""
        self.cfg = load_config()
        self.state = PresentationState.create([
            np.full((80, 100, 3), 127, dtype=np.uint8) for _ in range(2)
        ])

    def test_gate_requires_more_than_second_and_fires_once(self):
        """Довге утримання не перемикає слайди багато разів."""
        gate = GestureGate(self.cfg.gestures.hold_seconds)
        self.assertEqual(gate.update(Gesture.NEXT, 0), (None, None))
        self.assertEqual(gate.update(Gesture.NEXT, 1), (None, None))
        self.assertEqual(gate.update(Gesture.NEXT, 1.01), (Gesture.NEXT, Gesture.NEXT))
        self.assertEqual(gate.update(Gesture.NEXT, 5), (None, Gesture.NEXT))
        gate.update(None, 6)
        gate.update(Gesture.NEXT, 7)
        self.assertEqual(gate.update(Gesture.NEXT, 8.1), (Gesture.NEXT, Gesture.NEXT))

    def test_changed_gesture_resets_timer(self):
        """Інший жест або втрата руки відразу скидає підтвердження."""
        gate = GestureGate(1)
        gate.update(Gesture.DRAW, 0)
        gate.update(Gesture.COLOR, 0.9)
        self.assertEqual(gate.update(Gesture.DRAW, 1.1), (None, None))
        self.assertEqual(gate.update(Gesture.DRAW, 2), (None, None))
        self.assertEqual(gate.update(Gesture.DRAW, 2.2), (Gesture.DRAW, Gesture.DRAW))
        self.assertEqual(gate.update(None, 2.3), (None, None))

    def test_navigation_start_color_and_bounds(self):
        """Перевірити блокування до запуску, межі слайдів, очищення та палітру."""
        handle_gesture(self.state, Gesture.NEXT, self.cfg.drawing)
        self.assertEqual(self.state.index, 0)
        self.state.canvases[0].fill(255)
        handle_gesture(self.state, Gesture.START, self.cfg.drawing)
        self.assertTrue(self.state.started)
        self.assertFalse(self.state.canvases[0].any())

        for _ in range(3):
            handle_gesture(self.state, Gesture.NEXT, self.cfg.drawing)
        self.assertEqual(self.state.index, 1)
        for _ in range(3):
            handle_gesture(self.state, Gesture.PREVIOUS, self.cfg.drawing)
            handle_gesture(self.state, Gesture.COLOR, self.cfg.drawing)
        self.assertEqual(self.state.index, 0)
        self.assertEqual(self.state.color_index, 0)

    def test_mask_and_eraser_restore_original(self):
        """Гумка відкриває слайд, не змінюючи його оригінал."""
        slide, canvas = self.state.slides[0], self.state.canvases[0]
        stroke(canvas, (70, 40), (10, 40), (0, 0, 255), 6)
        np.testing.assert_array_equal(composite(slide, canvas)[40, 40], [0, 0, 255])
        stroke(canvas, (45, 40), (35, 40), (0, 0, 0), 20)
        np.testing.assert_array_equal(composite(slide, canvas)[40, 40], slide[40, 40])
        self.assertTrue(np.all(slide == 127))

    def test_lost_hand_breaks_stroke(self):
        """Повернення руки не створює лінії через весь слайд."""
        self.state.started = True
        hand = [SimpleNamespace(x=0.1, y=0.1) for _ in range(21)]
        update_drawing(self.state, Gesture.DRAW, hand, self.cfg.drawing)
        self.assertFalse(self.state.canvases[0].any())
        hand[8].x = 0.2
        update_drawing(self.state, Gesture.DRAW, hand, self.cfg.drawing)
        self.assertTrue(self.state.canvases[0].any())

        update_drawing(self.state, None, None, self.cfg.drawing)
        before = self.state.canvases[0].copy()
        hand[8].x = 0.9
        update_drawing(self.state, Gesture.DRAW, hand, self.cfg.drawing)
        np.testing.assert_array_equal(before, self.state.canvases[0])

    def test_unicode_slides_and_natural_order(self):
        """Завантажувати Unicode-шляхи та ставити slide2 перед slide10."""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'Слайди'
            folder.mkdir()
            for name, level in [('slide10.png', 100), ('slide2.png', 20)]:
                image = np.full((8, 8, 3), level, dtype=np.uint8)
                cv2.imencode('.png', image)[1].tofile(str(folder / name))
            slides = load_slides(folder, self.cfg.paths.extensions)
            self.assertEqual([int(s[0, 0, 0]) for s in slides], [20, 100])

    def test_geometry_and_clipping(self):
        """Вироджений кут безпечний, координати не виходять за полотно."""
        self.assertEqual(distance((0, 0), (3, 4)), 5)
        self.assertTrue(straight((0, 0), (0, 1), (0, 2), -0.75))
        self.assertFalse(straight((0, 0), (0, 0), (0, 2), -0.75))
        self.assertEqual(landmark_point([SimpleNamespace(x=2, y=-1)], [0], 100, 80), (99, 0))

    def test_invalid_configuration(self):
        """Відхиляти помилки до створення вікон та камери."""
        for config in [
            replace(self.cfg, camera=replace(self.cfg.camera, median_kernel=4)),
            replace(self.cfg, gestures=replace(self.cfg.gestures, hold_seconds=0)),
            replace(self.cfg, drawing=replace(self.cfg.drawing, colors=[[0, 0, 0]], color_names=['BLACK'])),
            replace(self.cfg, gestures=replace(self.cfg.gestures, finger_tips=[8, 12, 16, 30])),
        ]:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    validate(config)

    def test_yaml_relative_path_types_and_unknown_keys(self):
        """Відносний шлях належить YAML, невідомі ключі не ігноруються."""
        data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.yaml'
            path.write_text(yaml.safe_dump(data), encoding='utf-8')
            self.assertEqual(load_config(path).slides_dir, Path(directory) / 'slides')

            data['camera']['median_kernel'] = '5'
            path.write_text(yaml.safe_dump(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_config(path)

            data['camera']['median_kernel'] = 5
            data['camera']['typo'] = 1
            path.write_text(yaml.safe_dump(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_config(path)

    def test_cli_overrides_yaml(self):
        """Явні аргументи мають пріоритет над YAML, включно з камерою 0."""
        args = parse_args(['--camera', '0', '--slides', '.', '--swap-hands'])
        cfg, folder = configure(args)
        self.assertEqual(cfg.camera.index, 0)
        self.assertTrue(cfg.camera.swap_hands)
        self.assertEqual(folder, Path.cwd())
        cfg, _ = configure(parse_args(['--no-swap-hands']))
        self.assertFalse(cfg.camera.swap_hands)

    def test_frame_pipeline_without_hand(self):
        """Пройти реальний шлях обробки кадру з порожнім результатом моделі."""
        detector = SimpleNamespace(process=lambda image: SimpleNamespace(multi_hand_landmarks=None))
        gate = GestureGate(1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        screen, preview = process_frame(frame, detector, gate, self.state, self.cfg)
        self.assertEqual(preview.shape, frame.shape)
        self.assertEqual(screen.shape, self.state.slides[0].shape)
        self.assertFalse(self.state.started)

    def test_all_gestures_from_landmarks(self):
        """Синтетичні точки перевіряють усі шість гілок класифікатора."""
        cases = [
            (Gesture.START, [True, True, True, True], True),
            (Gesture.NEXT, [True, False, False, False], False),
            (Gesture.PREVIOUS, [False, False, False, False], True),
            (Gesture.DRAW, [True, True, False, False], False),
            (Gesture.COLOR, [True, True, False, False], True),
            (Gesture.ERASE, [False, False, False, False], False),
        ]
        for expected, raised, thumb in cases:
            with self.subTest(gesture=expected):
                points = [(50, 100)] * 21
                for base, x, is_up in zip((5, 9, 13, 17), (40, 50, 60, 70), raised):
                    points[base] = (x, 60)
                    points[base + 1] = (x, 40)
                    points[base + 2] = (x, 30 if is_up else 55)
                    points[base + 3] = (x, 20 if is_up else 70)
                points[2:5] = [(40, 75), (25, 75), (10, 75)] if thumb else [(40, 75), (35, 70), (40, 75)]
                hand = [SimpleNamespace(x=x / 100, y=y / 100) for x, y in points]
                self.assertIs(classify(hand, 100, 100, self.cfg.gestures), expected)


if __name__ == '__main__':
    unittest.main()
