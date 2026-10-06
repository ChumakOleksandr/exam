import unittest
import numpy as np
from gestures import GestureGate
from main import composite, stroke, load_slides, ROOT


class AppTests(unittest.TestCase):
    def test_requires_more_than_second_and_fires_once(self):
        gate = GestureGate()
        self.assertEqual(gate.update('next', 0), (None, None))
        self.assertEqual(gate.update('next', 1), (None, None))
        self.assertEqual(gate.update('next', 1.01), ('next', 'next'))
        self.assertEqual(gate.update('next', 5), (None, 'next'))
        gate.update(None, 6)
        gate.update('next', 7)
        self.assertEqual(gate.update('next', 8.1), ('next', 'next'))

    def test_noise_and_loss_reset_timer(self):
        gate = GestureGate()
        gate.update('draw', 0)
        gate.update('color', 0.9)
        self.assertEqual(gate.update('draw', 1.1), (None, None))
        self.assertEqual(gate.update('draw', 2), (None, None))
        self.assertEqual(gate.update('draw', 2.2), ('draw', 'draw'))
        self.assertEqual(gate.update(None, 2.3), (None, None))

    def test_stroke_mask_and_eraser_restore_original(self):
        slide = np.full((80, 100, 3), 127, dtype=np.uint8)
        canvas = np.zeros_like(slide)
        stroke(canvas, (70, 40), (10, 40), (0, 0, 255), 3)
        result = composite(slide, canvas)
        np.testing.assert_array_equal(result[40, 40], [0, 0, 255])
        np.testing.assert_array_equal(result[0, 0], slide[0, 0])
        stroke(canvas, (45, 40), (35, 40), (0, 0, 0), 20)
        np.testing.assert_array_equal(composite(slide, canvas)[40, 40], slide[40, 40])
        self.assertTrue(np.all(slide == 127))

    def test_new_stroke_does_not_connect_to_old_position(self):
        canvas = np.zeros((80, 100, 3), dtype=np.uint8)
        stroke(canvas, (10, 10), None, (0, 0, 255), 4)
        self.assertFalse(canvas.any())
        stroke(canvas, (20, 10), (10, 10), (0, 0, 255), 4)
        stroke(canvas, (70, 60), None, (0, 0, 255), 4)
        stroke(canvas, (80, 60), (70, 60), (0, 0, 255), 4)
        self.assertFalse(canvas[30:50].any())
        self.assertTrue(canvas[10, 15].any())
        self.assertTrue(canvas[60, 75].any())

    def test_original_slides_load(self):
        slides = load_slides(ROOT / 'slides')
        self.assertEqual(len(slides), 7)
        self.assertTrue(all(s.ndim == 3 and s.size > 0 for s in slides))


if __name__ == '__main__':
    unittest.main()
