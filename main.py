"""Керування презентацією правою рукою: OpenCV + MediaPipe Hands."""
import argparse
from pathlib import Path
import re
import time
import cv2
import mediapipe as mp
import numpy as np
from gestures import GestureGate, classify

ROOT = Path(__file__).resolve().parent
SLIDES_DIR = Path(r'A:\exam\slides')
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
NAMES = ['RED', 'GREEN', 'BLUE']


def load_slides(folder):
    if not folder.is_dir():
        raise ValueError(f'Slides folder not found: {folder}')
    def key(path):
        return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', path.name)]
    paths = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}), key=key)
    slides = []
    for path in paths:
        slide = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if slide is None:
            raise ValueError(f'Cannot read slide: {path}')
        slides.append(slide)
    if not slides:
        raise ValueError(f'No slides in {folder}. Add PNG, JPG or JPEG images to this folder.')
    return slides


def composite(slide, canvas):
    result = slide.copy()
    mask = np.any(canvas != 0, axis=2)
    result[mask] = canvas[mask]
    return result


def stroke(canvas, point, previous, color, thickness):
    if previous is not None:
        cv2.line(canvas, point, previous, color, thickness)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--slides', type=Path, default=SLIDES_DIR)
    parser.add_argument('--swap-hands', action='store_true', help='For cameras that already mirror frames')
    args = parser.parse_args()
    slides = load_slides(args.slides)
    # Кожен слайд має власне чорне полотно його оригінального розміру.
    canvases = [np.zeros_like(s) for s in slides]
    index, color_index, started = 0, 0, False
    gate, previous, previous_mode = GestureGate(), None, None
    mp_hands = mp.solutions.hands
    capture = cv2.VideoCapture(args.camera)
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open camera. Check access or try --camera 1')
        cv2.namedWindow('Presentation', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        # Подвоєний початковий розмір вікон OpenCV: 400x300 -> 800x600.
        cv2.resizeWindow('Presentation', 800, 600)
        cv2.resizeWindow('Camera', 800, 600)
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('Camera stopped delivering frames')
                # Віддзеркалення та очищення шумів перед детекцією.
                frame = cv2.flip(frame, 1)
                frame = cv2.medianBlur(frame, 5)
                now = time.monotonic()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)
                hand = None
                target_label = 'Left' if args.swap_hands else 'Right'
                if result.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                            result.multi_hand_landmarks, result.multi_handedness or []):
                        label = handedness.classification[0]
                        if label.label == target_label and label.score >= 0.7:
                            hand = hand_landmarks.landmark
                            break
                fh, fw = frame.shape[:2]
                gesture = classify(hand, fw, fh) if hand else None
                event, active = gate.update(gesture, now)
                if event == 'start':
                    started = True
                    for canvas in canvases:
                        canvas.fill(0)
                elif started:
                    if event == 'next':
                        index = min(index + 1, len(slides) - 1)
                    elif event == 'previous':
                        index = max(index - 1, 0)
                    elif event == 'color':
                        color_index = (color_index + 1) % len(COLORS)
                mode = active if started and active in ('draw', 'erase') else None
                point = None
                if mode and hand:
                    h, w = canvases[index].shape[:2]
                    ids = (8,) if mode == 'draw' else (0, 5, 9, 13, 17)
                    x = sum(hand[i].x for i in ids) / len(ids)
                    y = sum(hand[i].y for i in ids) / len(ids)
                    point = (int(np.clip(x, 0, 1)*(w-1)), int(np.clip(y, 0, 1)*(h-1)))
                    if previous_mode != mode:
                        previous = None
                    # Гумка записує чорний колір: маска відкриває початковий слайд.
                    stroke(canvases[index], point, previous,
                           COLORS[color_index] if mode == 'draw' else (0, 0, 0),
                           12 if mode == 'draw' else 60)
                previous, previous_mode = point, mode
                if started:
                    screen = composite(slides[index], canvases[index])
                    if point:
                        cv2.circle(screen, point, 8 if mode == 'draw' else 30, (160, 160, 160), 1)
                else:
                    screen = np.zeros_like(slides[0])
                    cv2.putText(screen, 'Hold an open RIGHT palm > 1 sec to start', (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if hand:
                    for lm in hand:
                        cv2.circle(frame, (int(lm.x*fw), int(lm.y*fh)), 3, (0, 255, 255), -1)
                progress = min(1.0, (now-gate.since)/gate.hold) if gesture else 0
                lines = [f'RIGHT hand | Gesture: {gesture or "none"} | {progress:.0%}',
                         f'Slide {index+1}/{len(slides)} | Color: {NAMES[color_index]}',
                         'Palm: start/clear | Index: next | Thumb: previous',
                         'V: draw | V+thumb: color | Fist: erase | Esc: exit']
                for i, text in enumerate(lines):
                    cv2.putText(frame, text, (10, 25+25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
                    cv2.putText(frame, text, (10, 25+25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                cv2.imshow('Presentation', screen)
                cv2.imshow('Camera', frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in ('Presentation', 'Camera')):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(f'Error: {exc}')
