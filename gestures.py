"""Розпізнавання жестів і підтвердження за монотонним часом."""
import math


class GestureGate:
    def __init__(self, hold=1.0):
        self.hold = hold
        self.gesture = None
        self.since = 0.0
        self.fired = False

    def update(self, gesture, now):
        if gesture != self.gesture:
            self.gesture, self.since, self.fired = gesture, now, False
        ready = gesture is not None and now - self.since > self.hold
        event = gesture if ready and not self.fired else None
        if ready:
            self.fired = True
        return event, gesture if ready else None


def classify(lm, width, height):
    # Піксельні координати зберігають кути за будь-якого формату камери.
    p = [(v.x * width, v.y * height) for v in lm]
    def distance(a, b):
        return math.dist(p[a], p[b])
    def straight(a, b, c):
        u = (p[a][0]-p[b][0], p[a][1]-p[b][1])
        v = (p[c][0]-p[b][0], p[c][1]-p[b][1])
        norm = math.hypot(*u) * math.hypot(*v)
        return norm > 0 and (u[0]*v[0]+u[1]*v[1])/norm < -0.75
    scale = max(distance(0, 9), 1)
    up = [straight(t-3, t-2, t) and p[t][1] < p[t-2][1]-0.1*scale
          for t in (8, 12, 16, 20)]
    thumb = straight(2, 3, 4) and distance(4, 5) > 0.45*scale
    sideways = thumb and abs(p[4][0]-p[2][0]) > abs(p[4][1]-p[2][1])
    folded = [distance(t, 0) < distance(t-2, 0)+0.15*scale
              for t in (8, 12, 16, 20)]
    if all(up) and thumb:
        return 'start'
    if up[:2] == [True, True] and all(folded[2:]):
        return 'color' if sideways else ('draw' if not thumb else None)
    if up[0] and all(folded[1:]) and not thumb:
        return 'next'
    if all(folded) and sideways:
        return 'previous'
    if all(folded) and not thumb:
        return 'erase'
    return None
