"""日常物件的简洁剪影/图标：公交车、站牌、长椅、时钟。"""
import numpy as np

from .core import circle, fill_poly, line, rect, round_rect_rgba, over

BUS_YELLOW = (0.98, 0.74, 0.18)
BUS_TEAL = (0.24, 0.96, 0.84)


def bus(img, x, ground_y, w=420, color=BUS_TEAL, alpha=1.0, window=(0.06, 0.09, 0.16), lights=True):
    """侧视公交车，(x, ground_y) 为车头底部的位置，车头朝右。"""
    h = w * 0.36
    body = round_rect_rgba(int(w), int(h), int(h * 0.16), color, 1.0)
    over(img, body, x - w, ground_y - h - w * 0.05, alpha)
    top = ground_y - h - w * 0.05
    n = 6
    for i in range(n):
        wx0 = x - w + w * 0.05 + i * w * 0.135
        rect(img, wx0, top + h * 0.14, wx0 + w * 0.11, top + h * 0.5, window, alpha)
    rect(img, x - w * 0.11, top + h * 0.14, x - w * 0.03, top + h * 0.85, window, alpha)   # 车门
    if lights:
        circle(img, x - w * 0.012, top + h * 0.78, w * 0.014, (1.0, 0.95, 0.7), alpha=alpha)
    for wx in (x - w * 0.8, x - w * 0.22):
        circle(img, wx, ground_y - w * 0.045, w * 0.07, (0.05, 0.05, 0.07), alpha=alpha)
        circle(img, wx, ground_y - w * 0.045, w * 0.03, (0.55, 0.58, 0.64), alpha=alpha)


def bus_icon(img, cx, cy, s=40, color=BUS_TEAL, alpha=1.0):
    """正面小图标，用于时间轴上标记到站。"""
    over(img, round_rect_rgba(int(s), int(s * 1.1), int(s * 0.2), color, 1.0), cx - s / 2, cy - s * 0.55, alpha)
    rect(img, cx - s * 0.36, cy - s * 0.42, cx + s * 0.36, cy - s * 0.02, (0.06, 0.09, 0.16), alpha)
    for dx in (-0.28, 0.28):
        circle(img, cx + dx * s, cy + s * 0.32, s * 0.08, (1, 0.95, 0.75), alpha=alpha)


def bus_stop(img, x, ground_y, h=360, color=(0.02, 0.025, 0.04), sign=BUS_TEAL, alpha=1.0):
    rect(img, x - 5, ground_y - h, x + 5, ground_y, color, alpha)
    sw, sh = h * 0.36, h * 0.24
    over(img, round_rect_rgba(int(sw), int(sh), 10, sign, 1.0), x - sw / 2, ground_y - h - sh * 0.4, alpha)
    bus_icon(img, x, ground_y - h + sh * 0.12, s=sh * 0.55, color=(0.03, 0.04, 0.07), alpha=alpha)


def bench(img, x, ground_y, w=220, color=(0.02, 0.025, 0.04), alpha=1.0):
    rect(img, x - w / 2, ground_y - 60, x + w / 2, ground_y - 48, color, alpha)
    rect(img, x - w / 2, ground_y - 100, x + w / 2, ground_y - 88, color, alpha)
    for dx in (-0.4, 0.4):
        rect(img, x + dx * w - 5, ground_y - 100, x + dx * w + 5, ground_y, color, alpha)


def clock(img, cx, cy, r, minutes, color=(0.95, 0.96, 1.0), hand=(1.0, 0.54, 0.24), alpha=1.0, max_minutes=60):
    """表盘 + 一根表示已等待时间的指针，以及扫过的扇形。"""
    circle(img, cx, cy, r, (0.03, 0.04, 0.07), alpha=0.9 * alpha)
    circle(img, cx, cy, r, color, thickness=4, alpha=alpha)
    for k in range(12):
        a = k / 12 * 2 * np.pi
        line(img, (cx + np.sin(a) * r * 0.82, cy - np.cos(a) * r * 0.82),
             (cx + np.sin(a) * r * 0.92, cy - np.cos(a) * r * 0.92), color, 3, alpha=alpha)
    frac = (minutes % max_minutes) / max_minutes
    if frac > 0:
        pts = [(cx, cy)] + [(cx + np.sin(a) * r * 0.78, cy - np.cos(a) * r * 0.78)
                            for a in np.linspace(0, frac * 2 * np.pi, 40)]
        fill_poly(img, pts, hand, alpha=0.35 * alpha)
    a = frac * 2 * np.pi
    line(img, (cx, cy), (cx + np.sin(a) * r * 0.75, cy - np.cos(a) * r * 0.75), hand, 6, alpha=alpha)
    circle(img, cx, cy, 7, hand, alpha=alpha)
