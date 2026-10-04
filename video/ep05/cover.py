"""第 5 期封面：16:9（B站/抖音）与 3:4（小红书）。用法：python3 -m ep05.cover"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core import ORANGE, TEAL, WHITE, draw_text, fill_poly, over, round_rect_rgba, to_u8  # noqa: E402
from engine.cosmos import person_silhouette  # noqa: E402
from ep05.scenes import night_scene  # noqa: E402

OUT = ROOT / "out" / "ep05"


def stamp(img, text, cx, cy, size):
    w, h = int(size * 4.4), int(size * 1.45)
    over(img, round_rect_rgba(w, h, int(h * 0.22), ORANGE), cx - w / 2, cy - h / 2)
    over(img, round_rect_rgba(w - 12, h - 12, int(h * 0.2), (0.05, 0.04, 0.06)), cx - (w - 12) / 2, cy - (h - 12) / 2, 0.9)
    draw_text(img, text, cx, cy - size * 0.04, size, "title", ORANGE)


def landscape():
    base, sf, fg = night_scene()
    img = base + sf.render(0.0)
    over(img, fg, 0, 0)
    draw_text(img, "反直觉 #05 · 季终", 110, 120, 48, "title", TEAL, anchor="lm", shadow=0.8)
    draw_text(img, "夜空", 110, 330, 170, "title", WHITE, anchor="lm", shadow=0.9)
    draw_text(img, "为什么是黑的？", 110, 520, 140, "title", TEAL, anchor="lm", shadow=0.9)
    stamp(img, "难倒几百年", 400, 740, 76)
    return img


def portrait():
    w, h = 1080, 1440
    base, sf, _ = night_scene()
    img = cv2.resize(base + sf.render(0.0), (int(1920 * h / 1080), h))[:, 600:600 + w].copy()
    mask = np.zeros((h, w, 3), np.float32)
    xs = np.linspace(-10, w + 10, 120)
    ys = 1260 - 40 * np.sin(xs / w * np.pi * 1.4 + 0.6) - 14 * np.sin(xs / w * np.pi * 4.1 + 2.0)
    fill_poly(mask, list(zip(xs, ys)) + [(w + 10, h + 10), (-10, h + 10)], (1, 1, 1))
    person_silhouette(mask, 760, np.interp(760, xs, ys) + 5, 200, color=(1, 1, 1), look_up=0.6)
    img = img * (1 - mask) + mask * np.array((0.010, 0.013, 0.025))
    draw_text(img, "反直觉 #05 · 季终", w / 2, 90, 44, "title", TEAL, shadow=0.8)
    draw_text(img, "夜空", w / 2, 300, 160, "title", WHITE, shadow=0.9)
    draw_text(img, "为什么是黑的？", w / 2, 470, 120, "title", TEAL, shadow=0.9)
    stamp(img, "难倒几百年", w / 2, 650, 72)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, img in (("封面-16x9.jpg", landscape()), ("封面-3x4.jpg", portrait())):
        cv2.imwrite(str(OUT / name), cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("封面已保存：", OUT / name)


if __name__ == "__main__":
    main()
