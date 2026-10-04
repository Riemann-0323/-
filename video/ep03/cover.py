"""EP03 封面：16:9（B站/抖音）与 3:4（小红书）。

用法（在 video/ 目录下）：python3 -m ep03.cover
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core import ORANGE, TEAL, WHITE, draw_text, over, round_rect_rgba, to_u8, vgradient  # noqa: E402
from engine.cosmos import StarField, hill_silhouette, milky_way, person_silhouette, star_point  # noqa: E402

OUT = ROOT / "out" / "ep03"


def night(w, h, seed=5, angle=-28):
    mw, dens = milky_way(w, h, seed=seed, angle=angle, strength=0.55)
    img = vgradient((0.006, 0.010, 0.030), (0.06, 0.07, 0.14), h=h, w=w, curve=1.6) + mw
    img += StarField(w, h, n=int(w * h / 420), seed=1, density=dens, bright_n=60).render(0.0, out=(w, h))
    return img


def stamp(img, text, cx, cy, size):
    w, h = int(size * 5.6), int(size * 1.45)
    over(img, round_rect_rgba(w, h, int(h * 0.22), ORANGE), cx - w / 2, cy - h / 2)
    over(img, round_rect_rgba(w - 12, h - 12, int(h * 0.2), (0.03, 0.03, 0.06)), cx - (w - 12) / 2, cy - (h - 12) / 2, 0.9)
    draw_text(img, text, cx, cy - size * 0.04, size, "title", ORANGE)


def landscape():
    w, h = 1920, 1080
    img = night(w, h)
    xs, ys = hill_silhouette(img, 960, 50, seed=2, color=(0.01, 0.013, 0.025))
    person_silhouette(img, 1560, np.interp(1560, xs, ys) + 5, 170, color=(0.01, 0.013, 0.025))
    star_point(img, 1480, 300, 12, (1.0, 0.45, 0.18), k=1.6, spikes=0.7, spike_len=200)
    draw_text(img, "反直觉 #03", 110, 120, 48, "title", TEAL, anchor="lm", shadow=0.8)
    draw_text(img, "你看到的星星", 110, 330, 150, "title", WHITE, anchor="lm", shadow=0.9)
    draw_text(img, "还在吗？", 110, 520, 190, "title", TEAL, anchor="lm", shadow=0.9)
    stamp(img, "只对了一半", 470, 760, 92)
    return img


def portrait():
    w, h = 1080, 1440
    img = night(w, h, seed=8, angle=-62)
    pts = np.linspace(-10, w + 10, 80)
    ys = 1300 - 30 * np.sin(pts / w * 3.1 + 0.5)
    mask = np.zeros((h, w), np.uint8)
    poly = np.array(list(zip(pts, ys)) + [(w + 10, h + 10), (-10, h + 10)], np.int32)
    cv2.fillPoly(mask, [poly], 255, cv2.LINE_AA)
    img[mask > 0] = (0.01, 0.013, 0.025)
    person_silhouette(img, 760, float(np.interp(760, pts, ys)) + 5, 150, color=(0.01, 0.013, 0.025))
    star_point(img, 820, 420, 12, (1.0, 0.45, 0.18), k=1.6, spikes=0.7, spike_len=180)
    draw_text(img, "反直觉 #03", w / 2, 110, 44, "title", TEAL, shadow=0.8)
    draw_text(img, "你看到的星星", w / 2, 560, 128, "title", WHITE, shadow=0.9)
    draw_text(img, "还在吗？", w / 2, 730, 170, "title", TEAL, shadow=0.9)
    stamp(img, "只对了一半", w / 2, 960, 84)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, img in (("封面-16x9.jpg", landscape()), ("封面-3x4.jpg", portrait())):
        cv2.imwrite(str(OUT / name), cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("封面已保存：", OUT / name)


if __name__ == "__main__":
    main()
