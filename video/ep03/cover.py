"""第 3 期封面：16:9（B站/抖音）与 3:4（小红书）。用法：python3 -m ep03.cover"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core import ORANGE, TEAL, WHITE, draw_text, line, over, polyline, round_rect_rgba, to_u8, vgradient  # noqa: E402
from engine.cosmos import StarField, halo, sphere, star_texture  # noqa: E402
from ep03.scenes import barycenter_path  # noqa: E402

OUT = ROOT / "out" / "ep03"


def base(w, h):
    img = vgradient((0.006, 0.010, 0.030), (0.045, 0.060, 0.120), h=h, w=w, curve=1.6)
    img += StarField(w, h, n=int(w * h / 500), seed=3, bright_n=50).render(0.0, out=(w, h))
    return img


def sun_and_path(img, cx, cy, R):
    ts, xy = barycenter_path()
    pts = [(cx + x * R, cy - y * R) for x, y in xy[:1100]]
    halo(img, cx, cy, R * 4.2, (1.0, 0.72, 0.32), 0.6)
    over(img, sphere(int(2 * R), star_texture("sun"), 0.3, emissive=True, limb=0.45), cx - R, cy - R)
    polyline(img, pts, ORANGE, 4, alpha=0.9)
    bx, by = pts[-1]
    for s in (1, -1):
        line(img, (bx - 22, by - 22 * s), (bx + 22, by + 22 * s), ORANGE, 7)


def stamp(img, text, cx, cy, size):
    w, h = int(size * 5.4), int(size * 1.45)
    over(img, round_rect_rgba(w, h, int(h * 0.22), ORANGE), cx - w / 2, cy - h / 2)
    over(img, round_rect_rgba(w - 12, h - 12, int(h * 0.2), (0.05, 0.04, 0.06)), cx - (w - 12) / 2, cy - (h - 12) / 2, 0.9)
    draw_text(img, text, cx, cy - size * 0.04, size, "title", ORANGE)


def landscape():
    w, h = 1920, 1080
    img = base(w, h)
    sun_and_path(img, 1420, 560, 190)
    draw_text(img, "反直觉 #03", 110, 120, 48, "title", TEAL, anchor="lm", shadow=0.8)
    draw_text(img, "太阳系的中心", 110, 330, 140, "title", WHITE, anchor="lm", shadow=0.9)
    draw_text(img, "不在太阳里", 110, 520, 170, "title", TEAL, anchor="lm", shadow=0.9)
    stamp(img, "太阳也在绕圈", 470, 740, 76)
    return img


def portrait():
    w, h = 1080, 1440
    img = base(w, h)
    sun_and_path(img, w / 2, 1040, 160)
    draw_text(img, "反直觉 #03", w / 2, 90, 44, "title", TEAL, shadow=0.8)
    draw_text(img, "太阳系的中心", w / 2, 300, 120, "title", WHITE, shadow=0.9)
    draw_text(img, "不在太阳里", w / 2, 460, 150, "title", TEAL, shadow=0.9)
    stamp(img, "太阳也在绕圈", w / 2, 640, 72)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, img in (("封面-16x9.jpg", landscape()), ("封面-3x4.jpg", portrait())):
        cv2.imwrite(str(OUT / name), cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("封面已保存：", OUT / name)


if __name__ == "__main__":
    main()
