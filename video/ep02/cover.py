"""第 2 期封面：16:9（B站/抖音）与 3:4（小红书）。用法：python3 -m ep02.cover"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core import ORANGE, TEAL, WHITE, draw_text, over, rect, round_rect_rgba, to_u8, vgradient  # noqa: E402
from engine.cosmos import person_silhouette, rooftops  # noqa: E402
from engine.props import bench, bus, bus_stop, clock  # noqa: E402

OUT = ROOT / "out" / "ep02"
SIL = (0.02, 0.025, 0.04)


def stamp(img, text, cx, cy, size):
    w, h = int(size * 5.0), int(size * 1.45)
    over(img, round_rect_rgba(w, h, int(h * 0.22), ORANGE), cx - w / 2, cy - h / 2)
    over(img, round_rect_rgba(w - 12, h - 12, int(h * 0.2), (0.05, 0.04, 0.06)), cx - (w - 12) / 2, cy - (h - 12) / 2, 0.9)
    draw_text(img, text, cx, cy - size * 0.04, size, "title", ORANGE)


def landscape():
    w, h = 1920, 1080
    img = vgradient((0.16, 0.26, 0.56), (0.92, 0.62, 0.44), h=h, w=w, curve=1.2)
    rooftops(img, 900, seed=3, color=(0.10, 0.12, 0.20))
    rect(img, 0, 900, w, h, (0.05, 0.06, 0.09))
    bus_stop(img, 1180, 900, 380)
    bench(img, 1420, 900)
    person_silhouette(img, 1300, 900, 240, color=SIL, look_up=0.0)
    bus(img, 1900, 900, 420)
    clock(img, 1640, 230, 120, 49)
    draw_text(img, "反直觉 #02", 110, 120, 48, "title", TEAL, anchor="lm", shadow=0.8)
    draw_text(img, "10 分钟一班", 110, 330, 150, "title", WHITE, anchor="lm", shadow=0.9)
    draw_text(img, "为什么总等更久？", 110, 520, 120, "title", TEAL, anchor="lm", shadow=0.9)
    stamp(img, "不是运气差", 420, 720, 84)
    return img


def portrait():
    w, h = 1080, 1440
    img = vgradient((0.16, 0.26, 0.56), (0.92, 0.62, 0.44), h=h, w=w, curve=1.2)
    rect(img, 0, 1250, w, h, (0.05, 0.06, 0.09))
    bus_stop(img, 260, 1250, 380)
    person_silhouette(img, 420, 1250, 240, color=SIL, look_up=0.0)
    bus(img, 1080, 1250, 460)
    clock(img, 820, 300, 130, 49)
    draw_text(img, "反直觉 #02", w / 2, 90, 44, "title", TEAL, shadow=0.8)
    draw_text(img, "公交 10 分钟一班", w / 2, 560, 104, "title", WHITE, shadow=0.9)
    draw_text(img, "你却总在等", w / 2, 710, 140, "title", TEAL, shadow=0.9)
    stamp(img, "不是运气差", w / 2, 900, 80)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, img in (("封面-16x9.jpg", landscape()), ("封面-3x4.jpg", portrait())):
        cv2.imwrite(str(OUT / name), cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("封面已保存：", OUT / name)


if __name__ == "__main__":
    main()
