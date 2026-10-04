"""第 4 期封面：16:9（B站/抖音）与 3:4（小红书）。用法：python3 -m ep04.cover"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core import ORANGE, TEAL, WHITE, draw_text, over, round_rect_rgba, to_u8, vgradient  # noqa: E402
from engine.cosmos import StarField  # noqa: E402
from ep04.scenes import globe  # noqa: E402

OUT = ROOT / "out" / "ep04"


def base(w, h):
    img = vgradient((0.006, 0.010, 0.030), (0.045, 0.060, 0.120), h=h, w=w, curve=1.6)
    img += StarField(w, h, n=int(w * h / 500), seed=4, bright_n=50).render(0.0, out=(w, h))
    return img


def stamp(img, text, cx, cy, size):
    w, h = int(size * 4.4), int(size * 1.45)
    over(img, round_rect_rgba(w, h, int(h * 0.22), ORANGE), cx - w / 2, cy - h / 2)
    over(img, round_rect_rgba(w - 12, h - 12, int(h * 0.2), (0.05, 0.04, 0.06)), cx - (w - 12) / 2, cy - (h - 12) / 2, 0.9)
    draw_text(img, text, cx, cy - size * 0.04, size, "title", ORANGE)


def landscape():
    w, h = 1920, 1080
    img = base(w, h)
    globe(img, 0.6, 1400, 560, 360, spin=0.04, n=900)
    draw_text(img, "反直觉 #04", 110, 120, 48, "title", TEAL, anchor="lm", shadow=0.8)
    draw_text(img, "此刻地球上", 110, 330, 140, "title", WHITE, anchor="lm", shadow=0.9)
    draw_text(img, "一定有地方没风", 110, 520, 130, "title", TEAL, anchor="lm", shadow=0.9)
    stamp(img, "数学证明", 400, 740, 80)
    return img


def portrait():
    w, h = 1080, 1440
    img = base(w, h)
    globe(img, 0.6, w / 2, 1030, 300, spin=0.04, n=760)
    draw_text(img, "反直觉 #04", w / 2, 90, 44, "title", TEAL, shadow=0.8)
    draw_text(img, "此刻地球上", w / 2, 290, 120, "title", WHITE, shadow=0.9)
    draw_text(img, "一定有地方没风", w / 2, 450, 112, "title", TEAL, shadow=0.9)
    stamp(img, "数学证明", w / 2, 630, 72)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, img in (("封面-16x9.jpg", landscape()), ("封面-3x4.jpg", portrait())):
        cv2.imwrite(str(OUT / name), cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("封面已保存：", OUT / name)


if __name__ == "__main__":
    main()
