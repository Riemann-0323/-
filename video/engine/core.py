"""画面基础设施：画布、缓动、文字、合成、辉光。

所有画面都用 float32 的 RGB numpy 数组（0~1）表示，最后再转成 8 位输出。
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30

ASSETS = Path(__file__).resolve().parents[1] / ".assets"
FONTS = {
    "title": ASSETS / "fonts/SmileySans-Oblique.otf",
    "heavy": ASSETS / "fonts/SourceHanSansSC-Heavy.otf",
    "bold": ASSETS / "fonts/SourceHanSansSC-Bold.otf",
    "medium": ASSETS / "fonts/SourceHanSansSC-Medium.otf",
    "regular": ASSETS / "fonts/SourceHanSansSC-Regular.otf",
}


def rgb(hexstr):
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


BG = rgb("#0B1020")
TEAL = rgb("#3DF5D6")
ORANGE = rgb("#FF8A3D")
WHITE = (1.0, 1.0, 1.0)
GREY = (0.55, 0.58, 0.66)
DIM = (0.28, 0.31, 0.38)


# ---------- 缓动与时间 ----------

def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


def ramp(t, a, b):
    if b <= a:
        return float(t >= a)
    return clamp01((t - a) / (b - a))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def back_out(x, s=1.6):
    x = clamp01(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def window(t, a, b, fi=0.4, fo=0.4):
    """在 [a, b] 区间内显示，前后各有淡入淡出。"""
    return min(smooth(ramp(t, a, a + fi)), 1 - smooth(ramp(t, b - fo, b)))


def lerp(a, b, x):
    return a + (b - a) * x


# ---------- 画布 ----------

def canvas(color=BG, h=H, w=W):
    img = np.empty((h, w, 3), np.float32)
    img[:] = color
    return img


def vgradient(top, bottom, h=H, w=W, curve=1.0):
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None] ** curve
    top, bottom = np.array(top, np.float32), np.array(bottom, np.float32)
    col = top[None, :] * (1 - y) + bottom[None, :] * y
    return np.repeat(col[:, None, :], w, axis=1)


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def _clip_box(dst, src, x, y):
    """计算 src 贴到 dst 的 (x, y) 时的重叠区域。"""
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return None
    return (slice(y0, y1), slice(x0, x1)), (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))


def add(dst, src, x=0, y=0, k=1.0):
    """加法混合（发光物体）。"""
    box = _clip_box(dst, src, int(round(x)), int(round(y)))
    if box is None or k <= 0:
        return
    d, s = box
    dst[d] += src[s] * k


def over(dst, rgba, x=0, y=0, alpha=1.0):
    """普通透明度混合。rgba 是 (h, w, 4) float32，颜色未预乘。"""
    if alpha <= 0:
        return
    box = _clip_box(dst, rgba, int(round(x)), int(round(y)))
    if box is None:
        return
    d, s = box
    a = rgba[s][..., 3:4] * alpha
    dst[d] = dst[d] * (1 - a) + rgba[s][..., :3] * a


def darken(dst, mask, k=1.0):
    """用 0~1 的遮罩压暗（mask 越大越暗）。"""
    dst *= 1 - np.clip(mask * k, 0, 1)[..., None]


def glow(img, sigma=12, k=0.6, scale=4):
    """在缩小的图上模糊再放大，叠加回原图，得到柔和的辉光。"""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / scale)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img + big * k


def blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def vignette(img, strength=0.35):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    cx, cy = img.shape[1] / 2, img.shape[0] / 2
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / 1.4142
    img *= (1 - strength * r ** 2)[..., None]


@lru_cache(maxsize=4)
def vignette_mask(strength=0.35, h=H, w=W):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 1.4142
    return (1 - strength * r ** 2)[..., None].astype(np.float32)


# ---------- 形状（在 8 位遮罩上抗锯齿绘制，再按颜色合成） ----------

def _composite_mask(img, mask, x0, y0, color, alpha):
    h, w = mask.shape
    box = _clip_box(img, mask, x0, y0)
    if box is None:
        return
    d, s = box
    a = mask[s].astype(np.float32)[..., None] * (alpha / 255)
    img[d] = img[d] * (1 - a) + np.array(color, np.float32) * a


def _draw_local(img, pts, pad, color, alpha, fn):
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    x0 = int(np.floor(pts[:, 0].min() - pad)); y0 = int(np.floor(pts[:, 1].min() - pad))
    x1 = int(np.ceil(pts[:, 0].max() + pad)); y1 = int(np.ceil(pts[:, 1].max() + pad))
    x0, y0 = max(x0, -pad), max(y0, -pad)
    x1, y1 = min(x1, img.shape[1] + pad), min(y1, img.shape[0] + pad)
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    local = ((pts - [x0, y0]) * 4).astype(np.int32)
    fn(mask, local)
    _composite_mask(img, mask, x0, y0, color, alpha)


def circle(img, cx, cy, r, color, thickness=-1, alpha=1.0):
    if r <= 0 or alpha <= 0:
        return
    th = -1 if thickness < 0 else max(1, int(round(thickness)))
    _draw_local(img, [(cx - r, cy - r), (cx + r, cy + r)], abs(th) + 3, color, alpha,
                lambda m, p: cv2.circle(m, tuple(((p[0] + p[1]) // 2).tolist()), int(r * 4), 255, th, cv2.LINE_AA, 2))


def line(img, p0, p1, color, thickness=2, alpha=1.0):
    th = max(1, int(round(thickness)))
    _draw_local(img, [p0, p1], th + 3, color, alpha,
                lambda m, p: cv2.line(m, tuple(p[0].tolist()), tuple(p[1].tolist()), 255, th, cv2.LINE_AA, 2))


def polyline(img, pts, color, thickness=2, closed=False, alpha=1.0):
    th = max(1, int(round(thickness)))
    _draw_local(img, pts, th + 3, color, alpha,
                lambda m, p: cv2.polylines(m, [p.reshape(-1, 1, 2)], closed, 255, th, cv2.LINE_AA, 2))


def fill_poly(img, pts, color, alpha=1.0):
    _draw_local(img, pts, 3, color, alpha,
                lambda m, p: cv2.fillPoly(m, [p.reshape(-1, 1, 2)], 255, cv2.LINE_AA, 2))


def ellipse(img, cx, cy, rx, ry, color, angle=0, thickness=-1, alpha=1.0):
    th = -1 if thickness < 0 else max(1, int(round(thickness)))
    R = max(rx, ry)
    _draw_local(img, [(cx - R, cy - R), (cx + R, cy + R)], abs(th) + 3, color, alpha,
                lambda m, p: cv2.ellipse(m, tuple(((p[0] + p[1]) // 2).tolist()), (int(rx * 4), int(ry * 4)),
                                         angle, 0, 360, 255, th, cv2.LINE_AA, 2))


def rect(img, x0, y0, x1, y1, color, alpha=1.0):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x0 >= x1 or y0 >= y1:
        return
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - alpha) + np.array(color, np.float32) * alpha


def round_rect_rgba(w, h, r, color, alpha=1.0):
    im = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), r * 2, fill=255)
    a = np.asarray(im.resize((w, h), Image.LANCZOS), np.float32)[..., None] / 255 * alpha
    out = np.empty((h, w, 4), np.float32)
    out[..., :3] = color
    out[..., 3:] = a
    return out


# ---------- 文字 ----------

@lru_cache(maxsize=64)
def font(name, size):
    return ImageFont.truetype(str(FONTS[name]), size)


@lru_cache(maxsize=1024)
def text_rgba(text, size, name="bold", color=WHITE, stroke=0, stroke_color=(0, 0, 0), spacing=0):
    """把一行文字渲染成 (h, w, 4) float32 数组。spacing 为字间距（像素）。"""
    f = font(name, size)
    pad = stroke + 4
    if spacing:
        widths = [f.getlength(ch) for ch in text]
        tw = int(sum(widths) + spacing * (len(text) - 1))
    else:
        tw = int(f.getlength(text))
    asc, desc = f.getmetrics()
    th = asc + desc
    im = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    fill = tuple(int(c * 255) for c in color) + (255,)
    sc = tuple(int(c * 255) for c in stroke_color) + (255,)
    if spacing:
        x = pad
        for ch, cw in zip(text, widths):
            d.text((x, pad), ch, font=f, fill=fill, stroke_width=stroke, stroke_fill=sc)
            x += cw + spacing
    else:
        d.text((pad, pad), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=sc)
    arr = np.asarray(im, np.float32) / 255
    arr.setflags(write=False)
    return arr


def draw_text(img, text, x, y, size, name="bold", color=WHITE, alpha=1.0, anchor="mm",
              stroke=0, stroke_color=(0, 0, 0), scale=1.0, spacing=0, shadow=0.0):
    """在 (x, y) 按 anchor 对齐绘制文字。anchor: l/m/r + t/m/b。"""
    if alpha <= 0.002 or not text:
        return
    t = text_rgba(text, size, name, tuple(color), stroke, tuple(stroke_color), spacing)
    if scale != 1.0:
        t = cv2.resize(t, (max(1, int(t.shape[1] * scale)), max(1, int(t.shape[0] * scale))),
                       interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    h, w = t.shape[:2]
    ox = {"l": 0, "m": w / 2, "r": w}[anchor[0]]
    oy = {"t": 0, "m": h / 2, "b": h}[anchor[1]]
    if shadow > 0:
        sh = np.zeros_like(t)
        sh[..., 3] = cv2.GaussianBlur(t[..., 3], (0, 0), max(2, size * 0.12)) * shadow
        over(img, sh, x - ox, y - oy + size * 0.04, alpha)
    over(img, t, x - ox, y - oy, alpha)
    return w, h


def text_width(text, size, name="bold", spacing=0):
    f = font(name, size)
    return f.getlength(text) + spacing * max(0, len(text) - 1)


# ---------- 噪声 ----------

def fbm(h, w, seed=0, octaves=6, base=3, persistence=0.55, tile_x=False):
    """分形值噪声，返回 0~1 的 float32 数组。"""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    aspect = w / h
    for o in range(octaves):
        gh = base * 2 ** o + 1
        gw = int(base * aspect * 2 ** o) + 1
        g = rng.random((gh, gw)).astype(np.float32)
        if tile_x:
            g[:, -1] = g[:, 0]
        acc += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
    acc /= total
    lo, hi = np.percentile(acc, 1), np.percentile(acc, 99)
    return np.clip((acc - lo) / (hi - lo), 0, 1)
