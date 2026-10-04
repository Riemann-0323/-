"""每期都会用到的画面组件：场景基类、文字动画、通用夜空、片头片尾、合成器（转场 + 字幕 + 角标）。"""
import bisect
from functools import lru_cache

import cv2
import numpy as np

from .core import (FPS, GREY, H, ORANGE, TEAL, W, WHITE, back_out, canvas, circle, draw_text, ease_in_out,
                   ease_out, line, over, polyline, ramp, round_rect_rgba, smooth, text_width, to_u8, vgradient,
                   vignette_mask)
from .cosmos import StarField, affine, star_point

SKY_TOP, SKY_BOTTOM = (0.006, 0.010, 0.030), (0.045, 0.060, 0.120)
SIL = (0.010, 0.013, 0.025)

# 动画节奏（整体偏快、偏紧凑）
POP_DUR = 0.3
FADE_DUR = 0.35


# ---------- 通用夜空 ----------

@lru_cache(maxsize=1)
def night_gradient():
    return vgradient(SKY_TOP, SKY_BOTTOM, curve=1.6)


@lru_cache(maxsize=1)
def big_sky():
    return StarField(W + 600, H + 400, n=6500, seed=101, bright_n=90)


def warp(img, M, size=(W, H)):
    return cv2.warpAffine(img, np.float32(M), size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def sky(t, zoom=1.0, rot=0.0, focus=(W / 2, H / 2), pan=(0, 0), gain=1.0, grad=True):
    """通用夜空：渐变 + 可缩放/旋转的星场。focus 为画面上保持不动的点。"""
    src = (focus[0] + 300 + pan[0], focus[1] + 200 + pan[1])
    M = affine(zoom, rot, center=focus, src_center=src)
    img = night_gradient().copy() if grad else canvas((0, 0, 0))
    img += big_sky().render(t, M) * gain
    return img


# ---------- 文字动画 ----------

def pop(t, t0, dur=POP_DUR):
    """弹出动画：返回 (缩放, 透明度)。"""
    x = ramp(t, t0, t0 + dur)
    return (0.6 + 0.4 * back_out(x)) if x < 1 else 1.0, smooth(ramp(t, t0, t0 + dur * 0.6))


def title_text(img, text, x, y, size, t, t0, color=WHITE, name="title", fade_out=None, shadow=0.6, **kw):
    s, a = pop(t, t0)
    if fade_out is not None:
        a *= 1 - smooth(ramp(t, fade_out, fade_out + 0.3))
    draw_text(img, text, x, y, size, name, color, alpha=a, scale=s, shadow=shadow, **kw)


def fade_text(img, text, x, y, size, t, t0, color=WHITE, name="bold", dur=FADE_DUR, rise=16, alpha=1.0, **kw):
    a = smooth(ramp(t, t0, t0 + dur)) * alpha
    dy = (1 - ease_out(ramp(t, t0, t0 + dur))) * rise
    draw_text(img, text, x, y + dy, size, name, color, alpha=a, shadow=0.6, **kw)


def pill(img, text, x, y, size, fg, bg, alpha=1.0, name="bold", pad=(22, 10)):
    if alpha <= 0:
        return
    w = int(text_width(text, size, name) + pad[0] * 2)
    h = int(size * 1.25 + pad[1] * 2)
    over(img, round_rect_rgba(w, h, h // 2, bg, 0.92), x - w / 2, y - h / 2, alpha)
    draw_text(img, text, x, y + 1, size, name, fg, alpha=alpha)


def count(t, t0, dur, value):
    return value * ease_out(ramp(t, t0, t0 + dur))


# ---------- 场景基类 ----------

class Scene:
    cut = False
    tag = True

    def __init__(self, span):
        self.span = span
        self.d = span.end - span.start
        self.L = span.lines

    def at(self, k):
        return self.L[min(k, len(self.L) - 1)][0]

    def end(self, k):
        return self.L[min(k, len(self.L) - 1)][1]

    def sfx(self):
        return []

    def draw(self, t):
        raise NotImplementedError


# ---------- 片头片尾 ----------

def logo(img, cx, cy, t, t0=0.0, scale=1.0):
    p = ease_in_out(ramp(t, t0, t0 + 0.6))
    r = 64 * scale
    if p <= 0:
        return
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi * p, 80)]
    polyline(img, pts, TEAL, 5 * scale)
    c, s = np.cos(-0.35), np.sin(-0.35)
    rot = []
    for a in np.linspace(0, 2 * np.pi * p, 90):
        dx, dy = r * 1.65 * np.cos(a), r * 0.55 * np.sin(a)
        rot.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    polyline(img, rot, (0.85, 0.9, 1.0), 2.5 * scale, alpha=0.8)
    ang = t * 1.6
    dx, dy = r * 1.65 * np.cos(ang), r * 0.55 * np.sin(ang)
    star_point(img, cx + dx * c - dy * s, cy + dx * s + dy * c, 3 * scale, ORANGE, k=p)
    circle(img, cx, cy, r * 0.42, TEAL, alpha=0.9 * p)


def make_title(label, title):
    """片头：账号名 + 本期期号和标题。label 如"第一季 · 反直觉 #01"。"""

    class Title(Scene):
        tag = False

        def draw(self, t):
            img = sky(t, gain=0.55, pan=(t * 8, 0)) * 0.8
            logo(img, W / 2, 250, t, 0.0)
            fade_text(img, "反直觉研究所", W / 2, 420, 118, t, 0.15, name="title")
            fade_text(img, "你的直觉，可能是错的", W / 2, 520, 38, t, 0.4, color=GREY, name="medium")
            tc = self.at(1) if len(self.L) > 1 else max(0.7, self.end(0) - 1.0)
            a = smooth(ramp(t, tc - 0.1, tc + 0.25))
            line(img, (W / 2 - 300 * a, 600), (W / 2 + 300 * a, 600), (0.28, 0.31, 0.38), 2)
            pill(img, label, W / 2, 670, 30, (0.02, 0.05, 0.08), TEAL, alpha=a)
            fade_text(img, title, W / 2, 775, 74, t, tc + 0.1, color=WHITE, name="title")
            return img

        def sfx(self):
            tc = self.at(1) if len(self.L) > 1 else max(0.7, self.end(0) - 1.0)
            return [(0.0, "whoosh", 0.6), (0.6, "ding", 0.45), (tc + 0.1, "low_ding", 0.55)]

    return Title


def make_outro(bye="我们下期见"):
    """片尾：标志 + 口号 + 告别语。"""

    class Outro(Scene):
        tag = False

        def draw(self, t):
            img = sky(t, gain=0.5) * 0.8
            logo(img, W / 2, 280, t, 0.0)
            fade_text(img, "反直觉研究所", W / 2, 450, 110, t, 0.15, name="title")
            fade_text(img, "你的直觉，可能是错的", W / 2, 560, 48, t, self.at(0), color=TEAL, name="title")
            fade_text(img, bye, W / 2, 660, 40, t, self.at(1), color=GREY, name="medium")
            img *= 1 - smooth(ramp(t, self.d - 0.6, self.d))
            return img

        def sfx(self):
            return [(0.05, "whoosh", 0.45), (self.at(1) + 0.2, "shimmer", 0.55)]

    return Outro


Outro = make_outro()


# ---------- 合成：场景 + 转场 + 字幕 + 期号角标 ----------

class Compositor:
    XF = 0.2

    def __init__(self, spans, cues, total, classes, tag):
        self.spans = spans
        self.starts = [s.start for s in spans]
        self.cues = cues
        self.cue_starts = [c[0] for c in cues]
        self.total = total
        self.classes = classes
        self.tag = tag
        self._objs = {}

    def obj(self, k):
        if k not in self._objs:
            sp = self.spans[k]
            self._objs[k] = self.classes[sp.name](sp)
        return self._objs[k]

    def scene_objects(self):
        return [self.obj(k) for k in range(len(self.spans))]

    def _draw(self, k, T):
        return self.obj(k).draw(max(0.0, T - self.spans[k].start))

    def frame(self, i):
        T = i / FPS
        k = max(0, bisect.bisect_right(self.starts, T) - 1)
        img = self._draw(k, T)
        if k + 1 < len(self.spans) and not self.obj(k + 1).cut:
            b = self.starts[k + 1]
            if T > b - self.XF:
                w = smooth((T - (b - self.XF)) / (2 * self.XF))
                img = img * (1 - w) + self._draw(k + 1, T) * w
        if k > 0 and not self.obj(k).cut:
            b = self.starts[k]
            if T < b + self.XF:
                w = smooth((T - (b - self.XF)) / (2 * self.XF))
                img = self._draw(k - 1, T) * (1 - w) + img * w
        img = img * vignette_mask(0.28)
        self.overlay(img, T, k)
        return to_u8(img)

    def overlay(self, img, T, k):
        if self.obj(k).tag:
            a = smooth(ramp(T - self.spans[k].start, 0, 0.3)) if k > 0 and not self.obj(k - 1).tag else 1.0
            circle(img, 58, 56, 7, TEAL, alpha=0.9 * a)
            draw_text(img, self.tag, 76, 56, 32, "title", WHITE, alpha=0.85 * a, anchor="lm", shadow=0.6)
        j = bisect.bisect_right(self.cue_starts, T) - 1
        if j >= 0:
            a, b, text = self.cues[j]
            if a <= T < b:
                al = smooth(ramp(T, a, a + 0.06)) * (1 - smooth(ramp(T, b - 0.06, b)))
                draw_text(img, text, W / 2, H - 92, 50, "bold", WHITE, alpha=al, stroke=5,
                          stroke_color=(0.02, 0.02, 0.04), shadow=0.5)
