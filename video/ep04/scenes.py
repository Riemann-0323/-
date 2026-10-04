"""第 4 期场景：此刻地球上，一定有一个地方没有风。"""
from functools import lru_cache

import numpy as np

from engine.core import (GREY, ORANGE, TEAL, W, WHITE, circle, draw_text, ease_in_out, fbm, line, over, ramp,
                         round_rect_rgba, smooth)
from engine.cosmos import earth_texture, halo, sphere
from engine.fields import calm_marker, draw_sphere_field, project, rot_x, rot_y, swirl
from engine.torus import cached as torus
from engine.ui import Outro, Scene, fade_text, make_title, pill, pop, sky, title_text

EARTH_C = (W / 2 - 220, 540)
CYCLONE = np.array([0.25, 0.35, 1.0]) / np.linalg.norm([0.25, 0.35, 1.0])


def wind(p):
    """示意风场：两个气旋 + 一股西风带。"""
    return (swirl(p, CYCLONE, 1.3, 0.75) + swirl(p, [-0.8, -0.3, 0.4], -1.0, 0.7)
            + swirl(p, [0.6, -0.6, -0.5], 0.8, 0.8) + np.cross([0, 1, 0], p) * 0.35)


def globe(img, t, cx, cy, R, spin=0.18, arrows=True, alpha=1.0, calm=True, n=760):
    R3 = rot_x(0.3) @ rot_y(t * spin)
    halo(img, cx, cy, R * 1.25, (0.3, 0.55, 1.0), 0.45 * alpha, power=1.5)
    over(img, sphere(2 * R, earth_texture(), -t * spin + 1.0, light=(-0.5, 0.3, 0.8)), cx - R, cy - R, alpha)
    if arrows:
        draw_sphere_field(img, cx, cy, R, wind, R3, n=n, color=(0.96, 0.98, 1.0), phase=t, alpha=alpha)
    x, y, z = project(CYCLONE, R3, cx, cy, R)
    if calm and z > 0.1:
        calm_marker(img, x, y, t, ORANGE, alpha=alpha)
    return (x, y, z), R3


# ---------- 1. 钩子 ----------

class Hook(Scene):
    tag = False

    def draw(self, t):
        img = sky(t, gain=0.7)
        (x, y, z), _ = globe(img, t, *EARTH_C, 330)
        fade_text(img, "此刻，地球上", 1420, 360, 76, t, self.at(0) + 0.1, name="title")
        fade_text(img, "一定有一个地方没有风", 1420, 460, 64, t, self.at(0) + 0.8, color=TEAL, name="title")
        s, a = pop(t, self.at(1) + 0.3)
        if a > 0:
            w, h = int(400 * s), int(120 * s)
            over(img, round_rect_rgba(w, h, 22, ORANGE), 1420 - w / 2, 640 - h / 2, a)
            over(img, round_rect_rgba(w - 12, h - 12, 18, (0.05, 0.04, 0.06)), 1420 - (w - 12) / 2, 640 - (h - 12) / 2, a * 0.9)
            draw_text(img, "数学证明", 1420, 636, 76, "title", ORANGE, alpha=a, scale=s)
        return img

    def sfx(self):
        return [(0.1, "whoosh", 0.5), (self.at(1) + 0.3, "swish", 0.9), (self.at(1) + 0.35, "low_ding", 0.7)]


# ---------- 3. 梳毛球 ----------

@lru_cache(maxsize=1)
def fur_texture():
    n = fbm(128, 256, 9, 5, base=8, tile_x=True)
    return (np.array((0.16, 0.42, 0.48)) * (0.85 + 0.3 * n[..., None])).astype(np.float32)


MESS = [(np.array(c) / np.linalg.norm(c), k) for c, k in
        [([0.3, 0.8, 0.5], 1.2), ([-0.7, 0.1, 0.7], -1.0), ([0.8, -0.3, 0.5], 0.9), ([-0.2, -0.8, 0.6], -1.1),
         ([0.1, 0.2, -1.0], 1.0), ([-0.6, 0.6, -0.4], 0.8)]]


class Comb(Scene):
    def draw(self, t):
        img = sky(t, gain=0.5) * 0.8
        cx, cy, R = W / 2 - 200, 560, 300
        R3 = rot_x(1.1) @ rot_y(0.4 + t * 0.12)
        over(img, sphere(2 * R, fur_texture(), -(0.4 + t * 0.12), light=(-0.5, 0.4, 0.75)), cx - R, cy - R)
        q = ease_in_out(ramp(t, self.at(0) + 1.2, self.at(1) + 0.6))

        def field(p):
            mess = sum(swirl(p, c, k, 0.9) for c, k in MESS) + np.cross([0.3, 0.2, 0.9], p) * 0.6
            comb = np.cross([0, 1, 0], p)
            return mess * (1 - q) + comb * q * 1.6
        draw_sphere_field(img, cx, cy, R, field, R3, n=1100, length=0.12, color=(0.86, 0.96, 0.95),
                          thickness=2.6, head=False)
        a = smooth(ramp(t, self.at(2) + 0.6, self.at(2) + 1.0))
        x, y, z = project([0, 1, 0], R3, cx, cy, R)
        if a > 0 and z > 0:
            circle(img, x, y, 34 + 4 * np.sin(t * 5), ORANGE, thickness=4, alpha=a)
            line(img, (x + 30, y - 24), (x + 160, y - 120), ORANGE, 3, alpha=a)
            draw_text(img, "总会留下一个旋", x + 170, y - 130, 40, "title", ORANGE, alpha=a, anchor="lm", shadow=0.7)
        fade_text(img, "能把毛球梳平吗？", 1420, 300, 64, t, 0.3, name="title")
        title_text(img, "不可能", 1420, 420, 110, t, self.at(1) + 0.2, color=ORANGE)
        fade_text(img, "毛球定理 · 庞加莱 · 1885", 1420, 800, 40, t, self.at(3) + 0.5, color=GREY, name="medium")
        return img

    def sfx(self):
        return [(self.at(1) + 0.2, "low_ding", 0.7), (self.at(2) + 0.6, "ding", 0.5)]


# ---------- 4. 风就是毛 ----------

class Wind(Scene):
    def draw(self, t):
        img = sky(t, gain=0.7)
        (x, y, z), _ = globe(img, t, *EARTH_C, 340, spin=0.04, calm=t > self.at(3))
        fade_text(img, "风 = 球面上的一根根毛", 1430, 300, 56, t, self.at(1) + 0.2, name="title")
        fade_text(img, "不可能处处都在吹", 1430, 390, 56, t, self.at(2) + 0.5, color=TEAL, name="title")
        a = smooth(ramp(t, self.at(3), self.at(3) + 0.4))
        if a > 0 and z > 0:
            line(img, (x + 28, y + 12), (1255, 600), ORANGE, 3, alpha=a)
            draw_text(img, "水平风速 = 0", 1270, 600, 50, "title", ORANGE, alpha=a, anchor="lm", shadow=0.8)
        fade_text(img, "常在气旋中心，或几股风互相抵消处", 1430, 800, 34, t, self.at(4) + 0.4, color=GREY, name="medium")
        return img


# ---------- 5. 甜甜圈 ----------

TOR_W, TOR_H, TOR_S = 1040, 660, 340


class Donut(Scene):
    def draw(self, t):
        img = sky(t, gain=0.5) * 0.85
        T = torus(TOR_W, TOR_H, TOR_S)
        ox, oy = W / 2 - TOR_W / 2 - 180, 210
        over(img, T.image(), ox, oy)
        a = smooth(ramp(t, self.at(1) - 0.2, self.at(1) + 0.3))
        T.arrows(img, ox, oy, t, alpha=a)
        fade_text(img, "如果地球是甜甜圈", 1500, 330, 60, t, 0.3, name="title")
        title_text(img, "可以处处梳平", 1500, 450, 72, t, self.at(1) + 0.3, color=TEAL)
        return img


class Tokamak(Scene):
    def draw(self, t):
        img = sky(t, gain=0.4) * 0.7
        T = torus(TOR_W, TOR_H, TOR_S)
        ox, oy = W / 2 - TOR_W / 2 - 180, 210
        tor = T.image(base=(0.10, 0.12, 0.2))
        over(img, tor, ox, oy, 0.9)
        halo(img, ox + TOR_W / 2, oy + TOR_H / 2, 520, (0.9, 0.35, 0.9), 0.25 + 0.05 * np.sin(t * 3))
        p = ease_in_out(ramp(t, 0.4, self.at(1) + 1.0))
        for k in range(6):
            T.helix(img, ox, oy, q=3.0, phase=k * np.pi / 3 + t * 0.4, color=(1.0, 0.62, 0.3), thickness=3,
                    alpha=0.9, upto=p)
        fade_text(img, "托卡马克（核聚变装置）", 1500, 330, 52, t, 0.3, name="title")
        fade_text(img, "磁场贴着环面绕圈", 1500, 430, 48, t, self.at(1), color=ORANGE, name="title")
        fade_text(img, "球面：注定有零点", 1500, 560, 40, t, self.at(1) + 1.6, color=GREY, name="bold")
        fade_text(img, "环面：可以处处不为零", 1500, 620, 40, t, self.at(1) + 2.4, color=TEAL, name="bold")
        return img


# ---------- 6. 对跖点 ----------

class Antipodal(Scene):
    P1 = np.array([-0.82, 0.42, 0.38]) / np.linalg.norm([-0.82, 0.42, 0.38])

    def draw(self, t):
        img = sky(t, gain=0.7)
        cx, cy, R = W / 2 - 160, 560, 320
        R3 = rot_x(0.25) @ rot_y(0.15 + t * 0.05)
        halo(img, cx, cy, R * 1.25, (0.3, 0.55, 1.0), 0.45, power=1.5)
        over(img, sphere(2 * R, earth_texture(), -(0.15 + t * 0.05) + 2.0, light=(-0.5, 0.3, 0.8)), cx - R, cy - R)
        x1, y1, _ = project(self.P1, R3, cx, cy, R)
        x2, y2, _ = project(-self.P1, R3, cx, cy, R)
        a = smooth(ramp(t, self.at(1), self.at(1) + 0.5))
        if a > 0:
            p = ease_in_out(ramp(t, self.at(1) + 0.2, self.at(1) + 1.2))
            n = 24
            for k in range(n):
                if k % 2 == 0 and k / n < p:
                    f0, f1 = k / n, (k + 1) / n
                    line(img, (x1 + (x2 - x1) * f0, y1 + (y2 - y1) * f0), (x1 + (x2 - x1) * f1, y1 + (y2 - y1) * f1),
                         WHITE, 3, alpha=a * 0.8)
            circle(img, x1, y1, 12, ORANGE, alpha=a)
            circle(img, x2, y2, 12, ORANGE, alpha=a * 0.6 * p)
            pill(img, "18.6 ℃ · 1013 hPa", x1 - 40, y1 - 70, 30, WHITE, (0.05, 0.06, 0.1), alpha=a)
            pill(img, "18.6 ℃ · 1013 hPa", x2 + 120, y2 + 70, 30, WHITE, (0.05, 0.06, 0.1), alpha=a * p)
            draw_text(img, "（地球另一端）", x2 + 120, y2 + 125, 28, "medium", GREY, alpha=a * p)
        fade_text(img, "博苏克–乌拉姆定理", 1500, 300, 56, t, self.at(2), color=TEAL, name="title")
        fade_text(img, "温度、气压完全相同（示意数值）", 1500, 380, 34, t, self.at(2) + 0.6, color=GREY, name="medium")
        return img

    def sfx(self):
        return [(self.at(1) + 0.2, "whoosh", 0.4), (self.at(2), "ding", 0.5)]


# ---------- 7. 前提 / 8. 结论 / 9. 提问 ----------

class Caveat(Scene):
    def draw(self, t):
        img = sky(t, gain=0.4) * 0.75
        pill(img, "前提", W / 2, 330, 40, (0.1, 0.04, 0.0), ORANGE, alpha=smooth(ramp(t, 0.1, 0.4)))
        fade_text(img, "风、温度、气压连续变化", W / 2, 470, 72, t, 0.4, name="title")
        fade_text(img, "真实大气没那么理想，但足够接近", W / 2, 580, 48, t, self.at(0) + 2.6, color=GREY, name="medium")
        return img


class Verdict(Scene):
    def draw(self, t):
        img = sky(t, gain=0.7)
        globe(img, t, W / 2, 620, 300, spin=0.04, n=600)
        title_text(img, "总有一个地方", W / 2, 150, 76, t, 0.3)
        title_text(img, "正安安静静", W / 2, 260, 76, t, 1.8, color=TEAL)
        return img


@lru_cache(maxsize=1)
def donut_planet():
    def tex(u, v):
        n = fbm(TOR_H, TOR_W, 4, 5, base=4)
        land = np.clip((n - 0.55) * 8, 0, 1)[..., None]
        return np.array((0.05, 0.2, 0.5)) * (1 - land) + np.array((0.25, 0.45, 0.2)) * land
    return torus(TOR_W, TOR_H, TOR_S).image(tex=tex)


class Question(Scene):
    def draw(self, t):
        img = sky(t, gain=0.6)
        T = torus(TOR_W, TOR_H, TOR_S)
        ox, oy = W / 2 - TOR_W / 2, 330
        over(img, donut_planet(), ox, oy)
        T.arrows(img, ox, oy, t, n_u=28, n_v=8, alpha=0.7)
        fade_text(img, "甜甜圈星球的天气会是什么样？", W / 2, 170, 64, t, 0.2, name="title")
        fade_text(img, "评论区脑洞一下 ↓", W / 2, 260, 46, t, 1.4, color=TEAL, name="title")
        return img


# ---------- 10. 下期预告 ----------

class Next(Scene):
    def draw(self, t):
        img = sky(t, gain=0.9) * (1 - 0.3 * smooth(ramp(t, 0, 2)))
        pill(img, "下期预告 · 季终", W / 2, 330, 34, (0.1, 0.04, 0.0), ORANGE, alpha=smooth(ramp(t, 0.05, 0.35)))
        title_text(img, "夜空为什么是黑的？", W / 2, 480, 96, t, 0.4)
        fade_text(img, "难倒了天文学家几百年", W / 2, 600, 48, t, 2.0, color=GREY, name="medium")
        return img

    def sfx(self):
        return [(0.4, "ding", 0.5)]


SCENE_CLASSES = {"hook": Hook, "title": make_title("第一季 · 反直觉 #04", "此刻地球上，一定有一个地方没有风"),
                 "comb": Comb, "wind": Wind, "donut": Donut, "tokamak": Tokamak, "antipodal": Antipodal,
                 "caveat": Caveat, "verdict": Verdict, "question": Question, "next": Next, "outro": Outro}


def music(spans, total):
    sp = {s.name: s for s in spans}
    return [(0, sp["donut"].start, "calm"), (sp["donut"].start - 0.5, total, "wonder")], []
