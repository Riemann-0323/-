"""第 3 期场景：太阳系的中心，经常不在太阳里。"""
from functools import lru_cache

import numpy as np

from engine.core import (GREY, ORANGE, TEAL, W, WHITE, H, circle, draw_text, ease_in_out, ease_out, fill_poly,
                         lerp, line, over, polyline, ramp, rect, smooth)
from engine.cosmos import (earth_texture, halo, moon_texture, planet_texture, sphere, star_point, star_texture)
from engine.fields import calm_marker, draw_sphere_field, rot_x, rot_y, swirl
from engine.ui import Outro, Scene, fade_text, make_title, pill, sky, title_text

SUN_GLOW = (1.0, 0.72, 0.32)
CROSS_C = ORANGE


def cross(img, x, y, s=16, color=CROSS_C, alpha=1.0, label=None, label_dx=26):
    if alpha <= 0:
        return
    line(img, (x - s, y - s), (x + s, y + s), color, 5, alpha=alpha)
    line(img, (x - s, y + s), (x + s, y - s), color, 5, alpha=alpha)
    if label:
        draw_text(img, label, x + label_dx, y - 30, 34, "bold", color, alpha=alpha, anchor="lm", shadow=0.7)


def sun(img, x, y, d, t, glow=1.0):
    halo(img, x, y, d * 2.2, SUN_GLOW, 0.5 * glow)
    over(img, sphere(int(d), star_texture("sun"), t * 0.05, emissive=True, limb=0.45), x - d / 2, y - d / 2)


def body(img, x, y, d, tex, t, spin=0.2, light=(-0.7, -0.2, 0.68)):
    over(img, sphere(int(d), tex, t * spin, light=light), x - d / 2, y - d / 2)


def orbit(img, cx, cy, r, alpha=0.35, color=(0.55, 0.62, 0.78)):
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 160)]
    polyline(img, pts, color, 2, closed=True, alpha=alpha)


# ---------- 1. 钩子 ----------

class Hook(Scene):
    tag = False
    ORBITS = [(230, 1.6, (0.75, 0.6, 0.5), 10), (330, 1.0, (0.4, 0.6, 1.0), 13), (460, 0.55, (0.95, 0.8, 0.6), 22),
              (600, 0.35, (0.9, 0.85, 0.65), 18)]

    def draw(self, t):
        img = sky(t, gain=0.6)
        B = (W / 2, H / 2 + 20)
        sd = 190
        off = sd / 2 + 26
        k = ease_in_out(ramp(t, self.at(2), self.at(2) + 0.8))
        ang = (t - self.at(2)) * 1.4 if t > self.at(2) else 0.0
        sx, sy = B[0] - off * np.cos(ang) * k - off * (1 - k), B[1] - off * np.sin(ang) * k
        for r, w, c, d in self.ORBITS:
            orbit(img, *B, r)
            a = t * w * 0.5 + r
            circle(img, B[0] + r * np.cos(a), B[1] + r * np.sin(a), d / 2, c)
        sun(img, sx, sy, sd, t)
        fade_text(img, "太阳 = 太阳系的中心？", W / 2, 140, 68, t, self.at(0) + 0.2, name="title")
        c = smooth(ramp(t, self.at(1) + 0.8, self.at(1) + 1.2))
        cross(img, *B, alpha=c, label="真正的中心" if k < 0.3 else None)
        title_text(img, "大部分时间，不在太阳里", W / 2, 960 - 60, 60, t, self.at(1) + 1.4, color=ORANGE)
        return img

    def sfx(self):
        return [(self.at(1) + 0.8, "ding", 0.6), (self.at(2), "whoosh", 0.5)]


# ---------- 3. 跷跷板 ----------

class Seesaw(Scene):
    def draw(self, t):
        img = sky(t, gain=0.4) * 0.7
        cx, cy, L = W / 2, 620, 1000
        m1, m2 = 5.0, 1.0
        fb = 1 / (1 + m1 / m2)                       # 平衡时支点距大人一端的比例
        f = lerp(0.5, fb, ease_in_out(ramp(t, 1.4, 3.4)))
        torque = m1 * f - m2 * (1 - f)
        tilt = np.clip(torque * 0.12, -0.16, 0.16) * (1 - smooth(ramp(t, 3.2, 3.6)))
        fx = cx - L / 2 + f * L
        ca, sa = np.cos(tilt), np.sin(tilt)
        left = (fx - f * L * ca, cy + f * L * sa)
        right = (fx + (1 - f) * L * ca, cy - (1 - f) * L * sa)
        line(img, left, right, (0.8, 0.82, 0.88), 14)
        fill_poly(img, [(fx, cy + 8), (fx - 46, cy + 110), (fx + 46, cy + 110)], ORANGE)
        rect(img, cx - 620, cy + 110, cx + 620, cy + 118, (0.4, 0.43, 0.5))
        circle(img, left[0], left[1] - 98, 90, (0.4, 0.6, 1.0))
        circle(img, right[0], right[1] - 48, 40, (0.95, 0.8, 0.55))
        draw_text(img, "大人", left[0], left[1] - 98, 40, "title", (0.02, 0.04, 0.1))
        draw_text(img, "小孩", right[0], right[1] - 120, 32, "title", WHITE, shadow=0.7)
        a = smooth(ramp(t, self.at(1) + 0.3, self.at(1) + 0.7))
        draw_text(img, "质心", fx, cy + 170, 52, "title", ORANGE, alpha=a, shadow=0.7)
        fade_text(img, "支点要靠近重的一边", W / 2, 200, 60, t, 0.6, name="title")
        return img

    def sfx(self):
        return [(self.at(1) + 0.3, "ding", 0.5)]


# ---------- 4. 地月质心 ----------

class EarthMoon(Scene):
    def draw(self, t):
        img = sky(t, gain=0.6)
        B = (840, 560)
        th = t * 0.55 + 0.4
        ed, md = 320, 86
        eo = ed / 2 * 0.733
        ex, ey = B[0] - eo * np.cos(th), B[1] - eo * np.sin(th)
        mx, my = B[0] + 470 * np.cos(th), B[1] + 470 * np.sin(th)
        orbit(img, *B, 470, alpha=0.3)
        orbit(img, *B, eo, alpha=0.5, color=TEAL)
        halo(img, ex, ey, ed * 0.62, (0.3, 0.55, 1.0), 0.45, power=1.5)
        body(img, ex, ey, ed, earth_texture(), t, 0.15, light=(0.8, -0.3, 0.5))
        body(img, mx, my, md, moon_texture(), t, 0.0, light=(0.8, -0.3, 0.5))
        c = smooth(ramp(t, self.at(1), self.at(1) + 0.4))
        cross(img, *B, s=12, alpha=c)
        fade_text(img, "地月质心", 1460, 330, 60, t, self.at(1), color=ORANGE, name="title", anchor="lm")
        fade_text(img, "离地心约 4700 公里", 1460, 410, 36, t, self.at(1) + 0.6, color=WHITE, name="bold", anchor="lm")
        fade_text(img, "在地表下约 1700 公里", 1460, 465, 36, t, self.at(1) + 1.2, color=WHITE, name="bold", anchor="lm")
        draw_text(img, "示意图，距离未按比例", 60, 900, 26, "regular", GREY, anchor="lm", alpha=0.8)
        return img


# ---------- 5. 日木质心 ----------

class Jupiter(Scene):
    def draw(self, t):
        img = sky(t, gain=0.6)
        B = (760, 560)
        sd = 340
        so = sd / 2 * 1.07
        th = t * 0.4 + 2.6
        sx, sy = B[0] - so * np.cos(th), B[1] - so * np.sin(th)
        jx, jy = B[0] + 540 * np.cos(th), B[1] + 540 * np.sin(th) * 0.8
        orbit(img, *B, 540, alpha=0.25)
        orbit(img, *B, so, alpha=0.6, color=TEAL)
        sun(img, sx, sy, sd, t)
        body(img, jx, jy, 120, planet_texture("jupiter"), t, 0.6, light=(-np.cos(th), np.sin(th), 0.5))
        c = smooth(ramp(t, self.at(1), self.at(1) + 0.4))
        cross(img, *B, s=14, alpha=c)
        if c > 0:
            line(img, (sx, sy), B, WHITE, 2, alpha=c * 0.8)
        fade_text(img, "木星 ≈ 太阳质量的 1/1000", 1300, 160, 44, t, self.at(0) + 0.6, color=WHITE, name="bold")
        fade_text(img, "日木质心：离太阳中心约 74 万公里", 1300, 840, 40, t, self.at(1) + 0.3, color=ORANGE, name="bold")
        fade_text(img, "太阳半径：约 69.6 万公里", 1300, 900, 34, t, self.at(1) + 1.2, color=GREY, name="medium")
        title_text(img, "质心在太阳外面", 1300, 240, 64, t, self.at(2), color=TEAL)
        return img

    def sfx(self):
        return [(self.at(2), "low_ding", 0.6)]


# ---------- 6. 太阳系质心的花边轨迹 ----------

PLANETS = [(9.546e-4, 5.203, 11.86, 34.4), (2.858e-4, 9.537, 29.46, 49.9),
           (4.366e-5, 19.19, 84.01, 313.2), (5.151e-5, 30.07, 164.8, 304.9)]
R_SUN_AU = 695700 / 1.496e8


@lru_cache(maxsize=1)
def barycenter_path(years=50, n=1500):
    ts = np.linspace(0, years, n)
    xy = np.zeros((n, 2))
    for m, a, P, l0 in PLANETS:
        ang = np.radians(l0) + 2 * np.pi * ts / P
        xy += m * a * np.stack([np.cos(ang), np.sin(ang)], 1)
    xy /= 1 + sum(p[0] for p in PLANETS)
    return ts, xy / R_SUN_AU          # 以太阳半径为单位


class Dance(Scene):
    def draw(self, t):
        img = sky(t, gain=0.6)
        C, R = (W / 2 - 160, 540), 165
        sun(img, *C, 2 * R, t, glow=0.8)
        ts, xy = barycenter_path()
        p = ramp(t, 0.6, self.d - 0.6)
        k = max(2, int(len(ts) * p))
        pts = [(C[0] + x * R, C[1] - y * R) for x, y in xy[:k]]
        outside = np.hypot(xy[:k, 0], xy[:k, 1]) > 1
        for i in range(1, k, 2):
            line(img, pts[i - 1], pts[min(i + 1, k - 1)], ORANGE if outside[i] else TEAL, 3, alpha=0.95)
        cross(img, *pts[-1], s=12)
        draw_text(img, f"{2000 + ts[k - 1]:.0f} 年", 1460, 300, 72, "title", WHITE, shadow=0.6)
        frac = outside.mean() if k > 10 else 0
        draw_text(img, f"在太阳外：{frac * 100:.0f}%", 1460, 400, 44, "bold", ORANGE, shadow=0.6)
        draw_text(img, "在太阳内", 1460, 470, 34, "bold", TEAL, shadow=0.6)
        draw_text(img, "太阳系质心相对太阳的位置（示意计算）", 1460, 860, 26, "medium", GREY, shadow=0.6)
        return img


# ---------- 7. 地球跟着太阳晃 ----------

class Earth(Scene):
    def draw(self, t):
        img = sky(t, gain=0.6)
        B = (W / 2, 540)
        wob = 70
        def sun_at(tt):
            a = tt * 0.45
            return B[0] + wob * np.cos(a), B[1] + wob * np.sin(a)
        trail = []
        for tt in np.linspace(max(0, t - 6), t, 120):
            sx, sy = sun_at(tt)
            e = tt * 2.2
            trail.append((sx + 300 * np.cos(e), sy + 300 * np.sin(e) * 0.95))
        if len(trail) > 1:
            polyline(img, trail, (0.4, 0.62, 1.0), 2, alpha=0.6)
        sx, sy = sun_at(t)
        orbit(img, *B, wob, alpha=0.6, color=TEAL)
        sun(img, sx, sy, 150, t)
        ex, ey = trail[-1]
        body(img, ex, ey, 54, earth_texture(), t, 0.3, light=(sx - ex, -(sy - ey), 40))
        cross(img, *B, s=10)
        title_text(img, "地球 → 绕着太阳转", W / 2, 150, 58, t, self.at(1) + 0.5, color=(0.5, 0.75, 1.0))
        title_text(img, "太阳 → 绕着一个空点转", W / 2, 900 - 40, 58, t, self.at(2) + 1.6, color=ORANGE)
        return img


# ---------- 8. 恒星摆动与光谱 ----------

def wl_rgb(nm):
    """可见光波长 → 近似 RGB。"""
    nm = np.asarray(nm, np.float32)
    r = np.where(nm < 440, -(nm - 440) / 60, np.where(nm < 490, 0, np.where(nm < 510, 0, np.where(nm < 580, (nm - 510) / 70, 1))))
    g = np.where(nm < 440, 0, np.where(nm < 490, (nm - 440) / 50, np.where(nm < 580, 1, np.where(nm < 645, -(nm - 645) / 65, 0))))
    b = np.where(nm < 490, 1, np.where(nm < 510, -(nm - 510) / 20, 0))
    f = np.where(nm < 420, 0.3 + 0.7 * (nm - 380) / 40, np.where(nm > 700, 0.3 + 0.7 * (780 - nm) / 80, 1))
    return np.clip(np.stack([r, g, b], -1) * f[..., None], 0, 1)


@lru_cache(maxsize=1)
def spectrum_strip(w=1200, h=70):
    nm = np.linspace(400, 700, w)
    col = wl_rgb(nm)
    return np.repeat(col[None], h, 0).astype(np.float32)


LINES_NM = [434, 486, 527, 589, 656]


class Wobble(Scene):
    def draw(self, t):
        img = sky(t, gain=0.5) * 0.85
        th = t * 1.3
        sx0, sy = 1240, 400
        sx = sx0 + 26 * np.cos(th)
        px, py = sx0 - 260 * np.cos(th), sy + 60 * np.sin(th)
        behind = np.sin(th) < 0
        if behind:
            circle(img, px, py, 12, (0.6, 0.7, 0.9))
        star_point(img, sx, sy, 16, (1.0, 0.9, 0.7), k=1.4)
        if not behind:
            circle(img, px, py, 12, (0.6, 0.7, 0.9))
        fill_poly(img, [(200, sy - 24), (330, sy - 40), (330, sy + 40), (200, sy + 24)], (0.75, 0.78, 0.85))
        draw_text(img, "望远镜", 265, sy + 80, 30, "bold", GREY)
        line(img, (340, sy), (sx - 60, sy), (0.45, 0.5, 0.6), 2, alpha=0.6)
        v = -np.sin(th)            # 朝向观测者为正
        x0, y0 = W / 2 - 600, 680
        strip = spectrum_strip().copy()
        for nm in LINES_NM:
            x = int((nm - 400) / 300 * 1200 - v * 22)
            strip[:, max(0, x - 3):x + 3] *= 0.05
        a = smooth(ramp(t, 0.3, 0.7))
        over(img, np.dstack([strip, np.full(strip.shape[:2], 1.0, np.float32)]), x0, y0, a)
        draw_text(img, "光谱", x0 - 30, y0 + 35, 34, "bold", WHITE, anchor="rm", alpha=a)
        lab, col = ("朝我们 → 偏蓝", (0.45, 0.65, 1.0)) if v > 0.2 else (("远离 → 偏红", (1.0, 0.45, 0.4)) if v < -0.2 else ("", WHITE))
        if lab:
            draw_text(img, lab, W / 2, 820, 52, "title", col, alpha=a * smooth(ramp(t, self.at(1), self.at(1) + 0.4)), shadow=0.6)
        title_text(img, "太阳晃动：约每秒 12–13 米", W / 2, 160, 58, t, self.at(0) + 0.3)
        return img


# ---------- 9. 发现与诺奖 ----------

class Discovery(Scene):
    def draw(self, t):
        img = sky(t, gain=0.5) * 0.85
        sx, sy = 480, 470
        star_point(img, sx, sy, 18, (1.0, 0.9, 0.7), k=1.5)
        th = t * 5.0
        circle(img, sx + 110 * np.cos(th), sy + 34 * np.sin(th), 9, (0.9, 0.6, 0.4))
        draw_text(img, "飞马座 51", sx, sy + 110, 34, "bold", WHITE, shadow=0.6)
        title_text(img, "1995", 1250, 230, 120, t, self.at(0) + 0.4, color=TEAL)
        fade_text(img, "飞马座 51b · 4.2 天绕一圈", 1250, 340, 44, t, self.at(0) + 1.4, name="bold")
        fade_text(img, "2019 诺贝尔物理学奖", 1250, 430, 48, t, self.at(1) + 0.2, color=ORANGE, name="title")
        a = smooth(ramp(t, self.at(2), self.at(2) + 0.4))
        if a > 0:
            x0, x1, y = 860, 1660, 620
            p = ease_out(ramp(t, self.at(2) + 0.2, self.at(2) + 1.2))
            rect(img, x0, y - 26, x0 + (x1 - x0) * p, y + 26, (0.35, 0.4, 0.5), a)
            rect(img, x0, y - 26, x0 + (x1 - x0) * p * 1140 / 6000, y + 26, ORANGE, a)
            draw_text(img, "已确认的系外行星：6000 多颗", x0, y - 60, 34, "bold", WHITE, alpha=a, anchor="lm")
            draw_text(img, "靠测晃动发现：1000 多颗", x0, y + 64, 34, "bold", ORANGE, alpha=a * smooth(ramp(t, self.at(2) + 1.2, self.at(2) + 1.6)), anchor="lm")
        return img

    def sfx(self):
        return [(self.at(0) + 0.4, "ding", 0.5), (self.at(1) + 0.2, "shimmer", 0.5)]


# ---------- 10. 冥王星与卡戎 ----------

class Pluto(Scene):
    def draw(self, t):
        img = sky(t, gain=0.7)
        B = (W / 2 - 120, 540)
        th = t * 0.5
        pd, cd = 230, 116
        po = pd / 2 * 1.79
        pxy = (B[0] - po * np.cos(th), B[1] - po * np.sin(th) * 0.9)
        cxy = (B[0] + 470 * np.cos(th), B[1] + 470 * np.sin(th) * 0.9)
        orbit(img, *B, po, alpha=0.5, color=TEAL)
        light = (-np.cos(th) * 0.3 + 0.6, 0.2, 0.7)
        body(img, *pxy, pd, planet_texture("pluto"), t, 0.15, light=light)
        body(img, *cxy, cd, planet_texture("charon"), t, 0.15, light=light)
        cross(img, *B, s=12, label="质心")
        draw_text(img, "冥王星", pxy[0], pxy[1] + pd / 2 + 36, 34, "bold", WHITE, shadow=0.7)
        draw_text(img, "卡戎", cxy[0], cxy[1] + cd / 2 + 36, 34, "bold", WHITE, shadow=0.7)
        title_text(img, "绕着空地跳双人舞", W / 2, 150, 64, t, self.at(1) + 0.2, color=TEAL)
        return img


# ---------- 11. 结论 / 12. 提问 ----------

class Verdict(Scene):
    def draw(self, t):
        img = sky(t, gain=0.6)
        C, R = (W / 2, 600), 120
        ts, xy = barycenter_path()
        pts = [(C[0] - x * R, C[1] + y * R) for x, y in xy[:900]]
        polyline(img, pts, ORANGE, 2, alpha=0.6)
        i = int((t * 60) % 900)
        sun(img, *pts[i], 2 * R * 0.6, t, glow=0.7)
        cross(img, *C, s=10)
        title_text(img, "没有纹丝不动的中心", W / 2, 180, 72, t, 0.2)
        title_text(img, "连太阳，也在跳舞", W / 2, 290, 72, t, 1.6, color=TEAL)
        return img


class Question(Scene):
    def draw(self, t):
        img = sky(t, gain=0.8)
        fade_text(img, "你还知道哪些绕着空地转的天体？", W / 2, 440, 64, t, 0.1, name="title")
        fade_text(img, "评论区告诉我 ↓", W / 2, 540, 46, t, 1.2, color=TEAL, name="title")
        return img


# ---------- 13. 下期预告 ----------

class Next(Scene):
    def draw(self, t):
        img = sky(t, gain=0.7)
        cx, cy, R = 700, 540, 300
        R3 = rot_x(0.35) @ rot_y(t * 0.25)
        halo(img, cx, cy, R * 1.25, (0.3, 0.55, 1.0), 0.45, power=1.5)
        over(img, sphere(2 * R, earth_texture(), -t * 0.25 + 1.2, light=(-0.5, 0.3, 0.8)), cx - R, cy - R)
        c1 = np.array([0.2, 0.25, 1.0])
        field = lambda p: swirl(p, c1, 1.0, 0.9) + swirl(p, [-0.9, -0.2, 0.3], -0.8, 0.8) + np.cross([0, 1, 0], p) * 0.25
        draw_sphere_field(img, cx, cy, R, field, R3, n=700, color=(0.95, 0.97, 1.0), phase=t)
        q = (c1 / np.linalg.norm(c1)) @ R3.T
        if q[2] > 0:
            calm_marker(img, cx + q[0] * R, cy - q[1] * R, t, ORANGE)
        pill(img, "下期预告", 1380, 330, 34, (0.1, 0.04, 0.0), ORANGE, alpha=smooth(ramp(t, 0.05, 0.35)))
        title_text(img, "此刻地球上", 1380, 450, 80, t, 0.3)
        title_text(img, "一定有个地方没有风", 1380, 560, 72, t, 1.3, color=TEAL)
        return img

    def sfx(self):
        return [(1.3, "ding", 0.5)]


SCENE_CLASSES = {"hook": Hook, "title": make_title("第一季 · 反直觉 #03", "太阳系的中心，经常不在太阳里"),
                 "seesaw": Seesaw, "earthmoon": EarthMoon, "jupiter": Jupiter, "dance": Dance, "earth": Earth,
                 "wobble": Wobble, "discovery": Discovery, "pluto": Pluto, "verdict": Verdict,
                 "question": Question, "next": Next, "outro": Outro}


def music(spans, total):
    sp = {s.name: s for s in spans}
    return [(0, sp["wobble"].start, "calm"), (sp["wobble"].start - 0.5, total, "wonder")], []
