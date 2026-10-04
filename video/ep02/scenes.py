"""第 2 期场景：公交车 10 分钟一班，为什么你总要等更久？"""
from functools import lru_cache

import numpy as np

from engine.core import (GREY, H, ORANGE, TEAL, W, WHITE, canvas, circle, draw_text, ease_in_out, ease_out,
                         fill_poly, lerp, line, over, ramp, rect, round_rect_rgba, smooth, vgradient)
from engine.cosmos import halo, person_silhouette, rooftops, sphere, star_texture
from engine.props import bench, bus, bus_icon, bus_stop, clock
from engine.ui import Outro, Scene, fade_text, make_title, pill, pop, sky, title_text

DUSK_TOP, DUSK_BOTTOM = (0.18, 0.30, 0.62), (0.90, 0.62, 0.46)
GROUND = 900
SIL = (0.02, 0.025, 0.04)
PANEL = (0.035, 0.045, 0.075)
LONG_C, SHORT_C = ORANGE, TEAL


# ---------- 公用：车站街景、时间轴 ----------

@lru_cache(maxsize=1)
def street():
    img = vgradient(DUSK_TOP, DUSK_BOTTOM, curve=1.2)
    rooftops(img, GROUND, seed=3, color=(0.10, 0.12, 0.20))
    rect(img, 0, GROUND, W, H, (0.05, 0.06, 0.09))
    bench(img, 700, GROUND)
    bus_stop(img, 430, GROUND, 380)
    person_silhouette(img, 590, GROUND, 230, color=SIL, look_up=0.0)
    return img


def dark_bg(t):
    img = sky(t, gain=0.25) * 0.6
    img += canvas(PANEL) * 0.4
    return img


class Axis:
    """一条横向时间轴：分钟 → 像素。"""

    def __init__(self, minutes=40, x0=180, x1=1740, y=560):
        self.m, self.x0, self.x1, self.y = minutes, x0, x1, y

    def x(self, m):
        return lerp(self.x0, self.x1, m / self.m)

    def draw(self, img, alpha=1.0, step=10):
        line(img, (self.x0 - 20, self.y), (self.x1 + 20, self.y), (0.55, 0.58, 0.66), 3, alpha=alpha)
        for m in range(0, self.m + 1, step):
            x = self.x(m)
            line(img, (x, self.y), (x, self.y + 14), (0.55, 0.58, 0.66), 2, alpha=alpha)
            draw_text(img, f"{m} 分", x, self.y + 44, 26, "medium", GREY, alpha=alpha)

    def gap(self, img, a, b, color, alpha=1.0, h=70, label=None):
        rect(img, self.x(a), self.y - h, self.x(b), self.y - 4, color, 0.28 * alpha)
        rect(img, self.x(a), self.y - 8, self.x(b), self.y - 4, color, 0.9 * alpha)
        if label:
            draw_text(img, label, (self.x(a) + self.x(b)) / 2, self.y - h - 30, 30, "bold", color, alpha=alpha)

    def buses(self, img, times, t, t0, step=0.15, alpha=1.0):
        for i, m in enumerate(times):
            s, a = pop(t, t0 + i * step)
            if a > 0:
                bus_icon(img, self.x(m), self.y - 120, s=46 * s, alpha=a * alpha)


def you_marker(img, x, y, alpha=1.0, color=WHITE):
    circle(img, x, y - 34, 9, color, alpha=alpha)
    fill_poly(img, [(x - 10, y - 22), (x + 10, y - 22), (x + 7, y), (x - 7, y)], color, alpha=alpha)


# ---------- 1. 钩子 ----------

class Hook(Scene):
    tag = False

    def draw(self, t):
        img = street().copy()
        clock(img, 1640, 250, 110, t * 1.4)
        fade_text(img, "平均 10 分钟一班", 1080, 300, 76, t, self.at(0) + 0.2, name="title")
        fade_text(img, "你要等多久？", 1080, 400, 64, t, self.at(0) + 1.4, color=(0.98, 0.98, 1.0), name="title")
        title_text(img, "5 分钟？", 1080, 540, 120, t, self.at(1), color=TEAL)
        s, a = pop(t, self.at(2) + 0.2)
        if a > 0:
            w, h = int(520 * s), int(130 * s)
            over(img, round_rect_rgba(w, h, 24, ORANGE), 1080 - w / 2, 700 - h / 2, a)
            over(img, round_rect_rgba(w - 12, h - 12, 20, (0.05, 0.04, 0.06)), 1080 - (w - 12) / 2, 700 - (h - 12) / 2, a * 0.9)
            draw_text(img, "通常更久", 1080, 696, 84, "title", ORANGE, alpha=a, scale=s)
        return img

    def sfx(self):
        return [(self.at(1), "ding", 0.5), (self.at(2) + 0.2, "swish", 0.9), (self.at(2) + 0.25, "low_ding", 0.7)]


# ---------- 3. 理想世界：严格 10 分钟一班 ----------

class Ideal(Scene):
    ARRIVALS = [3.0, 17.5, 8.2, 26.0, 33.4, 12.7]

    def draw(self, t):
        img = dark_bg(t)
        ax = Axis(40)
        ax.draw(img)
        ax.buses(img, [0, 10, 20, 30, 40], t, 0.2)
        fade_text(img, "理想：严格每 10 分钟一班", W / 2, 190, 60, t, 0.1, name="title")
        k = int(max(0, t - 1.4) // 0.9)
        if t > 1.4:
            m = self.ARRIVALS[k % len(self.ARRIVALS)]
            nxt = (int(m // 10) + 1) * 10
            p = ease_out(ramp((t - 1.4) % 0.9, 0, 0.3))
            ax.gap(img, m, m + (nxt - m) * p, TEAL, label=f"等 {nxt - m:.1f} 分")
            you_marker(img, ax.x(m), ax.y - 2)
        title_text(img, "平均等 5 分钟", W / 2, 820, 72, t, self.end(1) - 1.0, color=TEAL)
        return img


# ---------- 4. 现实：忽长忽短 ----------

GAPS = [0, 2, 20, 22, 40]


def draw_gaps(img, ax, alpha=1.0, labels=True):
    for a, b in zip(GAPS[:-1], GAPS[1:]):
        long = b - a > 5
        ax.gap(img, a, b, LONG_C if long else SHORT_C, alpha=alpha,
               label=(f"{b - a} 分" if labels else None))


class Reality(Scene):
    def draw(self, t):
        img = dark_bg(t)
        ax = Axis(40)
        ax.draw(img)
        fade_text(img, "现实：间隔忽长忽短", W / 2, 190, 60, t, 0.1, name="title")
        t1 = self.at(1) + 0.3
        a = smooth(ramp(t, t1, t1 + 0.4))
        if t < t1:
            ax.buses(img, [0, 10, 20, 30, 40], t, 0.3)
        else:
            ax.buses(img, GAPS, t, t1 - 0.2, step=0.05)
            draw_gaps(img, ax, alpha=a)
        fade_text(img, "平均间隔，仍然是 10 分钟", W / 2, 820, 46, t, t1 + 1.0, color=GREY, name="medium")
        return img


# ---------- 5. 你会落进哪一段？ ----------

@lru_cache(maxsize=1)
def arrivals(n=160):
    rng = np.random.default_rng(12)
    return rng.uniform(0, 40, n), rng.uniform(0, 1, n)


class Trap(Scene):
    def draw(self, t):
        img = dark_bg(t)
        ax = Axis(40)
        ax.draw(img)
        ax.buses(img, GAPS, t, 0.0, step=0.0)
        draw_gaps(img, ax, alpha=0.7, labels=False)
        fade_text(img, "随机到站的人，落在哪儿？", W / 2, 190, 60, t, 0.1, name="title")
        ms, ys = arrivals()
        p = ramp(t, 0.6, self.at(2) + 1.0)
        n = int(len(ms) * p)
        n_long = 0
        for i in range(n):
            m = ms[i]
            long = (2 <= m < 20) or (22 <= m < 40)
            n_long += long
            yy = ax.y - 160 - ys[i] * 120
            circle(img, ax.x(m), yy, 7, LONG_C if long else SHORT_C, alpha=0.9)
        if n > 10:
            draw_text(img, f"落进长空档 {n_long / n * 100:.0f}%", 640, 700, 50, "title", LONG_C, shadow=0.6)
            draw_text(img, f"落进短空档 {(n - n_long) / n * 100:.0f}%", 1280, 700, 50, "title", SHORT_C, shadow=0.6)
        title_text(img, "九成概率，撞进长空档", W / 2, 830, 64, t, self.at(2) + 0.4, color=ORANGE)
        return img

    def sfx(self):
        return [(self.at(2) + 0.4, "ding", 0.5)]


# ---------- 6. 算一算 ----------

def hbar(img, x, y, length, h, color, alpha=1.0):
    rect(img, x, y - h / 2, x + length, y + h / 2, color, alpha)


class Calc(Scene):
    def draw(self, t):
        img = dark_bg(t)
        unit = 90
        x0 = 520
        a1 = smooth(ramp(t, 0.2, 0.5))
        p1 = ease_out(ramp(t, 0.3, 1.0))
        draw_text(img, "撞进长空档", x0 - 30, 260, 40, "bold", WHITE, alpha=a1, anchor="rm")
        hbar(img, x0, 260, 9 * unit * p1, 44, LONG_C, a1)
        draw_text(img, "平均等 9 分", x0 + 9 * unit * p1 + 20, 260, 36, "bold", LONG_C, alpha=a1, anchor="lm")
        draw_text(img, "撞进短空档", x0 - 30, 350, 40, "bold", WHITE, alpha=a1, anchor="rm")
        hbar(img, x0, 350, 1 * unit * p1, 44, SHORT_C, a1)
        draw_text(img, "平均等 1 分", x0 + unit * p1 + 20, 350, 36, "bold", SHORT_C, alpha=a1, anchor="lm")
        t1 = self.at(1)
        fade_text(img, "90% × 9 分  +  10% × 1 分  =  8.2 分", W / 2, 480, 54, t, t1, name="title")
        a2 = smooth(ramp(t, t1 + 0.9, t1 + 1.2))
        p2 = ease_out(ramp(t, t1 + 1.0, t1 + 1.8))
        draw_text(img, "直觉", x0 - 30, 620, 40, "bold", GREY, alpha=a2, anchor="rm")
        hbar(img, x0, 620, 5 * unit * p2, 50, (0.4, 0.43, 0.5), a2)
        draw_text(img, "5 分", x0 + 5 * unit * p2 + 20, 620, 40, "bold", GREY, alpha=a2, anchor="lm")
        draw_text(img, "实际", x0 - 30, 720, 40, "bold", WHITE, alpha=a2, anchor="rm")
        hbar(img, x0, 720, 8.2 * unit * p2, 50, LONG_C, a2)
        draw_text(img, "8.2 分", x0 + 8.2 * unit * p2 + 20, 720, 40, "bold", LONG_C, alpha=a2, anchor="lm")
        title_text(img, "+64%", 1560, 670, 110, t, t1 + 2.0, color=ORANGE)
        return img

    def sfx(self):
        return [(self.at(1) + 2.0, "low_ding", 0.6)]


# ---------- 7. 一般规律：检验悖论 ----------

@lru_cache(maxsize=1)
def poisson_times():
    rng = np.random.default_rng(5)
    t, out = 0.0, [0.0]
    while t < 40:
        t += rng.exponential(10)
        if t < 40:
            out.append(t)
    return out + [40.0]


class Rule(Scene):
    def draw(self, t):
        img = dark_bg(t)
        title_text(img, "检验悖论：间隔越不均匀，等得越久", W / 2, 170, 58, t, 0.1)
        rows = [("均匀", [0, 10, 20, 30, 40], "5 分", TEAL, 0.5),
                ("忽长忽短", GAPS, "8.2 分", ORANGE, 1.2),
                ("完全随机", poisson_times(), "10 分", (1.0, 0.36, 0.36), self.at(1) + 0.2)]
        for i, (name, times, wait, col, t0) in enumerate(rows):
            a = smooth(ramp(t, t0, t0 + 0.35))
            if a <= 0:
                continue
            ax = Axis(40, x0=420, x1=1500, y=330 + i * 190)
            line(img, (ax.x0, ax.y), (ax.x1, ax.y), (0.55, 0.58, 0.66), 3, alpha=a)
            for m in times:
                circle(img, ax.x(m), ax.y, 11, col, alpha=a)
            draw_text(img, name, ax.x0 - 40, ax.y, 40, "bold", WHITE, alpha=a, anchor="rm")
            draw_text(img, f"平均等 {wait}", ax.x1 + 50, ax.y, 44, "title", col, alpha=a, anchor="lm")
        return img

    def sfx(self):
        return [(0.5, "click", 0.6), (1.2, "click", 0.6), (self.at(1) + 0.2, "ding", 0.5)]


# ---------- 8. 串车 ----------

class Bunching(Scene):
    def draw(self, t):
        img = vgradient((0.10, 0.14, 0.26), (0.30, 0.30, 0.40), curve=1.0)
        road_y = 640
        rect(img, 0, road_y, W, H, (0.08, 0.09, 0.12))
        for x in range(0, W, 120):
            rect(img, x, road_y + 120, x + 60, road_y + 128, (0.6, 0.6, 0.5), 0.6)
        u = ramp(t, 0.3, self.d - 0.4)
        xa = lerp(780, 1700, ease_out(u) * 0.75 + u * 0.25)
        xb = lerp(130, 1385, u)
        xc = lerp(-560, 1070, ease_in_out(u))
        stops = [1000, 1450]
        for sx in stops:
            rect(img, sx - 4, road_y - 200, sx + 4, road_y, (0.75, 0.78, 0.85))
            over(img, round_rect_rgba(60, 40, 8, TEAL), sx - 30, road_y - 230)
            crowd = 3 + int(9 * ramp(t, 0, self.at(1) + 1.5)) if sx > xa else 1
            for k in range(crowd):
                you_marker(img, sx + 30 + (k % 5) * 22, road_y - 6 - (k // 5) * 50, alpha=0.9, color=(0.9, 0.92, 0.98))
        for x, lab in ((xc, "C"), (xb, "B"), (xa, "A")):
            bus(img, x, road_y + 20, 300)
            draw_text(img, lab, x - 150, road_y - 150, 44, "title", WHITE, shadow=0.7)
        fade_text(img, "前车：人越来越多，越来越慢", W / 2, 180, 50, t, self.at(1), color=ORANGE, name="title")
        fade_text(img, "后车：人越来越少，越来越快", W / 2, 260, 50, t, self.at(2), color=TEAL, name="title")
        title_text(img, "串车", W / 2, 345, 110, t, self.at(3) + 0.2, color=WHITE)
        return img

    def sfx(self):
        return [(self.at(3) + 0.2, "low_ding", 0.6)]


# ---------- 9. 班级人数 ----------

class Classes(Scene):
    def draw(self, t):
        img = dark_bg(t)
        fade_text(img, "这个陷阱，不只在车站", W / 2, 150, 58, t, 0.1, name="title")
        a = smooth(ramp(t, self.at(1), self.at(1) + 0.4))
        boxes = [(480, 10, 2, TEAL, "小班 10 人"), (1290, 90, 10, ORANGE, "大班 90 人")]
        for cx, n, cols, col, lab in boxes:
            if a <= 0:
                continue
            rows = int(np.ceil(n / cols))
            bw, bh = cols * 52 + 40, rows * 52 + 40
            over(img, round_rect_rgba(bw, bh, 14, (0.08, 0.1, 0.16)), cx - bw / 2, 470 - bh / 2, a)
            k = int(n * ease_out(ramp(t, self.at(1) + 0.2, self.at(1) + 1.6)))
            for i in range(k):
                x = cx - bw / 2 + 46 + (i % cols) * 52
                y = 470 - bh / 2 + 46 + (i // cols) * 52
                circle(img, x, y, 16, col, alpha=a)
            draw_text(img, lab, cx, 470 + bh / 2 + 40, 36, "bold", col, alpha=a)
        fade_text(img, "学校说：平均每班 50 人", 480, 840, 40, t, self.at(1) + 1.6, color=WHITE, name="bold")
        title_text(img, "学生感受：平均 82 人", 1290, 840, 48, t, self.at(2) + 1.8, color=ORANGE)
        return img

    def sfx(self):
        return [(self.at(2) + 1.8, "ding", 0.5)]


# ---------- 10. 健身房 ----------

HOURS = np.arange(6, 24)
CROWD = np.array([8, 14, 18, 10, 8, 9, 14, 12, 9, 10, 14, 30, 52, 60, 48, 30, 18, 8], float)


class Gym(Scene):
    def draw(self, t):
        img = dark_bg(t)
        fade_text(img, "健身房：每个小时有多少人", W / 2, 150, 54, t, 0.1, name="title")
        x0, x1, y = 200, 1440, 760
        bw = (x1 - x0) / len(HOURS)
        p = ease_out(ramp(t, 0.3, 1.4))
        for i, (h, c) in enumerate(zip(HOURS, CROWD)):
            col = ORANGE if c > 40 else (0.36, 0.42, 0.55)
            rect(img, x0 + i * bw + 6, y - c * 8 * p, x0 + (i + 1) * bw - 6, y, col, 0.95)
            if h % 3 == 0:
                draw_text(img, f"{h} 点", x0 + (i + 0.5) * bw, y + 34, 24, "medium", GREY)
        avg_t = CROWD.mean()
        avg_p = (CROWD ** 2).sum() / CROWD.sum()
        a = smooth(ramp(t, self.at(0) + 2.2, self.at(0) + 2.6))
        if a > 0:
            line(img, (x0, y - avg_t * 8), (x1, y - avg_t * 8), TEAL, 3, alpha=a)
            draw_text(img, f"按时间平均：{avg_t:.0f} 人", x1 + 10, y - avg_t * 8, 30, "bold", TEAL, alpha=a, anchor="lm")
        b = smooth(ramp(t, self.at(0) + 3.4, self.at(0) + 3.8))
        if b > 0:
            line(img, (x0, y - avg_p * 8), (x1, y - avg_p * 8), ORANGE, 3, alpha=b)
            draw_text(img, f"你感受到的：{avg_p:.0f} 人", x1 + 10, y - avg_p * 8, 30, "bold", ORANGE, alpha=b, anchor="lm")
        return img


# ---------- 11. 结论 / 12. 提问 ----------

class Verdict(Scene):
    def draw(self, t):
        img = street().copy() * 0.8
        clock(img, 1640, 250, 110, 8.2 * 6 * smooth(ramp(t, 0, 2.0)))
        title_text(img, "不是你运气差", 1180, 380, 80, t, 0.2)
        title_text(img, "是你更容易掉进长等待", 1180, 500, 72, t, 1.4, color=ORANGE)
        return img


class Question(Scene):
    def draw(self, t):
        img = street().copy() * 0.85
        clock(img, 1640, 250, 110, t * 3)
        fade_text(img, "你等过最久的一趟车，是多久？", 1150, 400, 64, t, 0.1, name="title")
        fade_text(img, "评论区说说 ↓", 1150, 500, 46, t, 1.4, color=TEAL, name="title")
        return img


# ---------- 13. 下期预告 ----------

class Next(Scene):
    def draw(self, t):
        img = sky(t, gain=0.7)
        sx, sy, sd = 760, 540, 300
        halo(img, sx, sy, 640, (1.0, 0.72, 0.32), 0.5)
        over(img, sphere(sd, star_texture("sun"), t * 0.05, emissive=True, limb=0.45), sx - sd / 2, sy - sd / 2)
        bx, by = sx + sd / 2 + 24, sy
        k = smooth(ramp(t, 1.0, 1.5))
        line(img, (bx - 16, by - 16), (bx + 16, by + 16), ORANGE, 5, alpha=k)
        line(img, (bx - 16, by + 16), (bx + 16, by - 16), ORANGE, 5, alpha=k)
        draw_text(img, "质心", bx + 30, by - 34, 34, "bold", ORANGE, alpha=k, anchor="lm")
        pill(img, "下期预告", 1400, 330, 34, (0.1, 0.04, 0.0), ORANGE, alpha=smooth(ramp(t, 0.05, 0.35)))
        title_text(img, "太阳系的中心", 1400, 450, 80, t, 0.3)
        title_text(img, "经常不在太阳里", 1400, 560, 80, t, 1.2, color=TEAL)
        return img

    def sfx(self):
        return [(1.0, "ding", 0.5)]


SCENE_CLASSES = {"hook": Hook, "title": make_title("第一季 · 反直觉 #02", "公交车 10 分钟一班，为什么你总要等更久？"),
                 "ideal": Ideal, "reality": Reality, "trap": Trap, "calc": Calc, "rule": Rule,
                 "bunching": Bunching, "classes": Classes, "gym": Gym, "verdict": Verdict,
                 "question": Question, "next": Next, "outro": Outro}


def music(spans, total):
    sp = {s.name: s for s in spans}
    return [(0, sp["verdict"].start, "calm"), (sp["verdict"].start - 0.5, total, "wonder")], []
