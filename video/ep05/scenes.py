"""第 5 期场景（季终）：夜空为什么是黑的？"""
from functools import lru_cache

import cv2
import numpy as np

from engine.core import (GREY, H, ORANGE, TEAL, W, WHITE, circle, draw_text, ease_in_out, ease_out, fbm, fill_poly,
                         line, over, polyline, ramp, round_rect_rgba, smooth, vgradient)
from engine.cosmos import (StarField, earth_texture, halo, hill_silhouette, milky_way, person_silhouette, sphere,
                           star_point, star_texture)
from engine.props import bus_icon
from engine.ui import Scene, count, fade_text, make_outro, make_title, night_gradient, pill, pop, sky, title_text

SUN = (1.0, 0.86, 0.62)
INK = (0.10, 0.06, 0.03)
PANEL = (0.03, 0.04, 0.07)


class S(Scene):
    def mid(self, k, f):
        """第 k 句里大约 f 比例处的时间（长句中途触发画面用）。"""
        return self.at(k) + (self.end(k) - self.at(k)) * f


def norm01(n):
    return (n - n.min()) / (n.max() - n.min() + 1e-9)


# ---------- 共享素材 ----------

@lru_cache(maxsize=1)
def night_scene():
    mw, dens = milky_way(W, H, seed=7, angle=-20, width=0.14, center=(0.55, 0.4), strength=0.35)
    sf = StarField(n=5200, seed=5, density=dens, bright_n=70)
    mask = np.zeros((H, W, 3), np.float32)
    xs, ys = hill_silhouette(mask, 920, 50, seed=4, color=(1, 1, 1))
    person_silhouette(mask, 1450, np.interp(1450, xs, ys) + 5, 160, color=(1, 1, 1), look_up=0.6)
    fg = np.zeros((H, W, 4), np.float32)
    fg[..., :3] = (0.010, 0.013, 0.025)
    fg[..., 3] = mask[..., 0]
    glow = vgradient((0, 0, 0), (0.05, 0.06, 0.12), curve=4.0)
    return night_gradient() + mw + glow, sf, fg


@lru_cache(maxsize=1)
def cmb_sky():
    n = norm01(fbm(H // 2, W // 2, 61, 5, base=12))
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    return (np.array((0.20, 0.13, 0.26)) * (0.7 + 0.6 * n[..., None])).astype(np.float32)


def night(t, glow=0.0):
    base, sf, fg = night_scene()
    img = base + sf.render(t)
    if glow > 0:
        img += cmb_sky() * glow
    over(img, fg, 0, 0)
    return img


# ---------- 1. 钩子 ----------

class Hook(S):
    tag = False

    def draw(self, t):
        img = night(t)
        title_text(img, "夜空，为什么是黑的？", W / 2, 300, 104, t, 0.15)
        fade_text(img, "难倒了天文学家几百年", W / 2, 425, 50, t, self.at(1) + 0.2, color=GREY, name="medium")
        s, a = pop(t, self.mid(1, 0.5))
        if a > 0:
            pill(img, "第一个答对方向的：一位恐怖小说作家", W / 2, 540, 42, (0.1, 0.04, 0.0), ORANGE, alpha=a)
        return img

    def sfx(self):
        return [(0.1, "whoosh", 0.5), (self.mid(1, 0.5), "low_ding", 0.6)]


# ---------- 3. 森林 ----------

@lru_cache(maxsize=1)
def forest():
    rng = np.random.default_rng(12)
    OB = np.array([W / 2, 560.0])
    P = rng.uniform([-60, -60], [W + 60, H + 60], (900, 2))
    P = P[np.hypot(*(P - OB).T) > 110][:430]
    r = rng.uniform(8, 15, len(P))
    ang = np.linspace(0, 2 * np.pi, 140, endpoint=False) + 0.013
    D = np.stack([np.cos(ang), np.sin(ang)], 1)
    rel = P - OB
    tc = D @ rel.T
    perp2 = (rel ** 2).sum(1)[None] - tc ** 2
    ok = (perp2 < r[None] ** 2) & (tc > 0)
    dist = np.where(ok, tc - np.sqrt(np.maximum(r[None] ** 2 - perp2, 0)), np.inf)
    hit = np.minimum(dist.min(1), 1500)
    rgb = np.zeros((H, W, 3), np.float32)
    mask = np.zeros((H, W, 3), np.float32)
    stars = np.zeros((H, W, 3), np.float32)
    for (x, y), rr in zip(P, r):
        circle(rgb, x, y, rr, (0.30, 0.20, 0.12))
        circle(rgb, x, y, rr * 0.6, (0.42, 0.30, 0.18))
        circle(mask, x, y, rr, (1, 1, 1))
        star_point(stars, x, y, rr * 0.32, (1.0, 0.95, 0.85))
    a = mask[..., :1]
    trees = np.dstack([rgb / np.maximum(a, 1e-3), a]).astype(np.float32)
    floor = vgradient((0.030, 0.060, 0.040), (0.050, 0.085, 0.050))
    return OB, D, hit, trees, stars, floor


class Forest(S):
    def draw(self, t):
        OB, D, hit, trees, stars, floor = forest()
        q = smooth(ramp(t, self.at(1) - 0.1, self.at(1) + 0.7))
        img = floor * (1 - q) + sky(t, gain=0.6) * q
        if q < 1:
            over(img, trees, 0, 0, 1 - q)
        if q > 0:
            img += stars * q
        L = 1500 * ease_out(ramp(t, 0.4, 2.6))
        col = (1.0, 0.85, 0.5)
        for d, h in zip(D, hit):
            end = OB + d * min(L, h)
            line(img, OB, end, col, 1.6, alpha=0.4)
            if L >= h:
                circle(img, end[0], end[1], 3.5, ORANGE if q < 0.5 else WHITE, alpha=0.9)
        circle(img, OB[0], OB[1], 22, TEAL)
        draw_text(img, "你", OB[0], OB[1] - 1, 26, "heavy", (0.02, 0.05, 0.08))
        a0 = smooth(ramp(t, 0.2, 0.5)) * (1 - q)
        pill(img, "无边的森林：视线总会撞上一棵树", W / 2, 130, 42, WHITE, PANEL, alpha=a0)
        pill(img, "无限的宇宙：视线总会撞上一颗星", W / 2, 130, 42, WHITE, PANEL, alpha=q)
        return img

    def sfx(self):
        return [(0.4, "whoosh", 0.4), (self.at(1), "shimmer", 0.5)]


# ---------- 4. 一暗一多 ----------

class Shells(S):
    EYE = (200, 540)
    PLATES = [(560, 80, 1), (920, 160, 2), (1280, 240, 3)]
    LABELS = [("1 颗", "亮度 1"), ("4 颗", "每颗 1/4"), ("9 颗", "每颗 1/9")]

    def draw(self, t):
        img = sky(t, gain=0.35) * 0.8
        ex, ey = self.EYE
        starts = [0.3, self.at(1) + 0.1, self.mid(1, 0.55)]
        cone = ease_in_out(ramp(t, 0.2, 0.9))
        x3, _, _ = self.PLATES[-1]
        slope = self.PLATES[0][1] / (self.PLATES[0][0] - ex)
        for sgn in (-1, 1):
            tip = (ex + (x3 + 200 - ex) * cone, ey + sgn * slope * (x3 + 200 - ex) * cone)
            line(img, (ex, ey), tip, GREY, 2, alpha=0.6)
        circle(img, ex, ey, 20, TEAL)
        draw_text(img, "你", ex, ey - 1, 24, "heavy", (0.02, 0.05, 0.08))
        for k, ((x, s, n), t0) in enumerate(zip(self.PLATES, starts)):
            a = smooth(ramp(t, t0, t0 + 0.35))
            if a <= 0:
                continue
            w = s * 0.7
            pts = [(x - w / 2, ey - s), (x + w / 2, ey - s), (x + w / 2, ey + s), (x - w / 2, ey + s)]
            fill_poly(img, pts, (0.24, 0.96, 0.84), alpha=0.07 * a)
            polyline(img, pts, TEAL, 2, closed=True, alpha=0.7 * a)
            for i in range(1, n):
                line(img, (x - w / 2 + w * i / n, ey - s), (x - w / 2 + w * i / n, ey + s), TEAL, 1, alpha=0.35 * a)
                line(img, (x - w / 2, ey - s + 2 * s * i / n), (x + w / 2, ey - s + 2 * s * i / n), TEAL, 1, alpha=0.35 * a)
            for i in range(n):
                for j in range(n):
                    cx, cy = x - w / 2 + w * (i + 0.5) / n, ey - s + 2 * s * (j + 0.5) / n
                    star_point(img, cx, cy, 11 / n ** 0.5, (1.0, 0.92, 0.75), k=a / n)
            draw_text(img, f"距离 {k + 1}", x, ey - s - 40, 34, "bold", GREY, alpha=a)
            draw_text(img, self.LABELS[k][0], x, ey + s + 45, 40, "title", WHITE, alpha=a)
            draw_text(img, self.LABELS[k][1], x, ey + s + 92, 32, "medium", ORANGE, alpha=a)
        fade_text(img, "远处的星星很暗？", W / 2, 120, 52, t, 0.2, name="title")
        return img

    def sfx(self):
        return [(self.at(1) + 0.1, "click", 0.5), (self.mid(1, 0.55), "click", 0.5)]


# ---------- 5. 夜空像太阳一样亮 ----------

@lru_cache(maxsize=1)
def star_layers():
    rng = np.random.default_rng(31)
    out = []
    for k in range(6):
        n = int(400 * 3 ** k)
        img = np.zeros((H, W), np.float32)
        np.add.at(img, (rng.integers(0, H, n), rng.integers(0, W, n)), 1.0)
        img = cv2.GaussianBlur(img, (0, 0), 1.0) * (260.0 / 3 ** k)
        out.append(img)
    return out


@lru_cache(maxsize=1)
def sun_surface():
    n = norm01(fbm(H // 4, W // 4, 77, 5, base=18))
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    return (np.array(SUN) * (0.9 + 0.12 * n[..., None])).astype(np.float32)


class Bright(S):
    def draw(self, t):
        img = sky(t, gain=0.5) * 0.7
        p = ramp(t, 0.2, self.mid(0, 0.62))
        acc = np.zeros((H, W), np.float32)
        for k, layer in enumerate(star_layers()):
            a = smooth(ramp(p, k / 7, k / 7 + 0.18))
            if a > 0:
                acc += layer * a
        img += np.clip(acc, 0, 1.5)[..., None] * np.array(SUN, np.float32) * 0.8
        f = smooth(ramp(t, self.mid(0, 0.55), self.mid(0, 0.8)))
        if f > 0:
            img = img * (1 - f) + sun_surface() * f
        n = min(6, int(p * 7) + 1)
        a0 = smooth(ramp(t, 0.2, 0.5)) * (1 - f)
        pill(img, "一暗一多，正好抵消", W / 2, 130, 44, WHITE, PANEL, alpha=a0)
        fade_text(img, f"叠加 {n} 层", W / 2, 215, 40, t, 0.4, color=GREY, name="bold", alpha=1 - f)
        pill(img, "每一层贡献的光一样多", W / 2, 310, 40, ORANGE, PANEL,
             alpha=smooth(ramp(t, self.mid(0, 0.25), self.mid(0, 0.25) + 0.3)) * (1 - f))
        title_text(img, "整个夜空 ≈ 太阳表面", W / 2, 470, 110, t, self.mid(0, 0.78), color=INK, shadow=0)
        fade_text(img, "（无限、永恒、均匀的宇宙里）", W / 2, 590, 40, t, self.mid(0, 0.85), color=(0.35, 0.22, 0.1),
                  name="medium")
        return img

    def sfx(self):
        return [(self.mid(0, 0.55), "swish", 0.6), (self.mid(0, 0.78), "boom", 0.5)]


# ---------- 6. 历史 ----------

YEAR0, YEAR1, AX0, AX1, AXY = 1590, 1920, 200, 1720, 640


def year_x(yr):
    return AX0 + (yr - YEAR0) / (YEAR1 - YEAR0) * (AX1 - AX0)


def timeline(img, t, events, axis_t0=0.1):
    p = ease_in_out(ramp(t, axis_t0, axis_t0 + 0.6))
    line(img, (AX0, AXY), (AX0 + (AX1 - AX0) * p, AXY), GREY, 3)
    for yr in range(1600, 1901, 50):
        if year_x(yr) <= AX0 + (AX1 - AX0) * p:
            line(img, (year_x(yr), AXY - 8), (year_x(yr), AXY + 8), GREY, 2)
            draw_text(img, str(yr), year_x(yr), AXY + 36, 26, "medium", GREY)
    for yr, name, desc, up, t0, col in events:
        s, a = pop(t, t0)
        if a <= 0:
            continue
        x = year_x(yr)
        circle(img, x, AXY, 12 * s, col, alpha=a)
        y = AXY - 190 if up else AXY + 120
        line(img, (x, AXY + (-16 if up else 16)), (x, y + (60 if up else -40)), col, 2, alpha=a * 0.7)
        draw_text(img, f"{yr} · {name}", x, y, 46, "title", col, alpha=a, scale=s, shadow=0.6)
        draw_text(img, desc, x, y + 52, 32, "medium", WHITE, alpha=a, shadow=0.6)


class History(S):
    def draw(self, t):
        img = sky(t, gain=0.4) * 0.8
        fade_text(img, "一个问题，困扰了几百年", W / 2, 150, 56, t, 0.15, name="title")
        timeline(img, t, [(1610, "开普勒", "宇宙不可能无限？", True, 0.5, TEAL),
                          (1823, "奥伯斯", "正式提出 → “奥伯斯佯谬”", False, self.at(1) + 0.2, ORANGE)])
        return img

    def sfx(self):
        return [(0.5, "ding", 0.45), (self.at(1) + 0.2, "ding", 0.45)]


# ---------- 7. 尘埃猜想 ----------

@lru_cache(maxsize=1)
def dust_cloud(w=560, h=440):
    n = norm01(fbm(h, w, 17, 5, base=4))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    a = np.clip((n - 0.3) * 2.4 + (1 - d) * 1.4 - 0.7, 0, 1) * np.clip(1 - d, 0, 1) ** 0.5
    return n.astype(np.float32), a.astype(np.float32)


class Dust(S):
    STAR, CLOUD, EYE = (300, 540), (W / 2 - 120, 540), (1560, 540)

    def draw(self, t):
        img = sky(t, gain=0.4) * 0.75
        n, a = dust_cloud()
        h = smooth(ramp(t, self.mid(0, 0.45), self.mid(0, 0.8)))
        sx, sy = self.STAR
        cx, cy = self.CLOUD
        ex, ey = self.EYE
        p = ease_out(ramp(t, 0.3, 1.2))
        for dy in (-120, -60, 0, 60, 120):
            line(img, (sx + 40, sy + dy * 0.2), (sx + 40 + (cx - 200 - sx) * p, sy + dy), (1.0, 0.88, 0.6), 2, alpha=0.6)
        halo(img, sx, sy, 160, (1.0, 0.85, 0.6), 0.8)
        star_point(img, sx, sy, 14, (1.0, 0.95, 0.85))
        if h > 0:
            halo(img, cx, cy, 360, (1.0, 0.4, 0.12), 0.6 * h)
            for dy in (-90, 0, 90):
                line(img, (cx + 230, cy + dy), (cx + 230 + (ex - 40 - cx - 230) * h, ey + dy * 0.2), ORANGE, 2,
                     alpha=0.7 * h)
        dark, hot = np.array((0.10, 0.07, 0.05)), np.array((1.0, 0.42, 0.12))
        col = (dark * (1 - h) + hot * h)[None, None] * (0.6 + 0.7 * n[..., None])
        over(img, np.dstack([col, a]).astype(np.float32), cx - n.shape[1] / 2, cy - n.shape[0] / 2)
        circle(img, ex, ey, 22, TEAL)
        draw_text(img, "你", ex, ey - 1, 26, "heavy", (0.02, 0.05, 0.08))
        fade_text(img, "猜想：星际尘埃挡住了光？", W / 2, 150, 56, t, 0.15, name="title")
        fade_text(img, "吸收 → 变热 → 自己发光", cx, 840, 48, t, self.mid(0, 0.5), color=ORANGE, name="title")
        s, al = pop(t, self.mid(0, 0.82))
        if al > 0:
            pill(img, "× 行不通", ex, 360, 44, WHITE, (0.75, 0.18, 0.12), alpha=al)
        return img

    def sfx(self):
        return [(self.mid(0, 0.5), "swish", 0.5), (self.mid(0, 0.82), "click", 0.6)]


# ---------- 8. 爱伦·坡 ----------

QUOTE = ["望远镜里那些看不到星星的方向，", "是因为背景太远，", "那里的光，至今还没能到达我们。"]


class Poe(S):
    def draw(self, t):
        img = sky(t, gain=0.4) * 0.75
        cx, cy, cw, ch = 640, 520, 880, 540
        a = smooth(ramp(t, 0.1, 0.45))
        over(img, round_rect_rgba(cw + 12, ch + 12, 26, (0.55, 0.40, 0.22)), cx - cw / 2 - 6, cy - ch / 2 - 6, a)
        over(img, round_rect_rgba(cw, ch, 22, (0.93, 0.88, 0.76)), cx - cw / 2, cy - ch / 2, a)
        draw_text(img, "Eureka · 1848", cx, cy - 205, 40, "title", (0.55, 0.30, 0.14), alpha=a)
        draw_text(img, "《我发现了》", cx, cy - 140, 56, "heavy", INK, alpha=a)
        total = sum(len(q) for q in QUOTE)
        shown = int(total * ramp(t, self.at(1) + 0.3, self.end(1) - 0.3))
        for i, q in enumerate(QUOTE):
            k = max(0, min(len(q), shown))
            shown -= len(q)
            if k:
                draw_text(img, q[:k], cx - cw / 2 + 70, cy - 40 + i * 72, 40, "medium", INK, alpha=a, anchor="lm")
        fade_text(img, "—— 爱伦·坡（大意）", cx + cw / 2 - 60, cy + ch / 2 - 50, 30, t, self.end(1) - 0.4,
                  color=(0.45, 0.32, 0.2), name="medium", anchor="rm")
        rx = 1500
        s, al = pop(t, 0.3)
        if al > 0:
            pill(img, "1848 · 爱伦·坡", rx, 330, 46, (0.1, 0.04, 0.0), ORANGE, alpha=al)
        fade_text(img, "恐怖小说、推理小说作家", rx, 420, 36, t, 0.7, color=GREY, name="medium")
        s, al = pop(t, self.at(2) + 0.3)
        if al > 0:
            pill(img, "1901 · 开尔文", rx, 600, 46, (0.02, 0.06, 0.08), TEAL, alpha=al)
        fade_text(img, "用计算证实了", rx, 690, 40, t, self.at(2) + 0.7, color=TEAL, name="title")
        return img

    def sfx(self):
        return [(0.3, "ding", 0.45), (self.at(2) + 0.3, "ding", 0.45)]


# ---------- 9–10. 答案：宇宙有年龄、星星会燃尽 ----------

UC, UR = (640, 560), 400


@lru_cache(maxsize=1)
def universe_stars():
    rng = np.random.default_rng(8)
    P = rng.uniform([-40, -40], [1180, H + 40], (700, 2))
    P = P[np.hypot(*(P - UC).T) > 40]
    size = rng.uniform(1.6, 3.6, len(P))
    death = rng.uniform(0.05, 1.0, len(P))
    return P, size, death


def answer_column(img, t, hi, t1, t2, pill_t0=0.1):
    rx = 1500
    pill(img, "答案主要有两条", rx, 250, 40, WHITE, PANEL, alpha=smooth(ramp(t, pill_t0, pill_t0 + 0.3)))
    for k, (text, t0) in enumerate((("① 宇宙有年龄", t1), ("② 星星会燃尽", t2))):
        c = TEAL if hi == k else GREY
        fade_text(img, text, rx, 360 + k * 90, 58, t, t0, color=c, name="title")


def observer(img):
    circle(img, UC[0], UC[1], 20, TEAL)
    draw_text(img, "你", UC[0], UC[1] - 1, 24, "heavy", (0.02, 0.05, 0.08))


class Answer(S):
    def draw(self, t):
        img = sky(t, gain=0.25) * 0.7
        P, size, _ = universe_stars()
        g = ease_in_out(ramp(t, self.at(1) + 0.2, self.mid(1, 0.45)))
        R = UR * g
        if R > 2:
            circle(img, *UC, R, (0.24, 0.96, 0.84), alpha=0.06)
            circle(img, *UC, R, TEAL, thickness=3, alpha=0.8)
        dist = np.hypot(*(P - UC).T)
        for (x, y), s, d in zip(P, size, dist):
            inside = d < R
            star_point(img, x, y, s, (1.0, 0.93, 0.8) if inside else (0.5, 0.55, 0.65), k=1.0 if inside else 0.35)
        q = smooth(ramp(t, self.mid(1, 0.6), self.mid(1, 0.75)))
        if q > 0:
            for j in range(0, len(P), 11):
                if dist[j] > UR + 60:
                    u = (UC - P[j]) / dist[j]
                    f = (t * 0.35 + j * 0.13) % 1.0
                    a0 = P[j] + u * (dist[j] - UR - 40) * f
                    line(img, a0, a0 + u * 40, (1.0, 0.85, 0.5), 3.5, alpha=q * (1 - 0.6 * f))
                    circle(img, *(a0 + u * 40), 4, (1.0, 0.9, 0.6), alpha=q * (1 - 0.6 * f))
            draw_text(img, "光还在路上…", UC[0] + 330, UC[1] - UR - 20, 38, "title", (1.0, 0.85, 0.5), alpha=q,
                      shadow=0.8)
        observer(img)
        v = count(t, self.at(1) + 0.2, self.mid(1, 0.45) - self.at(1) - 0.2, 138)
        if g > 0:
            pill(img, f"宇宙年龄 {v:.0f} 亿年", UC[0], UC[1] - UR - 40, 34, (0.02, 0.06, 0.08), TEAL,
                 alpha=smooth(min(1.0, g * 4)))
        answer_column(img, t, 0 if t > self.at(1) else -1, self.at(0) + 0.3, self.at(0) + 0.8)
        fade_text(img, "光只来得及从有限远的地方赶来", 1500, 620, 34, t, self.mid(1, 0.5), color=WHITE, name="medium")
        return img

    def sfx(self):
        return [(self.at(1) + 0.2, "whoosh", 0.4)]


def wave(img, x0, x1, y, t, amp=30, lam0=26, lam1=120, a=1.0):
    xs = np.linspace(x0, x1, 360)
    f = (xs - x0) / (x1 - x0)
    lam = lam0 + (lam1 - lam0) * f
    phase = np.cumsum(2 * np.pi / lam * (xs[1] - xs[0])) - t * 6
    ys = y + amp * (1 - 0.6 * f) * np.sin(phase)
    c0, c1 = np.array((0.4, 0.7, 1.0)), np.array((1.0, 0.25, 0.15))
    for k in range(0, len(xs) - 1, 6):
        c = c0 * (1 - f[k]) + c1 * f[k]
        polyline(img, list(zip(xs[k:k + 7], ys[k:k + 7])), tuple(c), 3, alpha=a * (1 - 0.5 * f[k]))


class Fuel(S):
    def draw(self, t):
        img = sky(t, gain=0.25) * 0.7
        P, size, death = universe_stars()
        circle(img, *UC, UR, (0.24, 0.96, 0.84), alpha=0.06)
        circle(img, *UC, UR, TEAL, thickness=3, alpha=0.8)
        dist = np.hypot(*(P - UC).T)
        life = ramp(t, self.at(0) + 0.6, self.end(0))
        for (x, y), s, d, dd in zip(P, size, dist, death):
            if d >= UR:
                star_point(img, x, y, s, (0.5, 0.55, 0.65), k=0.35)
                continue
            e = (life - dd * 1.4) / 0.12
            if e < 0:
                star_point(img, x, y, s, (1.0, 0.93, 0.8))
            elif e < 1:
                star_point(img, x, y, s * (1 + 1.5 * e), (1.0, 0.45, 0.2), k=1 - e)
        observer(img)
        answer_column(img, t, 1, -1, -1, pill_t0=-1)
        fade_text(img, "能量不够填满整个天空", 1500, 560, 36, t, self.mid(0, 0.55), color=WHITE, name="medium")
        a = smooth(ramp(t, self.at(1), self.at(1) + 0.4))
        fade_text(img, "+ 宇宙在膨胀", 1500, 680, 50, t, self.at(1), color=ORANGE, name="title")
        if a > 0:
            wave(img, 1250, 1760, 790, t, a=a)
            fade_text(img, "光被拉长 · 变红 · 变暗", 1500, 870, 34, t, self.at(1) + 0.6, color=GREY, name="medium")
        return img

    def sfx(self):
        return [(self.at(1), "swish", 0.5)]


# ---------- 11. 夜空其实在发光 ----------

@lru_cache(maxsize=1)
def plasma():
    n = norm01(fbm(H // 3, W // 3, 91, 5, base=8))
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    return (np.array((1.0, 0.55, 0.22)) * (0.55 + 0.6 * n[..., None])).astype(np.float32)


class Glow(S):
    def draw(self, t):
        img = sky(t, gain=0.6) * 0.8
        g = smooth(ramp(t, self.at(1), self.at(1) + 1.0)) * (1 - smooth(ramp(t, self.at(2) + 0.3, self.mid(2, 0.7))))
        if g > 0:
            img = img * (1 - g) + plasma() * g
        title_text(img, "夜空其实并不黑", W / 2, 480, 120, t, 0.15, fade_out=self.at(1) - 0.2)
        a = smooth(ramp(t, self.at(1) + 0.5, self.at(1) + 0.8)) * g
        pill(img, "宇宙诞生后约 38 万年", W / 2, 400, 52, WHITE, PANEL, alpha=a)
        pill(img, "约 2700 ℃ · 和白炽灯丝差不多", W / 2, 520, 44, ORANGE, PANEL,
             alpha=smooth(ramp(t, self.mid(1, 0.5), self.mid(1, 0.5) + 0.3)) * g)
        w = smooth(ramp(t, self.at(2), self.at(2) + 0.4))
        if w > 0:
            k = ease_in_out(ramp(t, self.at(2) + 0.2, self.mid(2, 0.8)))
            xs = np.linspace(160, W - 160, 700)
            lam = 40 * (1 + 9 * k)
            ys = 560 + 70 * np.sin(2 * np.pi * (xs - 160) / lam - t * 4)
            c = np.array((1.0, 0.6, 0.25)) * (1 - k) + np.array((0.55, 0.45, 0.6)) * k
            polyline(img, list(zip(xs, ys)), tuple(c), 4, alpha=w)
            pill(img, "可见光  →  拉长 1000 多倍  →  微波", W / 2, 330, 44, WHITE, PANEL,
                 alpha=smooth(ramp(t, self.mid(2, 0.3), self.mid(2, 0.3) + 0.3)))
        return img

    def sfx(self):
        return [(0.15, "low_ding", 0.6), (self.at(1), "shimmer", 0.6), (self.at(2) + 0.2, "whoosh", 0.4)]


# ---------- 12. 微波背景 ----------

@lru_cache(maxsize=1)
def cmb_map(w=1240, h=620):
    n = norm01(fbm(h, w, 5, 5, base=30))
    stops = [0.0, 0.32, 0.5, 0.68, 1.0]
    cols = np.array([(0.0, 0.05, 0.35), (0.2, 0.5, 0.95), (0.95, 0.92, 0.85), (1.0, 0.6, 0.2), (0.6, 0.05, 0.02)])
    v = np.clip((n - 0.5) * 1.3 + 0.5, 0, 1)
    rgb = np.stack([np.interp(v, stops, cols[:, c]) for c in range(3)], -1)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((xx - w / 2) / (w / 2 - 2), (yy - h / 2) / (h / 2 - 2))
    a = np.clip((1 - d) * 300, 0, 1)
    return np.dstack([rgb, a]).astype(np.float32)


class CMB(S):
    def draw(self, t):
        sw = smooth(ramp(t, self.mid(0, 0.45), self.mid(0, 0.6)))
        img = sky(t, gain=0.4) * 0.7
        if sw < 1:
            m = cmb_map()
            a = smooth(ramp(t, 0.1, 0.6)) * (1 - sw)
            over(img, m, W / 2 - m.shape[1] / 2, 520 - m.shape[0] / 2, a)
            title_text(img, "宇宙微波背景辐射", W / 2, 130, 72, t, 0.1, fade_out=self.mid(0, 0.45))
            fade_text(img, "2.7 K ≈ −270 ℃", W / 2, 885, 48, t, self.mid(0, 0.2), color=TEAL, name="title",
                      alpha=1 - sw)
        if sw > 0:
            img = img * (1 - sw) + night(t, glow=sw * smooth(ramp(t, self.mid(0, 0.6), self.mid(0, 0.85)))) * sw
            fade_text(img, "如果眼睛能看见微波", W / 2, 220, 52, t, self.mid(0, 0.55), color=GREY, name="title")
            fade_text(img, "整个夜空都在发光", W / 2, 320, 84, t, self.mid(0, 0.75), color=WHITE, name="title")
        return img

    def sfx(self):
        return [(0.1, "shimmer", 0.5), (self.mid(0, 0.75), "low_ding", 0.6)]


# ---------- 13. 结论 ----------

class Verdict(S):
    def draw(self, t):
        img = night(t)
        title_text(img, "夜空是黑的", W / 2, 230, 96, t, 0.2)
        title_text(img, "因为宇宙有一个开始", W / 2, 360, 84, t, self.mid(0, 0.55), color=TEAL)
        s, a = pop(t, self.at(1) + 0.3)
        if a > 0:
            pill(img, "这片黑 = 宇宙年龄的证据", W / 2, 500, 44, (0.1, 0.04, 0.0), ORANGE, alpha=a)
        return img

    def sfx(self):
        return [(self.mid(0, 0.55), "low_ding", 0.6), (self.at(1) + 0.3, "ding", 0.5)]


# ---------- 14. 季终回顾 ----------

CARDS = [("#01", "你看到的星星", "还在吗？"), ("#02", "公交车为什么", "总要等更久？"), ("#03", "太阳系的中心", "不在太阳里"),
         ("#04", "地球上一定", "有地方没风"), ("#05", "夜空为什么", "是黑的？")]


def card_icon(img, k, x, y, t):
    if k == 0:
        halo(img, x, y, 120, (1.0, 0.5, 0.25), 0.7)
        star_point(img, x, y, 9, (1.0, 0.75, 0.55))
    elif k == 1:
        bus_icon(img, x, y, s=86)
    elif k == 2:
        halo(img, x, y, 110, (1.0, 0.72, 0.32), 0.6)
        over(img, sphere(100, star_texture("sun"), 0.3, emissive=True, limb=0.45), x - 50, y - 50)
        for sg in (1, -1):
            line(img, (x + 62, y - 52 + 10 * sg), (x + 82, y - 52 - 10 * sg), ORANGE, 5)
    elif k == 3:
        over(img, sphere(110, earth_texture(), t * 0.1 + 1.0, light=(-0.5, 0.3, 0.8)), x - 55, y - 55)
        circle(img, x + 12, y - 14, 7, ORANGE)
    else:
        over(img, round_rect_rgba(120, 110, 18, (0.01, 0.015, 0.04)), x - 60, y - 55)
        for dx, dy, r in ((-30, -24, 2.5), (22, -30, 2), (8, 6, 3), (-22, 24, 2), (34, 26, 2.2)):
            star_point(img, x + dx, y + dy, r, (0.9, 0.93, 1.0))


class Season(S):
    def draw(self, t):
        img = sky(t, gain=0.5) * 0.8
        title_text(img, "第一季 · 完", W / 2, 150, 80, t, 0.1)
        hi = int((t - self.at(1)) * 2.2) % 5 if t > self.at(1) else -1
        for k, (num, l1, l2) in enumerate(CARDS):
            x, y = 280 + k * 340, 520
            s, a = pop(t, 0.3 + k * 0.15)
            if a <= 0:
                continue
            cw, ch = int(300 * s), int(400 * s)
            border = TEAL if k == hi else (0.22, 0.26, 0.34)
            over(img, round_rect_rgba(cw + 8, ch + 8, 26, border), x - cw / 2 - 4, y - ch / 2 - 4, a)
            over(img, round_rect_rgba(cw, ch, 22, (0.06, 0.08, 0.14)), x - cw / 2, y - ch / 2, a)
            draw_text(img, num, x, y - 150 * s, 40, "title", TEAL if k == hi else GREY, alpha=a, scale=s)
            card_icon(img, k, x, y - 40 * s, t)
            draw_text(img, l1, x, y + 85 * s, 32, "bold", WHITE, alpha=a, scale=s)
            draw_text(img, l2, x, y + 130 * s, 32, "bold", WHITE, alpha=a, scale=s)
        fade_text(img, "哪一期最反直觉？第二季想看什么？评论区告诉我 ↓", W / 2, 820, 44, t, self.at(1) + 0.2,
                  color=TEAL, name="title")
        return img

    def sfx(self):
        return [(0.3 + k * 0.15, "click", 0.35) for k in range(5)]


SCENE_CLASSES = {"hook": Hook, "title": make_title("第一季 · 反直觉 #05 · 季终", "夜空为什么是黑的？"),
                 "forest": Forest, "shells": Shells, "bright": Bright, "history": History, "dust": Dust,
                 "poe": Poe, "answer": Answer, "fuel": Fuel, "glow": Glow, "cmb": CMB, "verdict": Verdict,
                 "season": Season, "outro": make_outro("我们第二季见")}


def music(spans, total):
    sp = {s.name: s for s in spans}
    return [(0, sp["shells"].start, "calm"), (sp["shells"].start, sp["bright"].end, "tension"),
            (sp["bright"].end - 0.5, sp["answer"].start, "calm"), (sp["answer"].start - 0.5, total, "wonder")], []
