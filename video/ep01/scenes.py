"""第 1 期场景：你看到的星星，还在吗？每个场景按局部时间 t（秒）绘制一帧。

场景内用 self.at(k) / self.end(k) 取得第 k 句配音的开始/结束时间，让画面和台词对齐。
"""
from functools import lru_cache

import cv2
import numpy as np

from engine.core import (DIM, GREY, H, ORANGE, TEAL, W, WHITE, add, blur, canvas, circle, draw_text, ease_in_out,
                         ease_out, fbm, fill_poly, lerp, line, over, polyline, ramp, rect, round_rect_rgba, smooth,
                         vgradient, window)
from engine.cosmos import (Galaxy, StarField, acacia, affine, dunes, earth_texture, gate_tower, halo,
                           hill_silhouette, milky_way, moon_texture, person_silhouette, rooftops, sphere,
                           star_point, star_texture)
from engine.props import bench, bus, bus_stop
from engine.ui import (Outro, Scene, big_sky, count, fade_text, make_title, night_gradient, pill, pop, sky,
                       title_text, warp)

SIRIUS_C, RED_C, SUN_C = (0.72, 0.84, 1.0), (1.0, 0.45, 0.18), (1.0, 0.85, 0.55)


# ---------- 共享素材（每个进程只生成一次） ----------

@lru_cache(maxsize=1)
def hook_assets():
    mw, dens = milky_way(W, H, seed=5)
    sf = StarField(n=5200, seed=1, density=dens, bright_n=70)
    mask = np.zeros((H, W, 3), np.float32)
    xs, ys = hill_silhouette(mask, 905, 55, seed=2, color=(1, 1, 1))
    person_silhouette(mask, 1310, np.interp(1310, xs, ys) + 5, 165, color=(1, 1, 1))
    a = np.zeros((H, W, 4), np.float32)
    a[..., :3] = (0.010, 0.013, 0.025)
    a[..., 3] = mask[..., 0]
    glow = vgradient((0, 0, 0), (0.05, 0.06, 0.12), curve=4.0)
    return night_gradient() + mw + glow, sf, a


@lru_cache(maxsize=1)
def galaxy():
    return Galaxy()


@lru_cache(maxsize=1)
def galaxy_top_image():
    g = galaxy()
    return g.render(1700, 1700, 850, 850, 600, rot=0.0, exposure=0.6)


@lru_cache(maxsize=1)
def andromeda_image():
    g = Galaxy(seed=23, winding=2.9)
    return g.render(2400, 1500, 1200, 750, 900, rot=0.4, tilt=0.27, tilt_angle=-0.5, exposure=0.55)


# ---------- 1. 开场钩子 ----------

class Hook(Scene):
    tag = False

    def draw(self, t):
        bg, sf, sil = hook_assets()
        z = 1.0 + 0.045 * t / self.d
        M = affine(z, 0, center=(W / 2, H * 0.62), src_center=(W / 2, H * 0.62), shift=(0, 12 * t / self.d))
        img = warp(bg, M) + sf.render(t, M)
        over(img, sil)
        dim = 1 - 0.6 * smooth(ramp(t, self.at(2), self.at(2) + 0.5))
        fade_text(img, "“你今晚看到的星星，", W / 2, 300, 76, t, self.at(0) + 0.1, alpha=dim, name="title")
        fade_text(img, "可能早就不存在了。”", W / 2, 400, 76, t, self.at(0) + 0.9, alpha=dim, name="title")
        t0 = self.at(2) + 0.55
        s, a = pop(t, t0, 0.5)
        if a > 0:
            w, h = int(560 * s), int(140 * s)
            box = round_rect_rgba(w, h, 26, ORANGE, 1.0)
            inner = round_rect_rgba(w - 12, h - 12, 22, (0.03, 0.03, 0.06), 1.0)
            over(img, box, W / 2 - w / 2, 560 - h / 2, a * 0.95)
            over(img, inner, W / 2 - (w - 12) / 2, 560 - (h - 12) / 2, a * 0.85)
            draw_text(img, "只对了一半", W / 2, 556, 96, "title", ORANGE, alpha=a, scale=s)
        return img

    def sfx(self):
        return [(0.2, "shimmer", 0.5), (self.at(2) + 0.55, "swish", 0.9), (self.at(2) + 0.6, "low_ding", 0.8)]


# ---------- 3. 光在赶路 ----------

@lru_cache(maxsize=1)
def streak_texture():
    sf = StarField(W * 2, H, n=5000, seed=7, bright_n=80)
    base = sf.render(0.0)
    k = np.ones((1, 90), np.float32) / 30
    pad = np.concatenate([base[:, -60:], base, base[:, :60]], axis=1)
    return cv2.filter2D(pad, -1, k)[:, 60:-60] * 2.6


class Light(Scene):
    def draw(self, t):
        zoom_out = smooth(ramp(t, self.end(2) - 1.8, self.end(2) + 0.1))
        tex = streak_texture()
        off = int((t * 2600 - zoom_out * 1200) % W)
        img = night_gradient() * 0.6
        streak = np.concatenate([tex[:, off:], tex[:, :off]], axis=1)[:, :W]
        img = img + streak * (1 - zoom_out)
        if zoom_out > 0:
            img += sky(t, zoom=1.6 - 0.6 * zoom_out, grad=False) * zoom_out
        hx = lerp(1260, 980, zoom_out)
        hy = 520
        tail = lerp(1100, 160, zoom_out)
        layer = np.zeros_like(img)
        for i in range(12):
            f = i / 12
            line(layer, (hx - tail * f, hy), (hx - tail * (f + 1 / 12), hy), (0.8, 0.95, 1.0), 6 - 4 * f, alpha=1)
            layer[max(0, hy - 6):hy + 6, int(max(0, hx - tail * (f + 1 / 12))):int(max(0, hx - tail * f))] *= (1 - f) ** 1.5
        img += blur(layer, 3) * 1.4 + layer * 0.6
        star_point(img, hx, hy, lerp(14, 6, zoom_out), (0.8, 0.95, 1.0), k=1.3)
        t2 = self.at(2) + 0.2
        v = count(t, t2, 1.2, 30)
        if t > t2:
            title_text(img, f"≈ {v:.0f} 万公里 / 秒", W / 2, 220, 96, t, t2, color=TEAL)
            fade_text(img, "1 秒钟就能绕地球赤道约 7.5 圈", W / 2, 320, 36, t, t2 + 0.6, color=GREY, name="medium")
        fade_text(img, "你看到的 = 星星过去的样子", W / 2, 220, 64, t, self.at(0) + 0.6, name="title",
                  alpha=1 - smooth(ramp(t, t2 - 0.4, t2)))
        if zoom_out > 0.3:
            fade_text(img, "但宇宙更大", W / 2, 760, 64, t, self.end(2) - 1.4, color=ORANGE, name="title")
        return img

    def sfx(self):
        return [(0.1, "whoosh", 0.8), (self.at(2) + 0.2, "ding", 0.6)]


# ---------- 4. 月光与阳光的时间 ----------

class Timers(Scene):
    EARTH, MOON, SUN = (330, 560, 300), (780, 660, 92), (1720, 560, 440)

    def draw(self, t):
        img = sky(t, gain=0.6, pan=(-t * 6, 0))
        ex, ey, ed = self.EARTH
        mx, my, md = self.MOON
        sx, sy, sd = self.SUN
        halo(img, sx, sy, 900, (1.0, 0.7, 0.3), 0.55)
        over(img, sphere(sd, star_texture("sun"), t * 0.05, emissive=True, limb=0.45), sx - sd / 2, sy - sd / 2)
        halo(img, ex, ey, ed * 0.62, (0.3, 0.55, 1.0), 0.45, power=1.5)
        over(img, sphere(ed, earth_texture(), t * 0.12, light=(0.85, -0.2, 0.5)), ex - ed / 2, ey - ed / 2)
        over(img, sphere(md, moon_texture(), 0.3, light=(0.85, -0.2, 0.5), ambient=0.03), mx - md / 2, my - md / 2)
        draw_text(img, "地球", ex, ey + ed / 2 + 40, 34, "bold", WHITE, shadow=0.6)
        draw_text(img, "月球", mx, my + md / 2 + 36, 34, "bold", WHITE, shadow=0.6)
        draw_text(img, "太阳", sx - 60, sy + sd / 2 + 40, 34, "bold", WHITE, shadow=0.6)
        draw_text(img, "示意图，距离未按比例", 60, 880, 26, "regular", GREY, anchor="lm", alpha=0.8)
        # 月光：按真实时间走 1.3 秒
        t0 = self.at(0) + 0.5
        p = ramp(t, t0, t0 + 1.3)
        a0 = window(t, self.at(0), self.at(1) + 0.2, 0.3, 0.4)
        if a0 > 0:
            px = lerp(mx - md / 2 - 6, ex + ed / 2 + 6, p)
            line(img, (mx - md / 2, my), (px, lerp(my, ey, p)), (0.75, 0.85, 1.0), 3, alpha=0.7 * a0)
            if p < 1:
                star_point(img, px, lerp(my, ey, p), 5, (0.8, 0.9, 1.0), k=a0)
            draw_text(img, f"{1.3 * p:.1f} 秒", (mx + ex) / 2, 420, 84, "title", TEAL, alpha=a0, shadow=0.6)
            draw_text(img, "月光到地球", (mx + ex) / 2, 340, 34, "medium", GREY, alpha=a0)
        # 阳光：8 分 20 秒（加速显示）
        t1 = self.at(1) + 0.3
        q = ramp(t, t1, t1 + 1.8)
        a1 = smooth(ramp(t, self.at(1), self.at(1) + 0.4))
        if a1 > 0:
            qx = lerp(sx - sd / 2 - 6, ex + ed / 2 + 6, q)
            line(img, (sx - sd / 2, sy), (qx, sy), (1.0, 0.85, 0.5), 3, alpha=0.7 * a1)
            if q < 1:
                star_point(img, qx, sy, 6, (1.0, 0.85, 0.5), k=a1)
            secs = int(500 * q)
            draw_text(img, f"{secs // 60} 分 {secs % 60:02d} 秒", 1180, 300, 84, "title", ORANGE, alpha=a1, shadow=0.6)
            draw_text(img, "阳光到地球（加速显示）", 1180, 220, 34, "medium", GREY, alpha=a1)
        return img

    def sfx(self):
        return [(self.at(0) + 1.8, "ding", 0.6), (self.at(1) + 2.1, "ding", 0.6)]


# ---------- 5. 天狼星 ----------

class Sirius(Scene):
    def draw(self, t):
        z = 1.0 + 0.06 * t / self.d
        img = sky(t, zoom=z, focus=(1240, 430))
        tw = 1 + 0.08 * np.sin(t * 7) + 0.05 * np.sin(t * 11.3)
        star_point(img, 1240, 430, 13, SIRIUS_C, k=1.4 * tw, spikes=0.9, spike_len=260)
        fade_text(img, "天狼星", 1240, 600, 52, t, 0.4, name="title")
        fade_text(img, "夜空中最亮的恒星", 1240, 660, 30, t, 0.7, color=GREY, name="medium")
        t0 = self.at(0) + 1.6
        title_text(img, "8.6 年", 600, 420, 150, t, t0, color=TEAL)
        fade_text(img, "光从天狼星出发，到达地球", 600, 300, 36, t, t0, color=GREY, name="medium")
        return img

    def sfx(self):
        return [(self.at(0) + 1.6, "ding", 0.6)]


# ---------- 6. 北极星与明朝 ----------

@lru_cache(maxsize=1)
def tower_layers():
    mask = np.zeros((H, W, 3), np.float32)
    gate_tower(mask, 960, 1010, 560, color=(1, 1, 1))
    rgba = np.zeros((H, W, 4), np.float32)
    rgba[..., :3] = (0.008, 0.01, 0.02)
    rgba[..., 3] = mask[..., 0]
    win = np.zeros((H, W, 3), np.float32)
    gate_tower(win, 960, 1010, 560, color=(0, 0, 0), windows=1.0)
    win = np.clip(win, 0, None)
    return rgba, win


class Polaris(Scene):
    P = (960, 200)

    def draw(self, t):
        img = vgradient((0.004, 0.008, 0.03), (0.08, 0.09, 0.18), curve=1.4)
        sf = big_sky()
        M = affine(1.0, t * 1.2, center=self.P, src_center=(self.P[0] + 300, self.P[1] + 200))
        img += sf.render(t, M)
        rgba, win = tower_layers()
        over(img, rgba)
        flick = 0.85 + 0.1 * np.sin(t * 5.3) + 0.05 * np.sin(t * 13)
        img += win * 0.55 * flick + blur(win, 14) * 0.9 * flick
        star_point(img, *self.P, 9, (0.95, 0.95, 1.0), k=1.3, spikes=0.6, spike_len=150)
        fade_text(img, "北极星", self.P[0] + 70, self.P[1] - 12, 44, t, 0.4, name="title", anchor="lm")
        fade_text(img, "距离 400 多光年", self.P[0] + 70, self.P[1] + 40, 30, t, 0.8, color=GREY, name="medium", anchor="lm")
        t0 = self.at(1) + 1.6
        title_text(img, "光出发时：约 1580 年前后", W / 2, 400, 80, t, t0, color=WHITE)
        title_text(img, "明朝 · 万历年间", W / 2, 500, 72, t, t0 + 0.5, color=TEAL)
        return img

    def sfx(self):
        return [(self.at(1) + 1.6, "low_ding", 0.7)]


# ---------- 7. 星空是一叠旧照片 ----------

class Photos(Scene):
    ITEMS = [((330, 430), "月亮", "1.3 秒前", "moon"),
             ((700, 640), "天狼星", "8.6 年前", SIRIUS_C),
             ((1000, 380), "织女星", "25 年前", (0.8, 0.88, 1.0)),
             ((1310, 640), "北极星", "400 多年前", (0.95, 0.95, 1.0)),
             ((1610, 420), "参宿四", "500 多年前", RED_C)]

    def draw(self, t):
        img = sky(t, zoom=1.0 + 0.03 * t / self.d, gain=0.9)
        fade_text(img, "星空 = 一叠「旧照片」", W / 2, 170, 70, t, 0.3, name="title")
        for i, ((x, y), name, when, col) in enumerate(self.ITEMS):
            t0 = 0.5 + i * 0.7
            s, a = pop(t, t0, 0.4)
            if a <= 0:
                continue
            fw, fh = int(240 * s), int(300 * s)
            frame = round_rect_rgba(fw, fh, 10, (0.95, 0.95, 0.92), 1.0)
            inner = round_rect_rgba(fw - 20, int(fh * 0.62), 4, (0.01, 0.015, 0.04), 1.0)
            over(img, frame, x - fw / 2, y - fh / 2, a * 0.95)
            over(img, inner, x - (fw - 20) / 2, y - fh / 2 + 10, a)
            cy = y - fh / 2 + 10 + fh * 0.31
            if col == "moon":
                md = int(80 * s)
                over(img, sphere(md, moon_texture(), 0.3, light=(0.6, -0.3, 0.75)), x - md / 2, cy - md / 2, a)
            else:
                star_point(img, x, cy, 7 * s, col, k=a, spikes=0.5, spike_len=int(50 * s))
            draw_text(img, name, x, y + fh * 0.25, int(34 * s), "bold", (0.1, 0.1, 0.14), alpha=a)
            draw_text(img, when, x, y + fh * 0.385, int(28 * s), "bold", (0.85, 0.35, 0.1), alpha=a)
        return img

    def sfx(self):
        return [(0.5 + i * 0.7, "click", 0.9) for i in range(5)]


# ---------- 8. 划掉"早就不存在了" ----------

class Strike(Scene):
    def draw(self, t):
        img = sky(t, gain=0.4) * 0.7
        fade_text(img, "问题出在哪？", W / 2, 330, 46, t, 0.2, color=GREY, name="medium")
        a = 1 - 0.45 * smooth(ramp(t, self.at(1) + 1.6, self.at(1) + 2.0))
        fade_text(img, "早就不存在了", W / 2, 500, 150, t, self.at(1) - 0.2, name="title", alpha=a)
        p = ease_out(ramp(t, self.at(1) + 1.2, self.at(1) + 1.55))
        if p > 0:
            x0, x1 = W / 2 - 470, W / 2 + 470
            line(img, (x0, 505), (lerp(x0, x1, p), 505 - 8 * p), ORANGE, 14)
        return img

    def sfx(self):
        return [(self.at(1) + 1.2, "swish", 1.0)]


# ---------- 9. 恒星的寿命 ----------

class Lifetimes(Scene):
    def draw(self, t):
        img = sky(t, gain=0.5)
        sx, sy, sd = 520, 410, 250
        bx, by, bd = 1380, 400, 360
        halo(img, sx, sy, 520, (1.0, 0.75, 0.35), 0.45)
        over(img, sphere(sd, star_texture("sun"), t * 0.08, emissive=True, limb=0.45), sx - sd / 2, sy - sd / 2)
        ab = smooth(ramp(t, self.at(2) - 0.3, self.at(2) + 0.4)) * 0.75 + 0.25
        halo(img, bx, by, 760, (0.45, 0.6, 1.0), 0.6 * ab)
        over(img, sphere(bd, star_texture("blue"), t * 0.15, emissive=True, limb=0.5), bx - bd / 2, by - bd / 2, ab)
        title_text(img, "恒星的寿命：几百万年 ~ 上百亿年", W / 2, 110, 60, t, 0.3)
        draw_text(img, "太阳", sx, 600, 40, "title", WHITE, shadow=0.6)
        draw_text(img, "大质量恒星", bx, 640, 40, "title", WHITE, alpha=ab, shadow=0.6)
        # 太阳的寿命条：总长约 100 亿年
        x0, x1, y = 250, 790, 720
        a1 = smooth(ramp(t, self.at(1) - 0.2, self.at(1) + 0.3))
        if a1 > 0:
            rect(img, x0, y - 14, x1, y + 14, (0.15, 0.17, 0.22), a1)
            p = ease_out(ramp(t, self.at(1) + 0.2, self.at(1) + 1.4))
            rect(img, x0, y - 14, x0 + (x1 - x0) * 0.46 * p, y + 14, TEAL, a1)
            draw_text(img, "已活 46 亿年", x0, y + 50, 28, "bold", TEAL, alpha=a1, anchor="lm")
            q = smooth(ramp(t, self.at(1) + 1.6, self.at(1) + 2.1))
            draw_text(img, "还能活约 50 亿年", x1, y + 50, 28, "bold", GREY, alpha=q, anchor="rm")
        a2 = smooth(ramp(t, self.at(2) + 0.6, self.at(2) + 1.1))
        if a2 > 0:
            rect(img, bx - 6, y - 14 + 40, bx + 6, y + 14 + 40, ORANGE, a2)
            draw_text(img, "只有几百万年", bx, y + 95, 32, "bold", ORANGE, alpha=a2)
        return img

    def sfx(self):
        return [(self.at(1) + 0.2, "whoosh", 0.4), (self.at(2) + 0.6, "low_ding", 0.6)]


# ---------- 10. 肉眼可见范围在银河系里 ----------

class Range(Scene):
    def draw(self, t):
        img = canvas((0.0, 0.0, 0.012))
        img += sky(t, gain=0.4, grad=False)
        G = galaxy_top_image()
        rot = 8 + t * 1.5
        z = 0.72 + 0.04 * t / self.d
        M = affine(z, rot, center=(W / 2, 480), src_center=(850, 850))
        img += warp(G, M)
        ang = np.deg2rad(-rot) + 0.9
        r = 600 * 0.55 * z
        px, py = W / 2 + r * np.cos(ang), 480 + r * np.sin(ang)
        t0 = self.at(0) + 0.8
        a = smooth(ramp(t, t0, t0 + 0.4))
        if a > 0:
            pulse = 1 + 0.25 * np.sin((t - t0) * 6)
            circle(img, px, py, 22 * pulse, TEAL, thickness=3, alpha=a)
            circle(img, px, py, 4, TEAL, alpha=a)
            line(img, (px + 22, py - 22), (px + 140, py - 120), TEAL, 2, alpha=a)
            draw_text(img, "肉眼能看到的星星", px + 150, py - 150, 36, "bold", WHITE, alpha=a, anchor="lm", shadow=0.8)
            draw_text(img, "几乎都在这个小圈里（几千光年）", px + 150, py - 104, 28, "medium", TEAL, alpha=a, anchor="lm", shadow=0.8)
        b = smooth(ramp(t, t0 + 1.0, t0 + 1.5))
        if b > 0:
            y = 900
            R = 600 * 1.2 * z
            line(img, (W / 2 - R, y), (W / 2 + R, y), GREY, 2, alpha=b)
            line(img, (W / 2 - R, y - 10), (W / 2 - R, y + 10), GREY, 2, alpha=b)
            line(img, (W / 2 + R, y - 10), (W / 2 + R, y + 10), GREY, 2, alpha=b)
            pill(img, "银河系直径约 10 万光年（示意）", W / 2, y, 26, WHITE, (0.03, 0.04, 0.08), alpha=b, name="medium")
        return img

    def sfx(self):
        return [(self.at(0) + 0.8, "low_ding", 0.6)]


# ---------- 11. 一生 80 年 vs 2 分钟 ----------

class Analogy(Scene):
    def draw(self, t):
        img = sky(t, gain=0.35) * 0.8
        title_text(img, "把恒星的一生，压缩成人的 80 年", W / 2, 170, 64, t, self.at(0) + 0.6)
        x0, x1, y = 220, 1640, 470
        a = smooth(ramp(t, self.at(0) + 1.2, self.at(0) + 1.6))
        if a > 0:
            rect(img, x0, y - 10, x1, y + 10, (0.16, 0.18, 0.24), a)
            p = ease_in_out(ramp(t, self.at(0) + 1.6, self.at(0) + 3.6))
            rect(img, x0, y - 10, x0 + (x1 - x0) * p, y + 10, TEAL, a)
            for k, age in enumerate((0, 20, 40, 60, 80)):
                x = lerp(x0, x1, k / 4)
                line(img, (x, y + 16), (x, y + 30), GREY, 2, alpha=a)
                draw_text(img, f"{age} 岁", x, y + 58, 30, "medium", GREY, alpha=a)
        b = smooth(ramp(t, self.at(1), self.at(1) + 0.5))
        if b > 0:
            cx, cy, R = x1, y, 120
            circle(img, cx, cy, R * b, (0.02, 0.03, 0.06), alpha=0.92 * b)
            circle(img, cx, cy, R * b, WHITE, thickness=4, alpha=b)
            line(img, (cx + R * 0.7 * b, cy + R * 0.7 * b), (cx + R * 1.15 * b, cy + R * 1.15 * b), WHITE, 9, alpha=b)
            rect(img, cx - 80 * b, cy - 9, cx - 4 * b, cy + 9, TEAL, b)
            rect(img, cx - 4 * b, cy - 9, cx + 4 * b, cy + 9, ORANGE, b)
            draw_text(img, "星光的旅途", cx - 260, y - 150, 34, "medium", GREY, alpha=b)
            title_text(img, "≈ 2 分钟", cx - 260, y - 90, 72, t, self.at(1) + 1.0, color=ORANGE)
        c = self.at(2) + 0.3
        fade_text(img, "两分钟前还好好的", W / 2 - 330, 760, 52, t, c, name="title")
        fade_text(img, "→", W / 2, 760, 52, t, c + 0.8, color=GREY, name="title")
        fade_text(img, "现在大概率也还好好的", W / 2 + 360, 760, 52, t, c + 1.4, color=TEAL, name="title")
        return img

    def sfx(self):
        return [(self.at(1) + 1.0, "ding", 0.6)]


# ---------- 12. 沙漠星空：2500 颗 ----------

@lru_cache(maxsize=1)
def desert_assets():
    mw, _ = milky_way(W, H, seed=9, angle=-62, width=0.13, center=(0.42, 0.5), strength=0.3)
    base = vgradient((0.004, 0.006, 0.02), (0.10, 0.08, 0.13), curve=2.2) + mw
    mask = np.zeros((H, W, 3), np.float32)
    dunes(mask, 880)
    return base, mask


class Desert(Scene):
    def draw(self, t):
        base, dune = desert_assets()
        M = affine(1.0, t * 0.9, center=(W / 2, H / 2), src_center=(W / 2 + 300, H / 2 + 200))
        img = base * 1.3 + big_sky().render(t, M) * 1.5
        m = dune[..., 0] > 0
        img[m] = dune[m]
        t0 = self.at(1) + 1.6
        if t > t0 - 0.1:
            v = count(t, t0, 1.4, 2500)
            title_text(img, f"≈ {v:,.0f} 颗", W / 2, 300, 120, t, t0, color=TEAL)
            fade_text(img, "晴朗暗夜 · 同一时刻 · 肉眼可见", W / 2, 410, 34, t, t0 + 0.4, color=GREY, name="medium")
        fade_text(img, "天文学家算了一笔账", W / 2, 300, 64, t, self.at(0) + 0.2, name="title",
                  alpha=1 - smooth(ramp(t, t0 - 0.6, t0 - 0.1)))
        return img

    def sfx(self):
        return [(self.at(1) + 1.6, "whoosh", 0.4), (self.at(1) + 3.0, "ding", 0.5)]


# ---------- 13. 2500 个点，1 万年才熄灭 1 颗 ----------

@lru_cache(maxsize=1)
def dot_layers():
    n = 50
    sp = 12.6
    x0, y0 = W / 2 - sp * (n - 1) / 2, 470 - sp * (n - 1) / 2
    rng = np.random.default_rng(4)
    layers = []
    pos = [(x0 + (i % n) * sp, y0 + (i // n) * sp) for i in range(n * n)]
    group = rng.integers(0, 3, n * n)
    cols = np.array([(0.85, 0.92, 1.0), (1.0, 1.0, 1.0), (1.0, 0.92, 0.8)], np.float32)[rng.integers(0, 3, n * n)]
    for g in range(3):
        L = np.zeros((H, W, 3), np.float32)
        for i in np.where(group == g)[0]:
            x, y = pos[i]
            L[int(y), int(x)] += cols[i] * 6
        layers.append(blur(L, 1.3) * 1.0 + blur(L, 4) * 0.8)
    return layers, pos


class Dots(Scene):
    PICK = 1337

    def draw(self, t):
        img = canvas((0.004, 0.006, 0.02))
        layers, pos = dot_layers()
        gain = 1.0 + 0.35 * smooth(ramp(t, self.at(1) + 0.8, self.at(1) + 1.6))
        for g, L in enumerate(layers):
            img += L * (0.8 + 0.2 * np.sin(t * (1.5 + g * 0.7) + g * 2)) * gain
        x, y = pos[self.PICK]
        t0 = self.at(0) + 1.8
        a = smooth(ramp(t, t0, t0 + 0.4)) * (1 - smooth(ramp(t, self.at(1) + 0.5, self.at(1) + 1.0)))
        if a > 0:
            circle(img, x, y, 14, ORANGE, thickness=3, alpha=a)
            fade = smooth(ramp(t, t0 + 1.0, t0 + 3.2))
            circle(img, x, y, 6, (0.004, 0.006, 0.02), alpha=fade)
            line(img, (x + 16, y - 10), (1330, 300), ORANGE, 2, alpha=a)
        fade_text(img, "平均约 1 万年", 1340, 280, 54, t, t0 + 0.4, color=ORANGE, name="title", anchor="lm")
        fade_text(img, "才有 1 颗走到尽头", 1340, 350, 40, t, t0 + 0.8, color=WHITE, name="medium", anchor="lm")
        fade_text(img, "≈ 2500 颗", 560, 280, 54, t, 0.3, color=GREY, name="title", anchor="rm")
        fade_text(img, "几乎全都", 560, 520, 70, t, self.at(1) + 1.0, color=TEAL, name="title", anchor="rm")
        fade_text(img, "还活着", 560, 610, 70, t, self.at(1) + 1.3, color=TEAL, name="title", anchor="rm")
        return img

    def sfx(self):
        return [(self.at(0) + 1.8, "low_ding", 0.6), (self.at(1) + 1.0, "shimmer", 0.5)]


# ---------- 14. 猎户座与参宿四 ----------

ORION = {"参宿四": (700, 300, 13, RED_C), "参宿五": (1160, 330, 8, (0.8, 0.88, 1.0)), "觜宿一": (945, 215, 5, (0.85, 0.9, 1.0)),
         "参宿一": (860, 575, 7, (0.8, 0.88, 1.0)), "参宿二": (940, 545, 7.5, (0.8, 0.88, 1.0)),
         "参宿三": (1020, 515, 7, (0.8, 0.88, 1.0)), "参宿六": (790, 830, 7, (0.8, 0.88, 1.0)),
         "参宿七": (1200, 800, 11, (0.7, 0.82, 1.0))}
ORION_LINES = [("参宿四", "觜宿一"), ("觜宿一", "参宿五"), ("参宿四", "参宿一"), ("参宿五", "参宿三"), ("参宿一", "参宿二"),
               ("参宿二", "参宿三"), ("参宿一", "参宿六"), ("参宿三", "参宿七")]


class Orion(Scene):
    def draw(self, t):
        F = ORION["参宿四"][:2]
        z = 1.0 + 1.6 * ease_in_out(ramp(t, self.d - 1.6, self.d + 0.3)) + 0.03 * t / self.d
        cam = lambda p: (F[0] + (p[0] - F[0]) * z, F[1] + (p[1] - F[1]) * z)
        img = sky(t, zoom=z, focus=F)
        la = smooth(ramp(t, 0.3, 1.6)) * (1 - smooth(ramp(t, self.d - 1.4, self.d - 0.6)))
        for a, b in ORION_LINES:
            line(img, cam(ORION[a][:2]), cam(ORION[b][:2]), (0.55, 0.7, 0.9), 2, alpha=0.45 * la)
        for name, (x, y, r, c) in ORION.items():
            k = 1.2 if name != "参宿四" else 1.5 + 0.15 * np.sin(t * 3)
            star_point(img, *cam((x, y)), r * z ** 0.5, c, k=k)
        bx, by = cam(F)
        t1 = self.at(1)
        a = smooth(ramp(t, t1, t1 + 0.4)) * (1 - smooth(ramp(t, self.d - 1.2, self.d - 0.6)))
        if a > 0:
            circle(img, bx, by, 44 + 4 * np.sin(t * 4), ORANGE, thickness=3, alpha=a)
            draw_text(img, "参宿四", bx - 70, by - 20, 56, "title", ORANGE, alpha=a, anchor="rm", shadow=0.8)
            draw_text(img, "红超巨星", bx - 70, by + 40, 32, "medium", WHITE, alpha=a, anchor="rm", shadow=0.8)
        fade_text(img, "猎户座", 1160, 440, 44, t, 0.6, color=GREY, name="title", alpha=la)
        t2 = self.at(2) + 0.2
        b = smooth(ramp(t, t2, t2 + 0.4)) * (1 - smooth(ramp(t, self.d - 1.2, self.d - 0.6)))
        pill(img, "距离：500 多光年", 1450, 200, 32, WHITE, (0.1, 0.12, 0.18), alpha=b)
        pill(img, "已步入晚年", 1450, 270, 32, (0.1, 0.05, 0.02), ORANGE, alpha=smooth(ramp(t, t2 + 1.4, t2 + 1.8)) * (b > 0))
        return img

    def sfx(self):
        return [(self.at(1), "low_ding", 0.6), (self.d - 1.6, "whoosh", 0.6)]


# ---------- 15. 大变暗 ----------

def dimming_curve(m):
    """m：从 2019 年 9 月起的月数。返回相对亮度（示意）。"""
    w = 2.3 if m < 5.2 else 1.5
    return 1 - 0.64 * np.exp(-((m - 5.2) / w) ** 2)


class Dimming(Scene):
    def draw(self, t):
        img = sky(t, gain=0.35)
        x0, x1, y0, y1 = 170, 1080, 230, 720
        rect(img, x0 - 40, y0 - 70, x1 + 40, y1 + 90, (0.02, 0.03, 0.06), 0.75)
        line(img, (x0, y1), (x1, y1), GREY, 2)
        line(img, (x0, y0), (x0, y1), GREY, 2)
        draw_text(img, "亮度", x0 - 10, y0 - 34, 28, "medium", GREY, anchor="lm")
        for m, lab in ((1, "2019.10"), (5, "2020.02"), (9, "2020.06")):
            x = lerp(x0, x1, m / 9.5)
            line(img, (x, y1), (x, y1 + 10), GREY, 2)
            draw_text(img, lab, x, y1 + 40, 26, "medium", GREY)
        draw_text(img, "示意曲线", x1, y0 - 34, 24, "regular", DIM, anchor="rm")
        p = ramp(t, self.at(0) + 0.3, self.at(0) + 5.5)
        ms = np.linspace(0, 9.5 * p, max(2, int(200 * p)))
        pts = [(lerp(x0, x1, m / 9.5), lerp(y1, y0, dimming_curve(m) * 0.9)) for m in ms]
        if len(pts) > 1:
            polyline(img, pts, ORANGE, 5)
            star_point(img, *pts[-1], 4, ORANGE, k=1.0)
        a = smooth(ramp(t, self.at(0) + 3.6, self.at(0) + 4.1))
        if a > 0:
            mx, my = lerp(x0, x1, 5.2 / 9.5), lerp(y1, y0, dimming_curve(5.2) * 0.9)
            circle(img, mx, my, 12, WHITE, thickness=3, alpha=a)
            draw_text(img, "2020 年 2 月", mx, my + 60, 32, "bold", WHITE, alpha=a, shadow=0.8)
            draw_text(img, "亮度只剩约 1/3", mx, my + 105, 32, "bold", ORANGE, alpha=a, shadow=0.8)
        cur = dimming_curve(9.5 * p)
        sx, sy, sd = 1500, 450, 380
        halo(img, sx, sy, 760 * (0.6 + 0.4 * cur), (1.0, 0.35, 0.1), 0.55 * cur)
        st = sphere(sd, star_texture("red"), t * 0.04, emissive=True, limb=0.85)
        st[..., :3] *= 0.35 + 0.65 * cur
        over(img, st, sx - sd / 2, sy - sd / 2)
        draw_text(img, "参宿四", sx, sy + sd / 2 + 50, 40, "title", WHITE, shadow=0.8)
        fade_text(img, "它要爆炸了吗？", 1500, 130, 56, t, self.at(1) + 0.2, color=ORANGE, name="title")
        return img

    def sfx(self):
        return [(self.at(0) + 3.6, "low_ding", 0.6)]


# ---------- 16. 尘埃 ----------

@lru_cache(maxsize=1)
def dust_noise():
    return fbm(560, 560, 77, 6, base=3)


class Dust(Scene):
    def draw(self, t):
        img = sky(t, gain=0.4)
        sx, sy, sd = W / 2, 470, 560
        halo(img, sx, sy, 1100, (1.0, 0.35, 0.1), 0.5)
        st = sphere(sd, star_texture("red"), t * 0.05, emissive=True, limb=0.85)
        n = dust_noise()
        yy, xx = np.mgrid[0:sd, 0:sd].astype(np.float32) / sd
        p = ease_out(ramp(t, 0.6, 3.6))
        region = np.clip((yy - 0.62 + 0.45 * p) * 2.2 + (0.5 - xx) * 0.6, 0, 1)
        dust = np.clip(region * np.clip((n - 0.25) * 1.6, 0, 1) * p, 0, 0.7)
        st[..., :3] *= (1 - dust)[..., None]
        over(img, st, sx - sd / 2, sy - sd / 2)
        haze = np.zeros((sd + 200, sd + 200, 3), np.float32)
        haze[100:-100, 100:-100] = dust[..., None] * np.array((0.18, 0.09, 0.05))
        haze = blur(haze, 30)
        add(img, -haze, sx - sd / 2 - 100, sy - sd / 2 - 100, 1.0)
        a = smooth(ramp(t, 2.0, 2.5))
        if a > 0:
            line(img, (sx - 120, sy + 170), (sx - 420, sy + 300), WHITE, 2, alpha=a)
            draw_text(img, "尘埃云", sx - 430, sy + 300, 40, "title", WHITE, alpha=a, anchor="rm", shadow=0.8)
        title_text(img, "是尘埃，不是爆炸", W / 2, 130, 72, t, self.at(0) + 2.4, color=TEAL)
        return img

    def sfx(self):
        return [(0.6, "whoosh", 0.4), (self.at(0) + 2.4, "ding", 0.5)]


# ---------- 17. 超新星与白天的光点 ----------

@lru_cache(maxsize=1)
def ray_noise():
    rng = np.random.default_rng(5)
    v = cv2.GaussianBlur(rng.random((1, 64)).astype(np.float32), (0, 0), 1.2)[0]
    v = (v - v.min()) / (v.max() - v.min()) * 2 - 1
    v[-1] = v[0]
    return v


@lru_cache(maxsize=1)
def day_sky():
    img = vgradient((0.16, 0.38, 0.78), (0.62, 0.78, 0.95), curve=1.2)
    rooftops(img, 960, color=(0.06, 0.09, 0.16))
    return img


class Supernova(Scene):
    def draw(self, t):
        boom = 0.75
        tday = 2.3
        if t < tday + 0.6:
            img = sky(t, gain=0.4)
            sx, sy = W / 2, 470
            if t < boom:
                sd = int(560 * (1 - 0.6 * ease_in_out(t / boom)))
                halo(img, sx, sy, 1100 * (1 - 0.4 * t / boom), (1.0, 0.4, 0.15), 0.5 + t)
                st = sphere(max(sd, 20), star_texture("red"), t * 0.05, emissive=True, limb=0.85)
                st[..., :3] *= 1 + 2 * (t / boom) ** 3
                over(img, st, sx - sd / 2, sy - sd / 2)
            else:
                u = t - boom
                flash = np.exp(-u * 3.2)
                img += np.array((1.0, 0.96, 0.9), np.float32) * flash * 1.3
                R = 80 + 900 * (1 - np.exp(-u * 1.4))
                yy, xx = np.ogrid[0:H, 0:W]
                d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2).astype(np.float32)
                ring = np.exp(-((d - R) / (40 + 60 * u)) ** 2) * np.exp(-u * 0.7)
                ang = np.arctan2(yy - sy, xx - sx)
                rays = 0.7 + 0.3 * np.interp(ang, np.linspace(-np.pi, np.pi, 64), ray_noise())
                img += (ring * rays)[..., None] * np.array((0.55, 0.75, 1.0), np.float32) * 1.4
                inner = np.exp(-(d / (R * 0.55)) ** 2) * np.exp(-u * 0.9)
                img += inner[..., None] * np.array((1.0, 0.55, 0.3), np.float32) * 0.8
                star_point(img, sx, sy, 22, (1.0, 0.95, 0.9), k=1.5 * np.exp(-u * 0.6) + 0.5)
            if t > tday:
                img *= 1 - smooth((t - tday) / 0.6)
                img += self.day(t) * smooth((t - tday) / 0.6)
            return img
        return self.day(t)

    def day(self, t):
        img = day_sky().copy()
        mx, my, md = 1360, 300, 130
        mm = sphere(md, moon_texture(), 0.0, light=(1, 0, 0.02), ambient=0.0)
        mm[..., 3] *= np.clip(mm[..., :3].mean(-1) * 3, 0, 1)
        mm[..., :3] = np.clip(mm[..., :3] * 1.6 + 0.15, 0, 1)
        over(img, mm, mx - md / 2, my - md / 2, 0.95)
        sx, sy = 640, 250
        halo(img, sx, sy, 300, (1, 1, 1), 0.5)
        star_point(img, sx, sy, 12, (1.0, 1.0, 1.0), k=1.6, spikes=0.6, spike_len=160)
        a = smooth(ramp(t, 2.8, 3.3))
        draw_text(img, "超新星", sx, sy + 90, 40, "title", WHITE, alpha=a, shadow=0.9)
        draw_text(img, "半个月亮", mx, my + 110, 36, "title", WHITE, alpha=a, shadow=0.9)
        draw_text(img, "≈", (sx + mx) / 2, (sy + my) / 2, 80, "title", WHITE, alpha=a, shadow=0.9)
        fade_text(img, "白天也能看见", W / 2, 470, 56, t, 3.6, name="title",
                  alpha=1 - smooth(ramp(t, self.at(1) - 0.3, self.at(1))))
        t1 = self.at(1) + 0.3
        b = smooth(ramp(t, t1, t1 + 0.4))
        if b > 0:
            pill(img, "未来 10 万年内的某一天", W / 2, 520, 64, TEAL, (0.02, 0.04, 0.08), alpha=b, name="title", pad=(40, 18))
        return img

    def sfx(self):
        return [(0.75, "boom", 1.0), (self.at(1) + 0.3, "ding", 0.5)]


# ---------- 18. 光还在路上 ----------

class Wave(Scene):
    def draw(self, t):
        img = sky(t, gain=0.6) * 0.8
        sx, sy = 260, 540
        R = 150 + 270 * t
        yy, xx = np.ogrid[0:H, 0:W]
        d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2).astype(np.float32)
        ring = np.exp(-((d - R) / 22) ** 2) * 0.9 + np.exp(-((d - R) / 90) ** 2) * 0.25
        inside = (d < R) * np.exp(-(R - d) / 400) * 0.08
        img += (ring + inside)[..., None] * np.array((0.6, 0.8, 1.0), np.float32)
        star_point(img, sx, sy, 10, (1.0, 0.85, 0.75), k=1.0)
        ex, ey, ed = 1660, 540, 74
        halo(img, ex, ey, 60, (0.3, 0.55, 1.0), 0.5, power=1.5)
        over(img, sphere(ed, earth_texture(), t * 0.2, light=(-0.8, -0.2, 0.55)), ex - ed / 2, ey - ed / 2)
        draw_text(img, "地球", ex, ey + 80, 32, "bold", WHITE, shadow=0.8)
        draw_text(img, "参宿四", sx, sy + 70, 32, "bold", WHITE, shadow=0.8, alpha=smooth(ramp(t, 0.3, 0.8)))
        fade_text(img, "光还在路上……", W / 2 + 120, 230, 72, t, self.at(0) + 2.0, name="title", color=TEAL)
        return img

    def sfx(self):
        return [(0.2, "whoosh", 0.5)]


# ---------- 19. 仙女座星系 ----------

class Andromeda(Scene):
    def draw(self, t):
        img = canvas((0, 0, 0.01)) + sky(t, gain=0.7, grad=False)
        A = andromeda_image()
        z = 0.9 + 0.12 * t / self.d
        M = affine(z, -3 + t * 0.4, center=(W / 2 + 120, 470), src_center=(1200, 750))
        img += warp(A, M)
        fade_text(img, "仙女座星系", 330, 220, 64, t, self.at(0) + 1.8, name="title")
        fade_text(img, "通常被认为是肉眼可见的最远天体", 330, 290, 30, t, self.at(1) + 0.4, color=GREY, name="medium")
        t1 = self.at(1) + 2.6
        if t > t1 - 0.1:
            v = count(t, t1, 1.4, 250)
            title_text(img, f"{v:.0f} 万光年", W / 2 + 330, 800, 110, t, t1, color=TEAL)
        return img

    def sfx(self):
        return [(0.3, "shimmer", 0.5), (self.at(1) + 2.6, "ding", 0.5)]


# ---------- 20. 远古荒原 ----------

@lru_cache(maxsize=1)
def savanna_assets():
    mw, _ = milky_way(W, H, seed=12, angle=-15, width=0.12, center=(0.5, 0.36), strength=0.3)
    base = vgradient((0.004, 0.006, 0.02), (0.16, 0.10, 0.10), curve=2.6) + mw
    mask = np.zeros((H, W, 3), np.float32)
    xs = np.linspace(-10, W + 10, 50)
    fill_poly(mask, list(zip(xs, 900 + 8 * np.sin(xs / 300))) + [(W + 10, H + 10), (-10, H + 10)], (1, 1, 1))
    for i, (x, h) in enumerate(((300, 230), (760, 160), (1450, 270), (1780, 140))):
        acacia(mask, x, 905, h, color=(1, 1, 1), seed=i)
    rgba = np.zeros((H, W, 4), np.float32)
    rgba[..., :3] = (0.01, 0.008, 0.012)
    rgba[..., 3] = mask[..., 0]
    return base, rgba


class Savanna(Scene):
    def draw(self, t):
        base, sil = savanna_assets()
        M = affine(1.0 + 0.03 * t / self.d, 0, center=(W / 2, H / 2), src_center=(W / 2 + 300, H / 2 + 200))
        img = base + big_sky().render(t, M)
        over(img, sil)
        title_text(img, "250 万年前出发的光", W / 2, 200, 76, t, self.at(0) + 0.6)
        fade_text(img, "那时，地球上还没有现代人类", W / 2, 320, 54, t, self.at(0) + 3.0, color=TEAL, name="title")
        fade_text(img, "（智人大约在 30 万年前才出现）", W / 2, 400, 30, t, self.at(0) + 3.6, color=GREY, name="medium")
        return img


# ---------- 21. 结论 ----------

class Verdict(Scene):
    def draw(self, t):
        img = sky(t, zoom=1.0 + 0.02 * t, gain=0.8) * 0.75
        fade_text(img, "你看到的星光，是它过去的样子", W / 2, 380, 66, t, self.at(1), name="title")
        title_text(img, "但那颗星，大概率还在", W / 2, 520, 100, t, self.at(2), color=TEAL)
        return img

    def sfx(self):
        return [(self.at(2), "shimmer", 0.7)]


# ---------- 22. 互动提问 ----------

class Question(Scene):
    def draw(self, t):
        bg, sf, sil = hook_assets()
        M = affine(1.06, 0, center=(W / 2, H * 0.62), src_center=(W / 2, H * 0.62), shift=(0, 12))
        img = warp(bg, M) + sf.render(t + 20, M)
        over(img, sil)
        fade_text(img, "你见过最美的星空在哪里？", W / 2, 300, 76, t, 0.2, name="title")
        fade_text(img, "评论区告诉我 ↓", W / 2, 410, 46, t, self.at(0) + 2.0, color=TEAL, name="title")
        return img


# ---------- 23. 下期预告 ----------

class Next(Scene):
    def draw(self, t):
        img = vgradient((0.20, 0.36, 0.70), (0.86, 0.66, 0.50), curve=1.2)
        rooftops(img, 900, seed=3, color=(0.10, 0.12, 0.20))
        rect(img, 0, 900, W, H, (0.05, 0.06, 0.09))
        bench(img, 700, 900)
        person_silhouette(img, 560, 900, 230, color=(0.02, 0.025, 0.04), look_up=0.0)
        bus_stop(img, 420, 900, 380)
        pill(img, "下期预告", W / 2, 200, 34, (0.1, 0.04, 0.0), ORANGE, alpha=smooth(ramp(t, 0.05, 0.35)))
        title_text(img, "公交车 10 分钟一班", 1250, 400, 72, t, 0.4)
        title_text(img, "为什么你总要等更久？", 1250, 510, 72, t, 1.6, color=(0.98, 0.98, 1.0))
        bx = lerp(W + 520, W + 520 - 700 * ease_out(ramp(t, 2.6, 4.0)), 1.0)
        bus(img, bx, 900, 520)
        return img

    def sfx(self):
        return [(1.6, "ding", 0.5), (2.6, "whoosh", 0.4)]


SCENE_CLASSES = {"hook": Hook, "title": make_title("第一季 · 反直觉 #01", "你看到的星星，还在吗？"),
                 "light": Light, "timers": Timers, "sirius": Sirius, "polaris": Polaris, "photos": Photos,
                 "strike": Strike, "lifetimes": Lifetimes, "range": Range, "analogy": Analogy, "desert": Desert,
                 "dots": Dots, "orion": Orion, "dimming": Dimming, "dust": Dust, "supernova": Supernova,
                 "wave": Wave, "andromeda": Andromeda, "savanna": Savanna, "verdict": Verdict,
                 "question": Question, "next": Next, "outro": Outro}


def music(spans, total):
    """背景音乐分段：平静 → 参宿四段紧张 → 仙女座之后转为开阔。超新星之后静音一下。"""
    sp = {s.name: s for s in spans}
    sections = [(0, sp["orion"].start, "calm"), (sp["orion"].start, sp["wave"].end, "tension"),
                (sp["andromeda"].start - 0.5, total, "wonder")]
    sup = sp["supernova"]
    mutes = [(sup.start + sup.lines[1][1] + 0.1, sp["wave"].start + 0.4)]
    return sections, mutes
