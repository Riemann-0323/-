"""天文题材的程序化素材：星空、银河、旋涡星系、行星、恒星、超新星、剪影。"""
from functools import lru_cache

import cv2
import numpy as np

from .core import H, W, blur, fbm, fill_poly, circle

STAR_COLORS = np.array([
    (0.70, 0.80, 1.00),   # 蓝白
    (0.88, 0.92, 1.00),
    (1.00, 1.00, 1.00),
    (1.00, 0.95, 0.82),   # 黄白
    (1.00, 0.84, 0.62),   # 橙
], np.float32)
STAR_WEIGHTS = np.array([0.12, 0.25, 0.30, 0.22, 0.11])


def splat(img, x, y, cols):
    """把点按双线性权重累加到图上（亚像素精度）。"""
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    h, w = img.shape[:2]
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        np.add.at(img, (yi[m], xi[m]), cols[m] * wgt[m])


class StarField:
    """一片星空。分成 3 组分别闪烁，渲染时可整体平移/缩放/旋转。"""

    def __init__(self, w=W, h=H, n=2600, seed=0, density=None, gain=1.0, bright_n=60):
        rng = np.random.default_rng(seed)
        if density is None:
            x, y = rng.uniform(0, w, n), rng.uniform(0, h, n)
        else:
            xs, ys = [], []
            while sum(len(a) for a in xs) < n:
                cx, cy = rng.uniform(0, w, n), rng.uniform(0, h, n)
                keep = rng.random(n) < density(cx, cy)
                xs.append(cx[keep]); ys.append(cy[keep])
            x, y = np.concatenate(xs)[:n], np.concatenate(ys)[:n]
        b = 0.05 + 0.9 * rng.random(n) ** 7
        b[:bright_n] = 0.7 + 0.6 * rng.random(bright_n)
        cols = STAR_COLORS[rng.choice(len(STAR_COLORS), n, p=STAR_WEIGHTS)] * (b * gain)[:, None]
        group = rng.integers(0, 3, n)
        self.size = (w, h)
        self.layers = []
        for g in range(3):
            m = group == g
            pts = np.zeros((h, w, 3), np.float32)
            splat(pts, x[m], y[m], cols[m] * 2.6)
            layer = blur(pts, 0.65)
            bm = m & (b > 0.45)
            if bm.any():
                bp = np.zeros((h, w, 3), np.float32)
                splat(bp, x[bm], y[bm], cols[bm])
                layer += blur(bp, 2.2) * 2.5 + blur(bp, 7) * 6
            self.layers.append(layer)
        self.phase = rng.random(3) * 6.28
        self.speed = (1.3, 1.9, 2.6)

    def render(self, t, M=None, out=(W, H), twinkle=0.22):
        img = np.zeros((self.size[1], self.size[0], 3), np.float32)
        for layer, ph, sp in zip(self.layers, self.phase, self.speed):
            img += layer * (1 - twinkle + twinkle * np.sin(t * sp + ph))
        if M is None and self.size == out:
            return img
        if M is None:
            M = np.float32([[1, 0, 0], [0, 1, 0]])
        return cv2.warpAffine(img, np.float32(M), out, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def affine(scale=1.0, angle_deg=0.0, center=(W / 2, H / 2), src_center=None, shift=(0, 0)):
    """绕 src_center 旋转缩放，并把它放到画面 center + shift 处。"""
    if src_center is None:
        src_center = center
    M = cv2.getRotationMatrix2D(tuple(map(float, src_center)), angle_deg, scale)
    M[0, 2] += center[0] - src_center[0] + shift[0]
    M[1, 2] += center[1] - src_center[1] + shift[1]
    return M


def milky_way(w, h, seed=5, angle=-28, width=0.16, center=(0.5, 0.45), strength=0.42):
    """银河带：沿一条斜线的发光云气 + 暗尘埃带。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(angle)
    nx, ny = -np.sin(a), np.cos(a)
    d = ((xx - center[0] * w) * nx + (yy - center[1] * h) * ny) / h
    n1 = fbm(h, w, seed, 7, base=3)
    n2 = fbm(h, w, seed + 1, 7, base=4)
    n3 = fbm(h, w, seed + 2, 5, base=2)
    warp = (n3 - 0.5) * 0.12
    band = np.exp(-((d + warp) / width) ** 2)
    core = np.exp(-((d + warp) / (width * 0.45)) ** 2)
    lum = band * (0.25 + 0.75 * n1 ** 1.6) + core * 0.5 * n1
    dust = core * np.clip((n2 - 0.45) * 3.0, 0, 1)
    cool = np.array((0.55, 0.62, 0.95), np.float32)
    warm = np.array((1.0, 0.86, 0.68), np.float32)
    mix = np.clip(core * 0.8 + (n3 - 0.5), 0, 1)[..., None]
    img = lum[..., None] * (cool * (1 - mix) + warm * mix) * strength
    img *= (1 - 0.85 * dust)[..., None]
    density = lambda x, y: 0.15 + 0.85 * np.exp(-(((x - center[0] * w) * nx + (y - center[1] * h) * ny) / h / width) ** 2)
    return img.astype(np.float32), density


# ---------- 旋涡星系 ----------

class Galaxy:
    """粒子旋涡星系：旋臂 + 核球 + 尘埃带 + 恒星形成区，渲染时做色调映射。"""

    def __init__(self, n=200000, seed=11, arms=2, winding=2.6, spread=0.2, bulge=0.11):
        rng = np.random.default_rng(seed)
        r = np.clip(rng.exponential(0.34, n), 0.03, 1.3)
        arm = rng.integers(0, arms, n)
        in_arm = rng.random(n) < 0.7
        theta = arm * 2 * np.pi / arms + np.log(r + 0.05) * winding
        theta += np.where(in_arm, rng.normal(0, spread, n), rng.uniform(0, 2 * np.pi, n))
        xy = np.stack([r * np.cos(theta), r * np.sin(theta)], 1) + rng.normal(0, 0.02, (n, 2))
        young = (np.clip(r * 1.6 - 0.25, 0, 1) * in_arm)[:, None]
        col = (1 - young) * np.array((1.0, 0.82, 0.6)) + young * np.array((0.6, 0.72, 1.0))
        nb = n // 3
        bxy = rng.normal(0, bulge, (nb, 2)) * [1, 0.9]
        bcol = np.tile(np.array((1.0, 0.84, 0.58)) * 0.45, (nb, 1))
        # 恒星形成区：旋臂上偏粉/蓝的亮结
        nk = 500
        rk = rng.uniform(0.25, 1.0, nk)
        ak = rng.integers(0, arms, nk)
        tk = ak * 2 * np.pi / arms + np.log(rk + 0.05) * winding + rng.normal(0, spread * 0.6, nk)
        kxy = np.stack([rk * np.cos(tk), rk * np.sin(tk)], 1)
        kcol = np.where(rng.random((nk, 1)) < 0.5, (1.0, 0.55, 0.75), (0.7, 0.85, 1.0)) * 18
        self.xy = np.concatenate([xy, bxy, kxy]).astype(np.float32)
        self.col = np.concatenate([col, bcol, kcol]).astype(np.float32)
        self.n = n + nb
        nd = n // 4
        rd = rng.uniform(0.34, 1.0, nd)
        armd = rng.integers(0, arms, nd)
        thd = armd * 2 * np.pi / arms + np.log(rd + 0.05) * winding - 0.2 + rng.normal(0, spread * 0.3, nd)
        keep = rng.random(nd) < np.clip(fbm(64, 64, seed + 5, 4, base=3)[(np.clip(rd * 31 + 32, 0, 63)).astype(int), (np.clip(thd % (2 * np.pi) / (2 * np.pi) * 63, 0, 63)).astype(int)] * 1.6 - 0.3, 0, 1)
        rd, thd = rd[keep], thd[keep]
        self.dust = (np.stack([rd * np.cos(thd), rd * np.sin(thd)], 1) + rng.normal(0, 0.012, (len(rd), 2))).astype(np.float32)

    def render(self, w, h, cx, cy, scale, rot=0.0, tilt=1.0, tilt_angle=0.0, exposure=1.0, dust=0.8):
        def project(p):
            c, s = np.cos(rot), np.sin(rot)
            x = p[:, 0] * c - p[:, 1] * s
            y = (p[:, 0] * s + p[:, 1] * c) * tilt
            c2, s2 = np.cos(tilt_angle), np.sin(tilt_angle)
            return cx + (x * c2 - y * s2) * scale, cy + (x * s2 + y * c2) * scale
        acc = np.zeros((h, w, 3), np.float32)
        x, y = project(self.xy)
        splat(acc, x, y, self.col)
        lum = blur(acc, 1.1) * 0.55 + blur(acc, scale * 0.012 + 1.5) * 0.9 + blur(acc, scale * 0.05 + 3) * 1.0
        norm = (scale ** 2 * tilt) / self.n * 3.2
        if dust > 0:
            dm = np.zeros((h, w), np.float32)
            dx, dy = project(self.dust)
            splat(dm[..., None], dx, dy, np.ones((len(dx), 1), np.float32))
            dm = blur(dm, scale * 0.004 + 0.8) * (scale ** 2 * tilt) / len(dx) * 0.5
            lum *= np.exp(-dm * dust)[..., None]
        return (1 - np.exp(-lum * norm * exposure)).astype(np.float32)


# ---------- 球体（行星、恒星） ----------

@lru_cache(maxsize=16)
def _sphere_grid(d):
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
    nx = (xx + 0.5 - d / 2) / (d / 2)
    ny = (yy + 0.5 - d / 2) / (d / 2)
    r2 = nx ** 2 + ny ** 2
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    alpha = np.clip((1 - np.sqrt(r2)) * d / 2 + 0.5, 0, 1)
    lat = np.arcsin(np.clip(-ny, -1, 1))
    lon = np.arctan2(nx, np.maximum(nz, 1e-4))
    return nx, ny, nz, alpha, lat, lon


def sphere(d, tex, lon_offset=0.0, light=(-0.55, -0.35, 0.76), ambient=0.04, emissive=False, limb=0.6):
    """把等距柱状投影纹理贴到球上，返回 (d, d, 4) RGBA。"""
    d = int(d)
    nx, ny, nz, alpha, lat, lon = _sphere_grid(d)
    th, tw = tex.shape[:2]
    u = ((lon + lon_offset) / (2 * np.pi) % 1.0) * tw
    v = (0.5 - lat / np.pi) * (th - 1)
    col = cv2.remap(tex, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    if emissive:
        shade = (1 - limb) + limb * nz ** 0.5
    else:
        L = np.array(light, np.float32)
        L /= np.linalg.norm(L)
        shade = ambient + (1 - ambient) * np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    out = np.empty((d, d, 4), np.float32)
    out[..., :3] = col * shade[..., None]
    out[..., 3] = alpha
    return out


@lru_cache(maxsize=2)
def earth_texture(seed=21):
    h, w = 256, 512
    land = fbm(h, w, seed, 7, base=3, tile_x=True)
    detail = fbm(h, w, seed + 1, 6, base=6, tile_x=True)
    clouds = fbm(h, w, seed + 2, 7, base=4, tile_x=True)
    lat = np.abs(np.linspace(-1, 1, h))[:, None] * np.ones((1, w))
    is_land = np.clip((land - 0.56) * 12, 0, 1)
    ocean = np.array((0.03, 0.12, 0.38)) * (0.8 + 0.4 * detail[..., None])
    green = np.array((0.18, 0.36, 0.14)); brown = np.array((0.52, 0.42, 0.26))
    landc = green * (1 - detail[..., None]) + brown * detail[..., None]
    tex = ocean * (1 - is_land[..., None]) + landc * is_land[..., None]
    ice = np.clip((lat - 0.82) * 10, 0, 1)[..., None]
    tex = tex * (1 - ice) + 0.92 * ice
    cl = np.clip((clouds - 0.55) * 3, 0, 1)[..., None] * 0.85
    tex = tex * (1 - cl) + cl
    return tex.astype(np.float32)


@lru_cache(maxsize=2)
def moon_texture(seed=31):
    h, w = 256, 512
    base = fbm(h, w, seed, 7, base=4, tile_x=True)
    maria = np.clip((fbm(h, w, seed + 1, 4, base=2, tile_x=True) - 0.55) * 4, 0, 1)
    g = 0.62 + 0.25 * base - 0.22 * maria
    rng = np.random.default_rng(seed)
    for _ in range(140):
        cx, cy, r = rng.uniform(0, w), rng.uniform(20, h - 20), rng.uniform(1.5, 9)
        cv2.circle(g, (int(cx), int(cy)), int(r), float(np.mean(g) - 0.12), -1, cv2.LINE_AA)
        cv2.circle(g, (int(cx), int(cy)), int(r), float(np.mean(g) + 0.1), 1, cv2.LINE_AA)
    g = blur(g.astype(np.float32), 0.8)
    return np.repeat(g[..., None], 3, 2).astype(np.float32)


@lru_cache(maxsize=4)
def star_texture(kind="sun", seed=41):
    h, w = 256, 512
    if kind == "sun":
        n = 0.55 + 0.45 * fbm(h, w, seed, 5, base=24, tile_x=True)
        lo, hi = np.array((1.0, 0.7, 0.28)), np.array((1.0, 0.97, 0.8))
    elif kind == "red":
        n = fbm(h, w, seed, 6, base=4, tile_x=True) * 0.65 + fbm(h, w, seed + 9, 5, base=10, tile_x=True) * 0.35
        n = np.clip((n - 0.5) * 1.8 + 0.5, 0, 1) ** 1.3
        lo, hi = np.array((0.45, 0.07, 0.02)), np.array((1.0, 0.72, 0.38))
    elif kind == "blue":
        n = 0.6 + 0.4 * fbm(h, w, seed, 5, base=20, tile_x=True)
        lo, hi = np.array((0.62, 0.76, 1.0)), np.array((0.95, 0.98, 1.0))
    else:
        n = fbm(h, w, seed, 6, base=10, tile_x=True)
        lo, hi = np.array((1.0, 0.78, 0.35)), np.array((1.0, 0.97, 0.82))
    return (lo * (1 - n[..., None]) + hi * n[..., None]).astype(np.float32)


@lru_cache(maxsize=96)
def radial_falloff(size, power=2.0):
    """以中心为 1、边缘为 0 的径向渐变（用于光晕）。"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    return (np.clip(1 - r, 0, 1) ** power).astype(np.float32)


def halo(img, cx, cy, radius, color, k=1.0, power=2.5):
    """在 (cx, cy) 叠加一个径向光晕。"""
    size = (int(radius * 2) // 8) * 8 + 1
    if size < 9:
        return
    g = radial_falloff(size, power)[..., None] * (np.array(color, np.float32) * k)
    from .core import add
    add(img, g, cx - size / 2, cy - size / 2)


@lru_cache(maxsize=128)
def star_sprite(r, color, spikes=0.0, spike_len=0):
    L = int(max(r * 9, spike_len)) + 2
    yy, xx = np.mgrid[-L:L + 1, -L:L + 1].astype(np.float32)
    d = np.sqrt(xx ** 2 + yy ** 2)
    c = np.array(color, np.float32)
    img = (np.exp(-(d / (r * 0.55)) ** 2) * 1.5)[..., None] * np.ones(3, np.float32)
    img += (np.exp(-(d / (r * 1.7)) ** 2) * 0.55 + np.exp(-d / (r * 3.2)) * 0.18)[..., None] * c
    if spikes > 0 and spike_len > 0:
        w = r * 0.1 + 0.6
        sp = (np.exp(-(yy / w) ** 2) * np.exp(-np.abs(xx) / (spike_len * 0.28))
              + np.exp(-(xx / w) ** 2) * np.exp(-np.abs(yy) / (spike_len * 0.28)))
        img += (sp * spikes * 0.9)[..., None] * (0.4 + 0.6 * c)
    return img.astype(np.float32), L


def star_point(img, cx, cy, r, color, k=1.0, spikes=0.0, spike_len=None):
    """明亮的点光源：核心 + 光晕 + 可选十字星芒。"""
    rq = max(0.5, round(r * 2) / 2)
    col = tuple(round(float(v), 2) for v in color)
    sl = int(spike_len or rq * 22) if spikes > 0 else 0
    sprite, L = star_sprite(rq, col, round(float(spikes), 1), sl)
    from .core import add
    add(img, sprite, cx - L, cy - L, k)


# ---------- 剪影 ----------

SIL = (0.012, 0.016, 0.03)


def hill_silhouette(img, base_y, amp, seed=0, color=SIL, freq=(1.3, 3.1, 7.7)):
    rng = np.random.default_rng(seed)
    xs = np.linspace(-10, W + 10, 200)
    ph = rng.random(3) * 6.28
    ys = base_y - amp * (0.55 * np.sin(xs / W * np.pi * freq[0] + ph[0]) + 0.3 * np.sin(xs / W * np.pi * freq[1] + ph[1])
                         + 0.15 * np.sin(xs / W * np.pi * freq[2] + ph[2]))
    pts = list(zip(xs, ys)) + [(W + 10, H + 10), (-10, H + 10)]
    fill_poly(img, pts, color)
    return xs, ys


def person_silhouette(img, x, y, h, color=SIL, look_up=0.35):
    """站立、抬头仰望的人形剪影，(x, y) 为脚底中心。"""
    s = h / 100
    head_x, head_y = x + 4 * s * look_up, y - 88 * s
    circle(img, head_x, head_y, 7.5 * s, color)
    body = [(x - 9 * s, y - 78 * s), (x + 9 * s, y - 78 * s), (x + 11 * s, y - 45 * s), (x + 7 * s, y - 44 * s),
            (x + 6 * s, y), (x + 1.5 * s, y), (x, y - 38 * s), (x - 1.5 * s, y), (x - 6 * s, y),
            (x - 7 * s, y - 44 * s), (x - 11 * s, y - 45 * s)]
    fill_poly(img, body, color)
    fill_poly(img, [(head_x - 4 * s, head_y + 5 * s), (head_x + 4 * s, head_y + 5 * s), (x + 4 * s, y - 76 * s), (x - 4 * s, y - 76 * s)], color)


def acacia(img, x, y, h, color=SIL, seed=0):
    rng = np.random.default_rng(seed)
    s = h / 100
    fill_poly(img, [(x - 3 * s, y), (x + 3 * s, y), (x + 2 * s, y - 55 * s), (x - 1 * s, y - 55 * s)], color)
    for dx in (-22, 18):
        fill_poly(img, [(x, y - 50 * s), (x + 2 * s, y - 52 * s), (x + dx * s + 2 * s, y - 78 * s), (x + dx * s - 1 * s, y - 78 * s)], color)
    cx = x + rng.uniform(-4, 4) * s
    for i in range(14):
        ex = cx + rng.uniform(-45, 45) * s
        ey = y - (80 + rng.uniform(-3, 5)) * s
        from .core import ellipse
        ellipse(img, ex, ey, rng.uniform(14, 26) * s, rng.uniform(4, 8) * s, color)


def _roof(img, cx, ye, hw, rh, curl, color, s):
    """歇山顶剪影：凹曲的屋面 + 翘起的檐角。ye 为檐口高度。"""
    left, right = [], []
    for k in np.linspace(0, 1, 24):
        x = cx - hw + k * hw * 0.55
        y = ye - rh * k ** 1.7 - curl * (1 - k) ** 5
        left.append((x, y))
    for (x, y) in reversed(left):
        right.append((2 * cx - x, y))
    pts = left + right + [(cx + hw * 0.9, ye + 8 * s), (cx - hw * 0.9, ye + 8 * s)]
    fill_poly(img, pts, color)
    ridge_y = ye - rh
    for sx in (-1, 1):   # 正脊两端的鸱吻
        x = cx + sx * hw * 0.45
        fill_poly(img, [(x - 7 * s, ridge_y + 2), (x + 7 * s, ridge_y + 2), (x + sx * 6 * s, ridge_y - 16 * s), (x - sx * 4 * s, ridge_y - 10 * s)], color)
    return ridge_y


def gate_tower(img, cx, base_y, w, color=SIL, windows=0.0):
    """中国古代城楼剪影：城墙垛口 + 两层重檐楼阁。返回屋脊高度。"""
    s = w / 600
    wall_top = base_y - 150 * s
    fill_poly(img, [(cx - 520 * s, H + 10), (cx - 500 * s, wall_top), (cx + 500 * s, wall_top), (cx + 520 * s, H + 10)], color)
    for i in range(-22, 23):
        x = cx + i * 22 * s
        fill_poly(img, [(x - 7 * s, wall_top + 1), (x + 7 * s, wall_top + 1), (x + 7 * s, wall_top - 13 * s), (x - 7 * s, wall_top - 13 * s)], color)
    plat = wall_top - 14 * s
    fill_poly(img, [(cx - 300 * s, wall_top), (cx + 300 * s, wall_top), (cx + 300 * s, plat), (cx - 300 * s, plat)], color)
    hall1 = plat - 78 * s
    fill_poly(img, [(cx - 240 * s, plat), (cx + 240 * s, plat), (cx + 240 * s, hall1), (cx - 240 * s, hall1)], color)
    if windows > 0:
        for i in range(-4, 5):
            from .core import rect
            rect(img, cx + i * 48 * s - 12 * s, plat - 58 * s, cx + i * 48 * s + 12 * s, plat - 22 * s, (1.0, 0.62, 0.25), windows)
    r1 = _roof(img, cx, hall1, 330 * s, 62 * s, 26 * s, color, s)
    hall2 = r1 - 52 * s
    fill_poly(img, [(cx - 190 * s, r1 + 4), (cx + 190 * s, r1 + 4), (cx + 190 * s, hall2), (cx - 190 * s, hall2)], color)
    if windows > 0:
        for i in range(-3, 4):
            from .core import rect
            rect(img, cx + i * 48 * s - 11 * s, r1 - 42 * s, cx + i * 48 * s + 11 * s, r1 - 12 * s, (1.0, 0.62, 0.25), windows * 0.8)
    return _roof(img, cx, hall2, 270 * s, 80 * s, 30 * s, color, s)


def dunes(img, base_y, seed=3):
    rng = np.random.default_rng(seed)
    for layer, (dy, amp, col) in enumerate(((0, 40, (0.03, 0.03, 0.05)), (60, 55, (0.015, 0.015, 0.025)))):
        xs = np.linspace(-10, W + 10, 160)
        ph = rng.random(3) * 6.28
        ys = base_y + dy - amp * (0.6 * np.sin(xs / W * np.pi * 2.2 + ph[0]) + 0.4 * np.sin(xs / W * np.pi * 5.3 + ph[1]))
        fill_poly(img, list(zip(xs, ys)) + [(W + 10, H + 10), (-10, H + 10)], col)


def rooftops(img, base_y, seed=8, color=(0.05, 0.07, 0.12)):
    rng = np.random.default_rng(seed)
    x = -20
    while x < W + 20:
        bw = rng.uniform(60, 180)
        bh = rng.uniform(30, 160)
        top = base_y - bh
        if rng.random() < 0.4:
            fill_poly(img, [(x, H + 5), (x, top + 25), (x + bw / 2, top - 15), (x + bw, top + 25), (x + bw, H + 5)], color)
        else:
            fill_poly(img, [(x, H + 5), (x, top), (x + bw, top), (x + bw, H + 5)], color)
        x += bw + rng.uniform(-10, 8)
