"""环面（甜甜圈）渲染：正交投影光线步进，一次算好着色/深度/参数坐标，之后可在表面上画箭头和磁力线。"""
from functools import lru_cache

import numpy as np

from .core import line


class Torus:
    def __init__(self, w, h, scale, R=1.0, r=0.42, tilt=0.55, yaw=0.0):
        self.w, self.h, self.scale, self.R, self.r = w, h, scale, R, r
        ct, st = np.cos(tilt), np.sin(tilt)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        self.M = (Rx @ Ry).astype(np.float32)          # 环面坐标 → 视图坐标
        self._render()

    def _sdf(self, P):
        q = np.sqrt(P[..., 0] ** 2 + P[..., 2] ** 2) - self.R
        return np.sqrt(q ** 2 + P[..., 1] ** 2) - self.r

    def _render(self):
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        X = (xs - self.w / 2) / self.scale
        Y = -(ys - self.h / 2) / self.scale
        z = np.full_like(X, 2.0)
        hit = np.zeros(X.shape, bool)
        Minv = self.M.T
        for _ in range(90):
            V = np.stack([X, Y, z], -1)
            P = V @ Minv.T
            d = self._sdf(P)
            hit |= d < 1e-3
            done = hit | (z < -2)
            z = np.where(done, z, z - np.clip(d, 1e-3, 0.5))
            if done.all():
                break
        hit &= z > -2
        V = np.stack([X, Y, z], -1)
        P = V @ Minv.T
        e = 1e-3
        n = np.stack([(self._sdf(P + [e, 0, 0]) - self._sdf(P - [e, 0, 0])),
                      (self._sdf(P + [0, e, 0]) - self._sdf(P - [0, e, 0])),
                      (self._sdf(P + [0, 0, e]) - self._sdf(P - [0, 0, e]))], -1)
        n = np.nan_to_num(n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9))
        nv = n @ self.M.T
        L = np.array([-0.45, 0.55, 0.7], np.float32)
        L /= np.linalg.norm(L)
        self.shade = np.clip((nv * L).sum(-1), 0, 1) * hit
        self.rim = (1 - np.clip(nv[..., 2], 0, 1)) ** 2 * hit
        self.depth = np.where(hit, z, -9).astype(np.float32)
        self.mask = hit.astype(np.float32)
        P = np.nan_to_num(P)
        self.u = np.arctan2(P[..., 2], P[..., 0])
        self.v = np.arctan2(P[..., 1], np.sqrt(P[..., 0] ** 2 + P[..., 2] ** 2) - self.R)

    def point(self, u, v):
        return np.stack([(self.R + self.r * np.cos(v)) * np.cos(u), self.r * np.sin(v),
                         (self.R + self.r * np.cos(v)) * np.sin(u)], -1)

    def to_screen(self, P, ox=0, oy=0):
        V = np.asarray(P, np.float32) @ self.M.T
        x = ox + self.w / 2 + V[..., 0] * self.scale
        y = oy + self.h / 2 - V[..., 1] * self.scale
        xi = np.clip(x - ox, 0, self.w - 1).astype(int)
        yi = np.clip(y - oy, 0, self.h - 1).astype(int)
        visible = V[..., 2] >= self.depth[yi, xi] - 0.03
        return x, y, visible

    def image(self, base=(0.20, 0.55, 0.62), tex=None):
        """返回 (h, w, 4) 的着色图。tex(u, v) -> (h, w, 3) 可选。"""
        col = tex(self.u, self.v) if tex is not None else np.ones((self.h, self.w, 3), np.float32) * base
        rgb = col * (0.12 + 0.88 * self.shade[..., None]) + self.rim[..., None] * 0.25 * np.array((0.4, 0.8, 1.0))
        return np.dstack([rgb, self.mask]).astype(np.float32)

    def arrows(self, img, ox, oy, t, n_u=36, n_v=12, color=(1, 1, 1), alpha=1.0, length=0.16, thickness=2.4):
        """沿环向（绕大圈）流动的箭头：处处不为零。"""
        for i in range(n_u):
            for j in range(n_v):
                u = (i + (t * 0.25) % 1.0) / n_u * 2 * np.pi
                v = j / n_v * 2 * np.pi
                P0, P1 = self.point(u, v), self.point(u + length, v)
                x0, y0, vis0 = self.to_screen(P0, ox, oy)
                x1, y1, vis1 = self.to_screen(P1, ox, oy)
                if vis0 and vis1:
                    line(img, (x0, y0), (x1, y1), color, thickness, alpha=alpha)
                    dx, dy = x1 - x0, y1 - y0
                    m = np.hypot(dx, dy) + 1e-6
                    hx, hy = -dx / m * 9, -dy / m * 9
                    line(img, (x1, y1), (x1 + hx - hy * 0.6, y1 + hy + hx * 0.6), color, thickness, alpha=alpha)
                    line(img, (x1, y1), (x1 + hx + hy * 0.6, y1 + hy - hx * 0.6), color, thickness, alpha=alpha)

    def helix(self, img, ox, oy, q=3.2, phase=0.0, color=(1.0, 0.6, 0.3), thickness=3, alpha=1.0, n=900, upto=1.0):
        """螺旋磁力线（绕大圈的同时绕小圈）。"""
        s = np.linspace(0, 2 * np.pi * upto, int(n * upto) + 2)
        P = self.point(s, q * s + phase)
        x, y, vis = self.to_screen(P, ox, oy)
        for k in range(1, len(s)):
            if vis[k] and vis[k - 1]:
                line(img, (x[k - 1], y[k - 1]), (x[k], y[k]), color, thickness, alpha=alpha)


@lru_cache(maxsize=4)
def cached(w, h, scale, tilt=0.55, yaw=0.0, r=0.42):
    return Torus(w, h, scale, r=r, tilt=tilt, yaw=yaw)
