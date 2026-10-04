"""球面/环面上的切向矢量场可视化（风、"毛"、磁力线）。"""
from functools import lru_cache

import numpy as np

from .core import circle, line


@lru_cache(maxsize=4)
def fibonacci_sphere(n=900):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1).astype(np.float32)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def tangent(p, v):
    """把矢量投影到球面切平面上。"""
    return v - (v * p).sum(-1, keepdims=True) * p


def swirl(p, center, k=1.0, width=0.5):
    """绕某个中心旋转的气旋（切向），强度随距离衰减。"""
    c = np.asarray(center, np.float32)
    c = c / np.linalg.norm(c)
    v = np.cross(c[None, :], p)
    d2 = ((p - c) ** 2).sum(-1, keepdims=True)
    return v * k * np.exp(-d2 / width ** 2)


def draw_sphere_field(img, cx, cy, R, field, R3, n=900, length=0.09, color=(1, 1, 1), thickness=2.2,
                      alpha=1.0, head=True, phase=0.0):
    """在正交投影的球面上画切向矢量场。R3 为模型→视图的旋转矩阵，field(p) 给出世界坐标下的切向量。"""
    p = fibonacci_sphere(n)
    v = tangent(p, field(p))
    pv, vv = p @ R3.T, v @ R3.T
    front = pv[:, 2] > 0.08
    pv, vv = pv[front], vv[front]
    mag = np.linalg.norm(vv, axis=1)
    vmax = np.percentile(np.linalg.norm(v, axis=1), 95) + 1e-6
    for (x, y, z), (dx, dy, dz), m in zip(pv, vv, mag):
        if m < 1e-4:
            continue
        s = min(m / vmax, 1.0)
        ux, uy = dx / m, dy / m
        L = length * (0.35 + 0.65 * s) * R
        # 让箭头沿风向缓慢"流动"
        off = ((phase * 0.6 + (x * 7.1 + y * 3.3) % 1.0) % 1.0 - 0.5) * L * 0.6
        x0, y0 = cx + x * R + ux * off, cy - y * R - uy * off
        x1, y1 = x0 + ux * L * z ** 0.3, y0 - uy * L * z ** 0.3
        a = alpha * (0.35 + 0.65 * s) * min(1.0, z * 3)
        line(img, (x0, y0), (x1, y1), color, thickness, alpha=a)
        if head and s > 0.25:
            hx, hy = -ux * L * 0.35, uy * L * 0.35
            line(img, (x1, y1), (x1 + hx * 0.8 - hy * 0.5, y1 + hy * 0.8 + hx * 0.5), color, thickness, alpha=a)
            line(img, (x1, y1), (x1 + hx * 0.8 + hy * 0.5, y1 + hy * 0.8 - hx * 0.5), color, thickness, alpha=a)


def project(p, R3, cx, cy, R):
    q = np.asarray(p, np.float32) @ R3.T
    return cx + q[0] * R, cy - q[1] * R, q[2]


def calm_marker(img, x, y, t, color, alpha=1.0):
    for k in range(3):
        r = 14 + ((t * 0.8 + k / 3) % 1.0) * 46
        circle(img, x, y, r, color, thickness=3, alpha=alpha * (1 - ((t * 0.8 + k / 3) % 1.0)))
    circle(img, x, y, 9, color, alpha=alpha)
