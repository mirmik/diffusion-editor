"""Small NumPy implicit modelling kernel. Coordinates and distances are metres.

Ellipsoid and loft distances are approximations, adequate for isosurface
extraction and artistic blends; they are not exact Euclidean SDFs.
"""
import numpy as np


def smooth_min(a, b, radius):
    if radius <= 0:
        return np.minimum(a, b)
    h = np.maximum(radius - np.abs(a - b), 0) / radius
    return np.minimum(a, b) - h * h * radius * 0.25


def ellipsoid(p, center, radii):
    q = [(p[i] - center[i]) / radii[i] for i in range(3)]
    k0 = np.sqrt(sum(v * v for v in q))
    k1 = np.sqrt(sum((q[i] / radii[i]) ** 2 for i in range(3)))
    return np.where(k1 > 1e-8, k0 * (k0 - 1) / np.maximum(k1, 1e-8), -min(radii))


def muscle(p, a, b, width, depth):
    """Oriented ellipsoid with endpoints on its long axis."""
    a, b = np.array(a), np.array(b)
    axis = b - a
    length = np.linalg.norm(axis)
    axis /= length
    side = np.cross(axis, [0, 1, 0])
    side /= np.linalg.norm(side)
    front = np.cross(side, axis)
    q = [p[i] - (a[i] + b[i]) * 0.5 for i in range(3)]
    local = [sum(q[i] * v[i] for i in range(3)) for v in (side, front, axis)]
    return ellipsoid(local, (0, 0, 0), (width, depth, length * 0.5))


def interpolate(z, xs, ys):
    """Shape-preserving cubic Hermite interpolation, without SciPy."""
    dx = np.diff(xs)
    slopes = np.diff(ys) / dx
    tangent = np.zeros_like(ys)
    tangent[0], tangent[-1] = slopes[0], slopes[-1]
    for i in range(1, len(ys) - 1):
        if slopes[i-1] * slopes[i] > 0:
            w1, w2 = 2 * dx[i] + dx[i-1], dx[i] + 2 * dx[i-1]
            tangent[i] = (w1 + w2) / (w1 / slopes[i-1] + w2 / slopes[i])
    index = np.clip(np.searchsorted(xs, z, side='right') - 1, 0, len(xs) - 2)
    t = np.clip((z - xs[index]) / dx[index], 0, 1)
    return ((2*t**3 - 3*t**2 + 1) * ys[index]
            + (t**3 - 2*t**2 + t) * dx[index] * tangent[index]
            + (-2*t**3 + 3*t**2) * ys[index+1]
            + (t**3 - t**2) * dx[index] * tangent[index+1])


def loft(p, keys):
    """Closed vertical loft: z, half width, front depth, back depth, y centre.

    Shape-preserving interpolation avoids horizontal bands at the profile keys.
    """
    keys = np.asarray(keys)
    z = p[2]
    rx, rf, rb, cy = [interpolate(z, keys[:, 0], keys[:, i]) for i in range(1, 5)]
    y = p[1] - cy
    ry = np.where(y < 0, rf, rb)
    radial = (np.sqrt((p[0] / rx) ** 2 + (y / ry) ** 2) - 1) * np.minimum(rx, ry)
    cap = np.maximum(keys[0, 0] - z, z - keys[-1, 0])
    return np.minimum(np.maximum(radial, cap), 0) + np.sqrt(np.maximum(radial, 0) ** 2 + np.maximum(cap, 0) ** 2)


class Sculpt:
    def __init__(self, p):
        self.p = p
        self.field = np.full(np.broadcast_shapes(*(a.shape for a in p)), 1.0, dtype=np.float32)

    def add(self, distance, blend=0.008):
        self.field = smooth_min(self.field, distance, blend)

    def cut(self, distance, blend=0.002):
        self.field = -smooth_min(-self.field, distance, blend)

    def ell(self, center, radii, blend=0.008):
        self.add(ellipsoid(self.p, center, radii), blend)

    def muscle(self, a, b, width, depth, blend=0.008):
        self.add(muscle(self.p, a, b, width, depth), blend)
