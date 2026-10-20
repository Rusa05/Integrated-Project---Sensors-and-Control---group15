"""LiDAR occupancy mapping, A* path planning and trajectory generation.

* OccupancyGrid : log-odds grid built from LiDAR scans placed with the EKF pose estimate.
                  The robot is told only the room extent (implied by the wall landmarks);
                  obstacles are discovered online. Unknown space is treated as free
                  (optimistic planning) and the path is re-checked as the map fills in.
* plan_path     : 8-connected A* on the inflated grid with a clearance cost, then
                  line-of-sight shortcutting.
* Trajectory    : smoothed path -> curvature -> speed profile v(s) limited by v_max,
                  lateral acceleration, yaw rate and acceleration -> time-parameterised
                  reference [x, y, th, v, w](t) for the LQR tracker.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import distance_transform_edt

from .config import Config
from .frames import rot2, wrap
from .sensing import scan_to_points


class OccupancyGrid:
    L_OCC, L_FREE, L_MIN, L_MAX, L_THRESH = 0.85, -0.4, -2.0, 3.5, 0.6

    def __init__(self, cfg: Config, size=(6.0, 4.0)):
        self.res = cfg.planner.resolution
        self.nx, self.ny = int(round(size[0] / self.res)), int(round(size[1] / self.res))
        self.logodds = np.zeros((self.nx, self.ny))
        self.inflate = cfg.planner.inflate
        self._dist = None

    def to_cell(self, p):
        return np.floor(np.asarray(p) / self.res).astype(int)

    def in_bounds(self, c):
        return (c[..., 0] >= 0) & (c[..., 0] < self.nx) & (c[..., 1] >= 0) & (c[..., 1] < self.ny)

    def integrate(self, pose, scan, angles, max_range=3.5):
        """Ray-trace free space and mark end points (vectorised sampling along each ray)."""
        pts_B = scan_to_points(scan, angles)
        if len(pts_B) == 0:
            return
        R = rot2(pose[2])
        pts = pts_B @ R.T + pose[:2]
        origin = pose[:2] + R @ np.array([-0.049, 0.0])
        # free cells: sample each ray up to 1 cell short of the hit
        d = pts - origin
        L = np.linalg.norm(d, axis=1)
        n = int(np.ceil(L.max() / (self.res * 0.7))) + 1
        f = np.linspace(0, 1, n)[None, :, None]
        stop = np.clip(1 - 1.0 * self.res / np.maximum(L, 1e-6), 0, 1)[:, None, None]
        samples = origin + d[:, None, :] * (f * stop)
        cells = self.to_cell(samples.reshape(-1, 2))
        cells = cells[self.in_bounds(cells)]
        cells = np.unique(cells, axis=0)
        self.logodds[cells[:, 0], cells[:, 1]] += self.L_FREE
        hit = self.to_cell(pts)
        hit = hit[self.in_bounds(hit) & (L < max_range - 0.05)]
        hit = np.unique(hit, axis=0)
        self.logodds[hit[:, 0], hit[:, 1]] += self.L_OCC - self.L_FREE
        np.clip(self.logodds, self.L_MIN, self.L_MAX, out=self.logodds)
        self._dist = None

    @property
    def occupied(self):
        return self.logodds > self.L_THRESH

    def clearance(self):
        """Distance [m] from each cell to the nearest occupied cell or the room boundary."""
        if self._dist is None:
            occ = self.occupied.copy()
            occ[0, :] = occ[-1, :] = True
            occ[:, 0] = occ[:, -1] = True
            self._dist = distance_transform_edt(~occ) * self.res
        return self._dist

    def free(self, cells, margin=0.0):
        c = np.atleast_2d(cells)
        ok = self.in_bounds(c)
        out = np.zeros(len(c), bool)
        cl = self.clearance()
        out[ok] = cl[c[ok, 0], c[ok, 1]] > self.inflate + margin
        return out

    def segment_free(self, a, b, margin=0.0):
        n = int(np.ceil(np.linalg.norm(np.subtract(b, a)) / (self.res * 0.5))) + 2
        pts = np.linspace(a, b, n)
        return bool(np.all(self.free(self.to_cell(pts), margin)))

    def path_free(self, pts, margin=-0.03):
        return bool(np.all(self.free(self.to_cell(np.asarray(pts)), margin)))


def plan_path(grid: OccupancyGrid, start, goal, clearance_weight=2.0):
    """A* from start to goal (world xy). Returns a shortcut polyline (list of xy) or None."""
    cl = grid.clearance()
    s, g = tuple(grid.to_cell(start)), tuple(grid.to_cell(goal))
    passable = cl > grid.inflate
    # Let the robot leave its current cell even if the inflated map clips it.
    near_start = np.zeros_like(passable)
    r = int(np.ceil(grid.inflate / grid.res))
    near_start[max(s[0] - r, 0):s[0] + r + 1, max(s[1] - r, 0):s[1] + r + 1] = True
    passable = passable | (near_start & (cl > 0.16))
    if not passable[g]:
        return None
    # soft cost to prefer staying away from obstacles
    extra = clearance_weight * np.exp(-(cl - grid.inflate) / 0.15)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    h = lambda c: np.hypot(c[0] - g[0], c[1] - g[1])
    openq = [(h(s), 0.0, s, None)]
    came, cost = {}, {s: 0.0}
    while openq:
        _, gc, c, parent = heapq.heappop(openq)
        if c in came:
            continue
        came[c] = parent
        if c == g:
            break
        for dx, dy in moves:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < grid.nx and 0 <= n[1] < grid.ny) or not passable[n]:
                continue
            step = np.hypot(dx, dy) * (1 + extra[n])
            nc = gc + step
            if nc < cost.get(n, np.inf):
                cost[n] = nc
                heapq.heappush(openq, (nc + h(n), nc, n, c))
    if g not in came:
        return None
    cells = []
    c = g
    while c is not None:
        cells.append(c)
        c = came[c]
    cells = cells[::-1]
    pts = [(np.array(c) + 0.5) * grid.res for c in cells]
    pts[0], pts[-1] = np.asarray(start, float), np.asarray(goal, float)
    # line-of-sight shortcutting (keeps a small extra margin)
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not grid.segment_free(pts[i], pts[j], margin=0.02):
            j -= 1
        out.append(pts[j])
        i = j
    return out


class Trajectory:
    """Time-parameterised reference along a smoothed polyline."""

    def __init__(self, waypoints, cfg: Config, v0=0.0, v_end=0.0, v_max=None, ds=0.01,
                 smooth_iters=60, grid: OccupancyGrid | None = None, end_heading=None):
        L = cfg.lqr
        v_max = v_max or L.v_max
        wp = np.asarray(waypoints, float)
        # densify
        seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
        s_wp = np.concatenate([[0], np.cumsum(seg)])
        s = np.arange(0, s_wp[-1], ds)
        s = np.append(s, s_wp[-1])
        xy = np.stack([np.interp(s, s_wp, wp[:, 0]), np.interp(s, s_wp, wp[:, 1])], 1)
        # Laplacian smoothing with fixed ends; undo iterations that would hit obstacles
        if end_heading is not None and len(xy) > 20:
            # lock the last 15 cm straight along the requested final heading
            k = min(15, len(xy) // 3)
            d = np.array([np.cos(end_heading), np.sin(end_heading)])
            xy[-k:] = xy[-1] - d * (s[-1] - s[-k:])[:, None]
        for _ in range(smooth_iters):
            new = xy.copy()
            new[1:-1] = 0.5 * xy[1:-1] + 0.25 * (xy[:-2] + xy[2:])
            if end_heading is not None and len(xy) > 20:
                new[-k:] = xy[-k:]
            if grid is not None and not grid.path_free(new):
                break
            xy = new
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        th = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
        th = np.unwrap(th)
        kappa = np.gradient(th) / np.maximum(np.gradient(s), 1e-6)
        kappa = np.convolve(kappa, np.ones(5) / 5, mode="same")
        v = np.minimum(v_max, np.sqrt(L.a_lat_max / np.maximum(np.abs(kappa), 1e-6)))
        v = np.minimum(v, L.w_max * 0.6 / np.maximum(np.abs(kappa), 1e-6))
        a_max = 0.15
        v[0], v[-1] = max(v0, 0.02), v_end
        for i in range(1, len(v)):            # forward pass (acceleration limit)
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_max * (s[i] - s[i - 1])))
        for i in range(len(v) - 2, -1, -1):   # backward pass (deceleration limit)
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_max * (s[i + 1] - s[i])))
        v = np.maximum(v, 0.02)
        v[-1] = max(v_end, 0.0)
        dt_seg = np.diff(s) / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)
        self.t = np.concatenate([[0], np.cumsum(dt_seg)])
        self.xy, self.th, self.v, self.s = xy, th, v, s
        self.w = v * kappa
        self.T = self.t[-1]
        self.length = s[-1]

    def ref(self, t):
        t = np.clip(t, 0, self.T)
        x = np.interp(t, self.t, self.xy[:, 0])
        y = np.interp(t, self.t, self.xy[:, 1])
        th = wrap(np.interp(t, self.t, self.th))
        v = np.interp(t, self.t, self.v) if t < self.T else 0.0
        w = np.interp(t, self.t, self.w) if t < self.T else 0.0
        return np.array([x, y, th, v, w])

    @property
    def start_heading(self):
        return float(wrap(self.th[min(5, len(self.th) - 1)]))
