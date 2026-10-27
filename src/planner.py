"""Plans a path around the obstacles and turns it into a timed reference (Part D).

  1. A* on the LiDAR map, staying at least 0.25 m away from obstacles
  2. shortcut and smooth the path
  3. add speeds (slower on curves), so the LQR knows where the robot should be at each moment
"""
from dataclasses import dataclass
import heapq
import math
import numpy as np
from scipy.signal import savgol_filter

from src import model as rm
from src.mapping import ROBOT_RADIUS

SOFT_MARGIN = 0.20      # m beyond ROBOT_RADIUS where cost rises
SOFT_WEIGHT = 4.0       # extra cost per cell at the obstacle edge of the soft margin
V_MAX = 0.18            # m/s cruise (Waffle Pi top speed 0.26 m/s; leaves headroom for feedback)
A_MAX = 0.20            # m/s^2
A_LAT_MAX = 0.15        # m/s^2, caps speed on curves: v <= sqrt(A_LAT_MAX / |kappa|)
DS = 0.01               # m, reference spacing


@dataclass
class Reference:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    omega: np.ndarray

    @property
    def duration(self):
        return float(self.t[-1])

    def at(self, t):
        """Reference state and input at time t (held at the end)."""
        t = min(max(t, 0.0), self.t[-1])
        k = int(np.searchsorted(self.t, t, side='right') - 1)
        k = min(k, len(self.t) - 2)
        a = (t - self.t[k]) / max(self.t[k + 1] - self.t[k], 1e-9)
        th = self.theta[k] + a * rm.wrap(self.theta[k + 1] - self.theta[k])
        lerp = lambda arr: arr[k] + a * (arr[k + 1] - arr[k])
        return np.array([lerp(self.x), lerp(self.y), rm.wrap(th)]), np.array([lerp(self.v), lerp(self.omega)])


class PlanningError(RuntimeError):
    pass


def astar(grid, start_xy, goal_xy):
    dist = grid.distance()
    blocked = dist < ROBOT_RADIUS
    penalty = SOFT_WEIGHT * np.clip((ROBOT_RADIUS + SOFT_MARGIN - dist) / SOFT_MARGIN, 0, 1)
    s, g = tuple(grid.to_cell(start_xy)), tuple(grid.to_cell(goal_xy))
    if blocked[g]:
        raise PlanningError(f'goal {goal_xy} is within {ROBOT_RADIUS} m of an obstacle')
    # The robot may start inside the inflated zone (e.g. near the dock); let it leave.
    blocked = blocked.copy()
    blocked[s] = False
    n = grid.n
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
    h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
    open_ = [(h(s), 0.0, s)]
    came, cost = {s: None}, {s: 0.0}
    while open_:
        _, c_cost, c = heapq.heappop(open_)
        if c == g:
            break
        if c_cost > cost[c]:
            continue
        for di, dj, step in moves:
            nb = (c[0] + di, c[1] + dj)
            if not (0 <= nb[0] < n and 0 <= nb[1] < n) or (blocked[nb] and nb != s):
                continue
            nc = c_cost + step * (1 + penalty[nb])
            if nc < cost.get(nb, math.inf):
                cost[nb], came[nb] = nc, c
                heapq.heappush(open_, (nc + h(nb), nc, nb))
    if g not in came:
        raise PlanningError('no path found')
    path, c = [], g
    while c is not None:
        path.append(c)
        c = came[c]
    pts = grid.to_world(np.array(path[::-1]))
    pts[0], pts[-1] = start_xy, goal_xy
    return pts


def _line_clear(grid, a, b, clearance):
    n = max(2, int(np.hypot(*(b - a)) / (grid.res / 2)))
    for t in np.linspace(0, 1, n):
        if grid.clearance_at(a + t * (b - a)) < clearance:
            return False
    return True


def smooth(grid, pts, iters=300, alpha=0.02, beta=0.3):
    # 1. shortcut: from each kept point, jump to the furthest point still in line of sight with the
    #    full soft margin, so corners keep room to be rounded in step 2 (falls back to less room
    #    in tight spaces).
    for clearance in (ROBOT_RADIUS + SOFT_MARGIN, ROBOT_RADIUS + 0.5 * SOFT_MARGIN, ROBOT_RADIUS):
        keep, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not _line_clear(grid, pts[i], pts[j], clearance):
                j -= 1
            keep.append(pts[j])
            i = j
        keep = np.array(keep)
        if len(keep) <= max(3, len(pts) // 4):
            break
    # 2. densify, then round the corners by iterative smoothing, refusing moves that lose clearance
    dense = [keep[0]]
    for a, b in zip(keep[:-1], keep[1:]):
        n = max(1, int(np.hypot(*(b - a)) / 0.05))
        dense += [a + (b - a) * k / n for k in range(1, n + 1)]
    p = np.array(dense)
    orig = p.copy()
    for _ in range(iters):
        for k in range(1, len(p) - 1):
            new = p[k] + alpha * (orig[k] - p[k]) + beta * (p[k - 1] + p[k + 1] - 2 * p[k])
            if grid.clearance_at(new) >= ROBOT_RADIUS:
                p[k] = new
    return p


def time_parametrise(pts, v_start=0.0, v_max=V_MAX, a_max=A_MAX):
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    if L < 1e-3:
        raise PlanningError('path too short')
    ss = np.linspace(0, L, max(3, int(L / DS) + 1))
    x, y = np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])
    # Savitzky-Golay smoothing (31 cm window, cubic) removes the 5 cm grid ripple that otherwise
    # shows up as a wobble in the feed-forward turn rate; it moves the path by well under 1 cm.
    win = min(31, len(ss) - (1 - len(ss) % 2))
    if win >= 5:
        x[1:-1] = savgol_filter(x, win, 3)[1:-1]
        y[1:-1] = savgol_filter(y, win, 3)[1:-1]
    dx, dy = np.gradient(x, ss), np.gradient(y, ss)
    theta = np.unwrap(np.arctan2(dy, dx))
    # Curvature is a second derivative of the path, so even sub-mm grid kinks become a visible
    # wobble in the feed-forward turn rate. Differentiate heading with a 41 cm Savitzky-Golay
    # derivative instead of a raw finite difference.
    win = min(41, len(ss) - (1 - len(ss) % 2))
    if win >= 5:
        theta = savgol_filter(theta, win, 3)
        kappa = savgol_filter(theta, win, 3, deriv=1, delta=DS)
    else:
        kappa = np.gradient(theta, ss)
    # speed limits: cruise, curvature, wheel speed (|v| + |omega| b/2 <= r w_max)
    wheel_lim = 0.9 * rm.WHEEL_RADIUS * rm.WHEEL_SPEED_LIMIT
    v = np.minimum(v_max, np.sqrt(A_LAT_MAX / np.maximum(np.abs(kappa), 1e-6)))
    v = np.minimum(v, wheel_lim / (1 + np.abs(kappa) * rm.WHEEL_SEPARATION / 2))
    # acceleration limits (forward and backward passes), start from v_start, stop at the end
    v[0], v[-1] = v_start, 0.0
    for k in range(1, len(v)):
        v[k] = min(v[k], math.sqrt(v[k - 1] ** 2 + 2 * a_max * (ss[k] - ss[k - 1])))
    for k in range(len(v) - 2, -1, -1):
        v[k] = min(v[k], math.sqrt(v[k + 1] ** 2 + 2 * a_max * (ss[k + 1] - ss[k])))
    v_mid = np.maximum((v[:-1] + v[1:]) / 2, 1e-3)
    t = np.concatenate([[0], np.cumsum(np.diff(ss) / v_mid)])
    return Reference(t, x, y, np.array([rm.wrap(a) for a in theta]), v, v * kappa)


def plan(grid, start_xy, goal_xy, v_start=0.0):
    pts = astar(grid, np.asarray(start_xy, float), np.asarray(goal_xy, float))
    return time_parametrise(smooth(grid, pts), v_start), pts
