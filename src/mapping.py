"""Builds a map of the obstacles from the LiDAR (Part D).

Each scan is placed on a 5 cm grid using the EKF position: cells a beam passes through count as
free, the cell where it hits something counts as occupied.
"""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.sensors import LIDAR_ANGLES

LIDAR_OFFSET = (-0.035, 0.0)    # m, LiDAR scan centre in base_link (turtlebot3_waffle_pi_track_a.xml)

# Inflation: the robot's footprint about the wheel axle spans x -0.197..0.077, y +-0.153, so the
# circle that contains it from the axle has radius sqrt(0.197^2 + 0.153^2) = 0.25 m.
ROBOT_RADIUS = 0.25


class OccupancyGrid:
    L_OCC, L_FREE, L_MIN, L_MAX, L_THRESH = 0.85, -0.4, -4.0, 4.0, 0.6

    def __init__(self, xmin=-2.2, ymin=-2.2, size=4.4, res=0.05):
        self.res = res
        self.origin = np.array([xmin, ymin])
        self.n = int(round(size / res))
        self.logodds = np.zeros((self.n, self.n))       # [ix, iy]
        self._dist = None

    # -- coordinates ----------------------------------------------------------------
    def to_cell(self, xy):
        ij = np.floor((np.asarray(xy) - self.origin) / self.res).astype(int)
        return np.clip(ij, 0, self.n - 1)

    def to_world(self, ij):
        return self.origin + (np.asarray(ij) + 0.5) * self.res

    # -- update -----------------------------------------------------------------------
    def update(self, pose, ranges, max_range=3.5):
        """Integrate one scan taken at the (estimated) pose."""
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        ox = x + c * LIDAR_OFFSET[0] - s * LIDAR_OFFSET[1]
        oy = y + s * LIDAR_OFFSET[0] + c * LIDAR_OFFSET[1]
        o = self.to_cell((ox, oy))
        for ang, r in zip(LIDAR_ANGLES, ranges):
            if not np.isfinite(r):
                continue            # no return (dropout or nothing in range): no information
            hit = r < max_range
            ex, ey = ox + r * math.cos(th + ang), oy + r * math.sin(th + ang)
            e = self.to_cell((ex, ey))
            cells = _bresenham(o[0], o[1], e[0], e[1])
            for (i, j) in cells[:-1]:
                self.logodds[i, j] += self.L_FREE
            if hit:
                self.logodds[e[0], e[1]] += self.L_OCC
        np.clip(self.logodds, self.L_MIN, self.L_MAX, out=self.logodds)
        self._dist = None

    # -- queries ----------------------------------------------------------------------
    def occupied(self):
        return self.logodds > self.L_THRESH

    def distance(self):
        """Distance (m) from each cell centre to the nearest occupied cell."""
        if self._dist is None:
            occ = self.occupied()
            self._dist = distance_transform_edt(~occ) * self.res if occ.any() else np.full(occ.shape, 99.0)
        return self._dist

    def clearance_at(self, xy):
        i, j = self.to_cell(xy)
        return self.distance()[i, j]


def _bresenham(x0, y0, x1, y1):
    cells = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
