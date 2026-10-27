"""LQR path-following controller (Part D).

Works out how far the robot is from the planned path (along, across and heading) and turns that
into wheel commands with an LQR gain. Also has a simple turn-on-the-spot controller, used where
the LQR can't work (when the robot isn't moving).
"""
import math
import numpy as np
from scipy.linalg import solve_discrete_are

from src import model as rm

E_MAX = (0.05, 0.05, 0.15)
U_MAX = (0.10, 0.8)


class LQRTracker:
    def __init__(self, e_max=E_MAX, u_max=U_MAX, r_scale=1.0, dt=rm.DT):
        self.Q = np.diag([1 / e ** 2 for e in e_max])
        self.R = r_scale * np.diag([1 / u ** 2 for u in u_max])
        self.dt = dt
        self._cache = {}

    def gain(self, v_r, w_r):
        # At v_r = w_r = 0 the cross-track error is uncontrollable (rank 2) and the Riccati
        # equation has no stabilising solution; use the gain for a slow crawl instead.
        if abs(v_r) < 0.02 and abs(w_r) < 0.05:
            v_r = 0.02
        key = (round(v_r, 2), round(w_r, 2))
        if key not in self._cache:
            A, B = rm.discretise(*rm.error_dynamics(*key), self.dt)
            P = solve_discrete_are(A, B, self.Q, self.R)
            self._cache[key] = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)
        return self._cache[key]

    def control(self, x_est, x_ref, u_ref):
        """Returns (v, omega) after wheel limits, and the tracking error used."""
        e = rm.tracking_error(x_est, x_ref)
        K = self.gain(*u_ref)
        u = np.array([u_ref[0] * math.cos(e[2]), u_ref[1]]) - K @ e
        return saturate(*u), e


def saturate(v, omega):
    """Scale (v, omega) together so neither wheel exceeds its limit (keeps the path curvature)."""
    wl, wr = (v - omega * rm.WHEEL_SEPARATION / 2) / rm.WHEEL_RADIUS, (v + omega * rm.WHEEL_SEPARATION / 2) / rm.WHEEL_RADIUS
    k = max(abs(wl), abs(wr)) / rm.WHEEL_SPEED_LIMIT
    return (v / k, omega / k) if k > 1 else (v, omega)


def turn_to(theta_est, theta_goal, k=2.0, w_max=0.8):
    """Rotate on the spot towards a heading (used where the tracker is not controllable: v_r = 0)."""
    err = rm.wrap(theta_goal - theta_est)
    return 0.0, float(np.clip(k * err, -w_max, w_max)), err
