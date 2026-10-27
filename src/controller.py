"""LQR trajectory tracker on the EKF estimate (Part D).

Error model (src/model.py): with e = reference pose relative to the robot, in the robot frame,
and u = u_ff + u_fb,
    e_dot = A(v_r, w_r) e + B u_fb,    A = [[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]],  B = -[[1,0],[0,0],[0,1]]
discretised with zero-order hold at dt = 0.02 s. The gain K comes from the discrete Riccati
equation, recomputed as the reference speed changes (gain scheduling, cached on a coarse grid).
    u = [v_r cos(e3), w_r] - K e            then wheel limits are applied.

Weights by Bryson's rule, Q = diag(1/e_max^2), R = diag(1/u_max^2):
    e_max: 5 cm along-track, 5 cm cross-track, 0.15 rad heading
    u_max: 0.10 m/s and 0.8 rad/s of corrective speed
The cross-track error is only controllable through heading while moving (v_r != 0), which is
why the gain on e2 scales with v_r and why the robot turns on the spot before it sets off.
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
