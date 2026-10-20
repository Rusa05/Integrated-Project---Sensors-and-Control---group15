"""Extended Kalman filter: wheel-odometry prediction + ArUco landmark updates.

State s = [x, y, theta] of base_link in W.

Prediction (every control step, 20 Hz) with the calibrated odometry model (kinematics.py):
    ds, dlat, dth = DiffDrive.forward3(dphi_l, dphi_r),   th_m = th + dth/2
    s+ = s + [R(th_m) [ds, dlat], dth]
    P+ = F P F^T + G diag(sig_l^2, sig_r^2) G^T + Q_rot
  where sig_i = k_s sqrt(|arc_i|) + q_floor is the per-wheel arc-length noise, G maps the
  wheel arc lengths into the pose increment and Q_rot = diag(k_xy^2, k_xy^2, k_th^2) |dth|
  covers turn-gain and ICR variation.

Update (each detected marker with a known id):
    z = [range, bearing, rel_yaw]  (yaw only used within yaw_max_range)
    R = diag(sigma(range)^2), sigma = a + b range  from scripts/03, with floors
    innovation gate: NIS < chi2(dof, gate_prob)   (rejects mis-detections / flips)
  Joseph-form covariance update.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from .config import Config
from .frames import rot2, wrap
from .kinematics import DiffDrive
from .model import meas_jacobian
from .sensing import MarkerNoiseModel


@dataclass
class UpdateRecord:
    t: float
    id: int
    nis: float
    dof: int
    accepted: bool
    innov: np.ndarray = field(default_factory=lambda: np.zeros(3))
    z: np.ndarray = field(default_factory=lambda: np.zeros(3))


class EKF:
    # Floors on the measurement std, covering effects the static characterisation cannot
    # see: motion during the frame, camera-extrinsic error and timing. Chosen from the
    # consistency sweep in scripts/06_tradeoffs.py (NEES vs R floor).
    RANGE_FLOOR = 0.01      # m
    BEARING_FLOOR = 0.005   # rad
    YAW_FLOOR = 0.03        # rad

    def __init__(self, cfg: Config, landmark_map: dict, sizes: dict | None = None,
                 r_scale: float = 1.0, q_scale: float = 1.0, use_yaw: bool | None = None):
        self.cfg = cfg
        self.dd = DiffDrive(cfg)
        self.map = landmark_map
        self.sizes = sizes or {k: cfg.ekf.ref_size for k in landmark_map}
        self.noise = MarkerNoiseModel(cfg)
        self.s = np.zeros(3)
        self.P = np.eye(3)
        self.initialised = False
        self.r_scale, self.q_scale = r_scale, q_scale
        self.use_yaw = cfg.ekf.use_yaw if use_yaw is None else use_yaw
        self.log: list[UpdateRecord] = []
        self.n_updates = 0

    # ------------------------------------------------------------------ initialisation
    def try_initialise(self, obs_list, t):
        """Global initialisation from one marker pose measurement (range, bearing, yaw)."""
        e = self.cfg.ekf
        cands = []
        for o in obs_list:
            # only unambiguous, close detections: no prior exists yet to resolve an IPPE flip
            size = self.sizes[o.id] if o.id in self.map else 0
            if (o.id not in self.map or o.z[0] > min(self.noise.yaw_max_range(size), 2.0)
                    or o.ambiguity > 0.5):
                continue
            lx, ly, lpsi = self.map[o.id]
            th = wrap(lpsi - o.z[2])
            p = np.array([lx, ly]) - rot2(th) @ (o.z[0] * np.array([np.cos(o.z[1]), np.sin(o.z[1])]))
            cands.append([p[0], p[1], th, o.z[0]])
        if not cands:
            return False
        best = min(cands, key=lambda c: c[3])          # closest marker = most accurate
        self.s = np.array(best[:3])
        sd = np.array(e.p0) * (1 + best[3])            # looser when initialised from far away
        self.P = np.diag(sd ** 2)
        self.initialised = True
        self.t_init = t
        return True

    # ------------------------------------------------------------------ prediction
    def predict(self, dphi, dt):
        if not self.initialised:
            return
        e, dd = self.cfg.ekf, self.dd
        ds, dlat, dth = dd.forward3(dphi[0], dphi[1], dt)
        th_m = self.s[2] + dth / 2
        c, s_ = np.cos(th_m), np.sin(th_m)
        self.s = dd.integrate(self.s, dphi[0], dphi[1], dt)
        self.s[2] = wrap(self.s[2])
        F = np.array([[1, 0, -ds * s_ - dlat * c], [0, 1, ds * c - dlat * s_], [0, 0, 1.0]])
        # G = d(pose)/d(arc_l, arc_r), with ds = (a_l + a_r)/2, dth = (a_r - a_l)/b,
        # dlat = -x_icr dth and th_m = th + dth/2.
        b = dd.b_equiv
        d_ds = np.array([0.5, 0.5])
        d_th = np.array([-1 / b, 1 / b])
        d_lat = -dd.x_icr * d_th
        d_thm = 0.5 * d_th
        G = np.vstack([d_ds * c - ds * s_ * d_thm - d_lat * s_ - dlat * c * d_thm,
                       d_ds * s_ + ds * c * d_thm + d_lat * c - dlat * s_ * d_thm,
                       d_th])
        arc = np.abs(dphi) * dd.r_tab.mean()
        sig = e.k_s * np.sqrt(arc) + e.q_floor
        Q = G @ np.diag(sig ** 2) @ G.T
        # residual rotation noise (turn-gain variation and ICR wander), from scripts/02
        a = abs(dth)
        Q += np.diag([e.k_rot_xy ** 2 * a, e.k_rot_xy ** 2 * a, e.k_rot_th ** 2 * a])
        self.P = F @ self.P @ F.T + Q * self.q_scale
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------ update
    def R_for(self, rng, size, view=None):
        sr, sb, sy = self.noise.sigma(rng, size, view)
        sr = max(sr, self.RANGE_FLOOR)
        sb = max(sb, self.BEARING_FLOOR)
        sy = max(sy, self.YAW_FLOOR)
        return np.diag(np.array([sr, sb, sy]) ** 2) * self.r_scale

    def update(self, obs_list, t):
        if not self.initialised:
            return
        e = self.cfg.ekf
        for o in obs_list:
            if o.id not in self.map:
                continue
            size = self.sizes[o.id]
            if o.z[0] > self.noise.max_range(size):
                continue
            lm = self.map[o.id]
            view = self.noise.view_angle(o.z)
            R = self.R_for(o.z[0], size, view)
            use_yaw = (self.use_yaw and o.z[0] <= self.noise.yaw_max_range(size)
                       and np.sqrt(R[2, 2] / self.r_scale) < e.yaw_sigma_max)
            dx, dy = lm[0] - self.s[0], lm[1] - self.s[1]
            zhat = np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - self.s[2]),
                             wrap(lm[2] - self.s[2])])
            H = meas_jacobian(self.s, lm, True)
            if not use_yaw:
                H, R = H[:2], R[:2, :2]
            S = H @ self.P @ H.T + R
            # Resolve the IPPE two-fold ambiguity: keep the solution that agrees with the prior.
            best = None
            for z in ([o.z] + ([o.z_alt] if o.z_alt is not None else [])):
                nu = z - zhat
                nu[1:] = wrap(nu[1:])
                nu = nu if use_yaw else nu[:2]
                d2 = float(nu @ np.linalg.solve(S, nu))
                if best is None or d2 < best[0]:
                    best = (d2, nu)
            nis, nu = best
            dof = len(nu)
            ok = nis < chi2.ppf(e.gate_prob, dof)
            self.log.append(UpdateRecord(t, o.id, nis, dof, ok, np.pad(nu, (0, 3 - dof)), o.z.copy()))
            if not ok:
                continue
            K = self.P @ H.T @ np.linalg.inv(S)
            self.s = self.s + K @ nu
            self.s[2] = wrap(self.s[2])
            I_KH = np.eye(3) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            self.n_updates += 1
