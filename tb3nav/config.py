"""All tunable parameters in one place, each with the reason it has that value.

Values marked (calibrated) are overwritten at start-up by results/calibration.json if that file
exists (produced by scripts/02_calibrate_odometry.py and scripts/03_calibrate_camera.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALIB_FILE = ROOT / "results" / "calibration.json"


@dataclass
class Timing:
    physics_dt: float = 0.002     # MuJoCo step; 500 Hz is also the motor-firmware rate
    control_dt: float = 0.05      # 20 Hz outer loop: >5x the closed-loop bandwidth (~0.5 Hz)
    camera_period: float = 0.1    # 10 Hz, typical of Raspberry Pi camera + ArUco on the TB3 SBC
    lidar_period: float = 0.2     # 5 Hz, the LDS-01 scan rate


@dataclass
class Robot:
    # Datasheet (ROBOTIS e-Manual, Waffle Pi): r = 0.033 m, wheel separation = 0.287 m.
    wheel_radius: float = 0.033            # (calibrated) effective rolling radius
    wheel_base: float = 0.288              # (calibrated) effective track width
    ticks_per_rev: int = 4096              # Dynamixel XM430 encoder resolution
    radius_table: tuple | None = None      # (calibrated) ([w_mean...], [r_eff...]), see kinematics.py
    turn_gain: float | None = None         # (calibrated) dtheta / (dphi_r - dphi_l)
    icr_x: float = 0.0                     # (calibrated) ICR offset along body x [m]
    turn_table: tuple | None = None        # (calibrated) ([|w_mean|...], [g...])
    max_wheel_speed: float = 6.0           # rad/s, below the 7.88 rad/s actuator ctrlrange
    # Motor firmware (emulates Dynamixel velocity-mode PI loop); see sim.py
    firmware_ki: float = 40.0


@dataclass
class Noise:
    # Encoders: quantisation is applied explicitly; this adds slip-like random walk per metre.
    enc_slip_std_per_m: float = 0.01       # rad of wheel angle per metre travelled (~0.33 mm/m)
    # LDS-01 datasheet: +-15 mm (0.12-0.5 m), +-5 % (0.5-3.5 m); we treat these as 3-sigma.
    lidar_std_near: float = 0.005
    lidar_rel_std_far: float = 0.0167
    lidar_dropout: float = 0.01
    lidar_min: float = 0.12
    lidar_max: float = 3.5
    lidar_beams: int = 360
    # Camera: additive Gaussian pixel noise (8-bit units), Pi camera in indoor light.
    pixel_noise_std: float = 3.0


@dataclass
class EKFParams:
    # Odometry noise model: sigma(ds_wheel) = k_s * sqrt(|ds_wheel|) + quantisation floor.
    # k_s chosen from the odometry calibration residuals (scripts/02), inflated x2 for slip.
    k_s: float = 0.01                      # (calibrated) m^0.5
    q_floor: float = 1e-4                  # m, ~2 encoder ticks of arc length
    # Residual rotation noise (turn gain varies with acceleration, ICR wanders):
    #   sigma_th = k_rot_th sqrt|dtheta|,  sigma_xy = k_rot_xy sqrt|dtheta|   (calibrated)
    k_rot_th: float = 0.02
    k_rot_xy: float = 0.01
    # Measurement noise for [range, bearing, relative yaw]; (calibrated) by scripts/03 as a
    # function of range: sigma = a + b * range.
    r_range: tuple = (0.004, 0.010)
    r_bearing: tuple = (0.004, 0.002)
    r_yaw: tuple = (0.03, 0.05)
    r_yaw_frontal: tuple = (0.0, 0.03)     # (calibrated) yaw sigma for head-on views
    r_yaw_oblique: tuple = (0.0, 0.01)     # (calibrated) yaw sigma for views >= 0.5 rad off-normal
    yaw_sigma_max: float = 0.08            # yaw is only used when its predicted sigma is below
    use_yaw: bool = True                   # use the marker's relative yaw (pose measurement)
    yaw_max_range: float = 2.0             # (calibrated) beyond this, IPPE yaw flips (scripts/03)
    max_range: float = 4.0                 # (calibrated) markers further away are ignored
    ref_size: float = 0.15                 # (calibrated) tag size the noise model was fitted for
    range_bias_poly: tuple = (0.0, 0.0)    # (calibrated) range bias = c1 r + c2 r^2
    gate_prob: float = 0.997               # chi-square innovation gate
    p0: tuple = (0.05, 0.05, 0.10)         # initial std [m, m, rad] after marker initialisation


@dataclass
class LQRParams:
    # Bryson's rule: Q_ii = 1/e_max^2, R_ii = 1/u_max^2.
    e_max: tuple = (0.10, 0.05, 0.15)      # along-track 10 cm, cross-track 5 cm, heading 0.15 rad
    u_max: tuple = (0.10, 0.6)             # feedback velocity 0.10 m/s, yaw rate 0.6 rad/s
    v_max: float = 0.15                    # m/s nominal cruise (actuator limit ~0.17 m/s)
    w_max: float = 1.0                     # rad/s
    a_lat_max: float = 0.15                # m/s^2, keeps |w| = |v k| modest on bends


@dataclass
class PlannerParams:
    resolution: float = 0.05               # m per occupancy cell
    inflate: float = 0.24                  # m; robot half-width 0.153 m + 9 cm safety margin
    replan_period: float = 2.0             # s
    safety_stop: float = 0.20              # m, front-sector LiDAR stop distance
    predock_dist: float = 0.70             # m in front of dock face where the final approach starts
    dock_speed: float = 0.05               # m/s on the final approach


@dataclass
class Config:
    timing: Timing = field(default_factory=Timing)
    robot: Robot = field(default_factory=Robot)
    noise: Noise = field(default_factory=Noise)
    ekf: EKFParams = field(default_factory=EKFParams)
    lqr: LQRParams = field(default_factory=LQRParams)
    planner: PlannerParams = field(default_factory=PlannerParams)

    def to_dict(self):
        return asdict(self)


def load_config(use_calibration: bool = True) -> Config:
    cfg = Config()
    if use_calibration and CALIB_FILE.exists():
        c = json.loads(CALIB_FILE.read_text())
        od = c.get("odometry", {})
        cfg.robot.wheel_radius = od.get("wheel_radius", cfg.robot.wheel_radius)
        cfg.robot.wheel_base = od.get("wheel_base", cfg.robot.wheel_base)
        cfg.ekf.k_s = od.get("k_s", cfg.ekf.k_s)
        if "radius_table" in od:
            cfg.robot.radius_table = tuple(od["radius_table"])
        cfg.robot.turn_gain = od.get("turn_gain", cfg.robot.turn_gain)
        cfg.robot.icr_x = od.get("icr_x", cfg.robot.icr_x)
        if "turn_table" in od:
            cfg.robot.turn_table = tuple(od["turn_table"])
        cfg.ekf.k_rot_th = od.get("k_rot_th", cfg.ekf.k_rot_th)
        cfg.ekf.k_rot_xy = od.get("k_rot_xy", cfg.ekf.k_rot_xy)
        mn = c.get("marker_noise", {})
        for k in ("r_range", "r_bearing", "r_yaw", "range_bias_poly", "yaw_max_range", "max_range", "ref_size",
                  "r_yaw_frontal", "r_yaw_oblique"):
            if k in mn:
                v = mn[k]
                setattr(cfg.ekf, k, tuple(v) if isinstance(v, list) else v)
    return cfg


def camera_intrinsics(width=640, height=480, fovy_deg=48.8, calib=True):
    """Pinhole K. Uses the calibrated K if available, else the analytic one from fovy."""
    if calib and CALIB_FILE.exists():
        c = json.loads(CALIB_FILE.read_text()).get("camera")
        if c:
            return np.array(c["K"]), np.array(c["dist"])
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
    K = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])
    return K, np.zeros(5)
