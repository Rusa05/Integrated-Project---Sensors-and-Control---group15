"""Step 1 - Platform model: derivation check, actuator identification, controllability and
observability analysis (Part A).

  1. Motor/actuator identification: wheel-speed step responses -> first-order time constant
     tau and the static gain of the firmware-closed velocity loop.
  2. Model validation: the kinematic (unicycle) model with calibrated parameters, driven by
     the measured wheel speeds, vs the MuJoCo trajectory for a random input sequence.
  3. Controllability of
       (a) the full 5-state model about a moving operating point,
       (b) the LQR tracking-error model as a function of (v_r, w_r)  -> fails at v_r = w_r = 0,
       (c) the unicycle linearised about a STANDSTILL point          -> rank 2 (Brockett),
  4. Observability for different sensor sets (odometry only; bearing only; range only;
     range + bearing; range + bearing + yaw), for a stationary and a moving robot, from the
     rank / conditioning of the observability Gramian.

Run:  python scripts/01_model_analysis.py      (prints a report, writes results/model_analysis.json)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from tb3nav import model as M
from tb3nav import world as W
from tb3nav.config import load_config
from tb3nav.frames import wrap
from tb3nav.kinematics import DiffDrive
from tb3nav.sim import Simulator

FIG = ROOT / "results" / "figures"
np.set_printoptions(precision=4, suppress=True)


def identify_actuator():
    """Step the wheel-speed set-point and fit w(t) = K w_cmd (1 - exp(-(t - t0)/tau))."""
    cfg = load_config()
    out = []
    for w_cmd in (1.0, 3.0, 5.0):
        spec = W.WorldSpec(W.default_landmarks(), [], start=(3, 2, 0))
        sim = Simulator(spec, cfg, render_camera=False)
        sim.step(0.5)
        ts, ws = [], []
        sim.set_wheel_speeds(w_cmd, w_cmd)
        t0 = sim.data.time
        for _ in range(300):
            sim.step(0.002)
            ts.append(sim.data.time - t0)
            ws.append(sim.data.qvel[sim.dof_adr[0]])   # identification uses the joint speed
        ts, ws = np.array(ts), np.array(ws)
        f = lambda t, K, tau: K * w_cmd * (1 - np.exp(-t / tau))
        (K, tau), _ = curve_fit(f, ts, ws, p0=(1.0, 0.02))
        out.append({"w_cmd": w_cmd, "gain": float(K), "tau_s": float(tau), "t": ts.tolist(),
                    "w": ws.tolist()})
        sim.close()
    return out


def validate_model(seed=7):
    """Drive random (v, w) commands; integrate the unicycle with ENCODER-derived wheel speeds
    and compare with the simulator (open-loop, 10 s) - checks the calibrated kinematics."""
    cfg = load_config()
    dd = DiffDrive(cfg)
    spec = W.WorldSpec(W.default_landmarks(), [], start=(3, 2, 0))
    sim = Simulator(spec, cfg, render_camera=False, seed=seed)
    rng = np.random.default_rng(seed)
    pkt = sim.step(0.2)
    prev = pkt.enc_ticks * 2 * np.pi / cfg.robot.ticks_per_rev
    s = sim.ground_truth_for_evaluation()      # initial condition only
    pred, true = [s.copy()], [s.copy()]
    for k in range(200):
        if k % 20 == 0:
            sim.set_wheel_speeds(*dd.inverse(rng.uniform(0.03, 0.15), rng.uniform(-0.5, 0.5)))
        pkt = sim.step(0.05)
        a = pkt.enc_ticks * 2 * np.pi / cfg.robot.ticks_per_rev
        ds, dth = dd.forward(*(a - prev), 0.05); prev = a
        s = s + [ds * np.cos(s[2] + dth / 2), ds * np.sin(s[2] + dth / 2), dth]
        pred.append(s.copy()); true.append(sim.ground_truth_for_evaluation())
    return np.array(pred), np.array(true)


def controllability_report():
    cfg = load_config()
    dd = DiffDrive(cfg)
    r, b = float(dd.r(5.0)), dd.b_equiv
    tau = 0.012
    rep = {}
    A, B = M.full_model_linear(np.array([0, 0, 0.3]), 4.0, 4.5, r, b, tau)
    Cm = M.ctrb(A, B)
    rep["full_5state_moving_rank"] = int(np.linalg.matrix_rank(Cm, tol=1e-9))
    A, B = M.full_model_linear(np.array([0, 0, 0.3]), 0.0, 0.0, r, b, tau)
    rep["full_5state_standstill_rank"] = int(np.linalg.matrix_rank(M.ctrb(A, B), tol=1e-9))
    A, B = M.unicycle_jacobians(np.array([0, 0, 0.3]), np.array([0.0, 0.0]))
    rep["unicycle_standstill_rank"] = int(np.linalg.matrix_rank(M.ctrb(A, B)))
    grid = []
    for v_r in (0.0, 0.01, 0.05, 0.10, 0.15):
        for w_r in (0.0, 0.3):
            A, B = M.error_model(v_r, w_r)
            Ad, Bd = M.discretise(A, B, cfg.timing.control_dt)
            sv = np.linalg.svd(M.ctrb(Ad, Bd), compute_uv=False)
            grid.append({"v_r": v_r, "w_r": w_r, "rank": int(np.linalg.matrix_rank(M.ctrb(A, B), tol=1e-9)),
                         "sigma_min": float(sv[-1]), "cond": float(sv[0] / max(sv[-1], 1e-15))})
    rep["error_model"] = grid
    return rep


def observability_report():
    """Observability Gramian rank / min eigenvalue for different sensor sets."""
    lm = np.array([6.0, 3.0, np.pi])      # the dock tag
    lm2 = np.array([3.0, 0.0, np.pi / 2]) # a south-wall tag
    dt, n = 0.05, 100
    cases = {}
    for motion, (v, w) in (("stationary", (0.0, 0.0)), ("driving", (0.12, 0.15))):
        s = np.array([2.0, 2.0, 0.2])
        traj = []
        for _ in range(n):
            traj.append([*s, v, w])
            s = s + dt * np.array([v * np.cos(s[2]), v * np.sin(s[2]), w])
        traj = np.array(traj)
        for name, rows, lms in (("range only, 1 tag", [0], [lm]),
                                ("bearing only, 1 tag", [1], [lm]),
                                ("range+bearing, 1 tag", [0, 1], [lm]),
                                ("range+bearing+yaw, 1 tag", [0, 1, 2], [lm]),
                                ("range+bearing, 2 tags", [0, 1], [lm, lm2]),
                                ("range+bearing, 2 dock tags 10 cm apart", [0, 1],
                                 [np.array([6.0, 2.95, np.pi]), np.array([6.0, 3.05, np.pi])])):
            G = np.zeros((3, 3))
            Phi = np.eye(3)
            for k in range(n):
                for L in lms:
                    H = M.meas_jacobian(traj[k, :3], L, True)[rows]
                    G += Phi.T @ H.T @ H @ Phi
                A, _ = M.unicycle_jacobians(traj[k, :3], traj[k, 3:5])
                Phi = (np.eye(3) + A * dt) @ Phi
            ev = np.linalg.eigvalsh(G)
            cases[f"{motion}: {name}"] = {"rank": int(np.sum(ev > 1e-6 * max(ev.max(), 1e-12))),
                                          "eig_min": float(ev[0]), "eig_max": float(ev[-1])}
        cases[f"{motion}: odometry only"] = {"rank": 0, "eig_min": 0.0, "eig_max": 0.0}
    return cases


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("=" * 70, "\n1. Actuator identification (firmware-closed wheel velocity loop)")
    act = identify_actuator()
    for a in act:
        print(f"  step to {a['w_cmd']:.0f} rad/s: static gain {a['gain']:.3f}, tau = {1000*a['tau_s']:.1f} ms")
    tau = float(np.mean([a["tau_s"] for a in act]))
    print(f"  -> tau = {1000*tau:.1f} ms << control period 50 ms: wheel dynamics are neglected "
          f"in the controller/estimator (reduced 3-state model).")

    print("=" * 70, "\n2. Open-loop model validation (calibrated unicycle vs MuJoCo, 10 s)")
    pred, true = validate_model()
    e = np.hypot(*(pred[:, :2] - true[:, :2]).T)
    eth = np.abs(wrap(pred[:, 2] - true[:, 2]))
    print(f"  position error after 10 s: {100*e[-1]:.1f} cm (max {100*e.max():.1f} cm); "
          f"heading error {np.degrees(eth[-1]):.1f} deg")

    print("=" * 70, "\n3. Controllability")
    cr = controllability_report()
    print(f"  full 5-state model, moving:     rank {cr['full_5state_moving_rank']}/5")
    print(f"  full 5-state model, standstill: rank {cr['full_5state_standstill_rank']}/5")
    print(f"  unicycle at standstill:         rank {cr['unicycle_standstill_rank']}/3  "
          f"(no lateral motion; Brockett: no smooth static feedback stabilises a point)")
    for g in cr["error_model"]:
        print(f"  error model v_r={g['v_r']:.2f} w_r={g['w_r']:.1f}: rank {g['rank']}/3, "
              f"sigma_min(Ctrb_d) = {g['sigma_min']:.2e}")

    print("=" * 70, "\n4. Observability (Gramian over 5 s)")
    obs = observability_report()
    for k, v in obs.items():
        print(f"  {k:40s} rank {v['rank']}/3   lambda_min = {v['eig_min']:.2e}")

    report = {"actuator": [{k: a[k] for k in ("w_cmd", "gain", "tau_s")} for a in act],
              "tau_mean_s": tau,
              "open_loop_validation": {"pos_err_10s_m": float(e[-1]), "pos_err_max_m": float(e.max()),
                                       "heading_err_10s_rad": float(eth[-1])},
              "controllability": cr, "observability": obs}
    (ROOT / "results" / "model_analysis.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a in act:
        ax[0].plot(1000 * np.array(a["t"]), a["w"], label=f"step to {a['w_cmd']:.0f} rad/s")
        t = np.array(a["t"])
        ax[0].plot(1000 * t, a["gain"] * a["w_cmd"] * (1 - np.exp(-t / a["tau_s"])), "k:", lw=1)
    ax[0].set_xlabel("time [ms]"); ax[0].set_ylabel("wheel speed [rad/s]")
    ax[0].set_title(f"Wheel step response, first-order fit tau = {1000*tau:.0f} ms"); ax[0].legend(fontsize=8)
    ax[1].plot(true[:, 0], true[:, 1], "k", label="MuJoCo")
    ax[1].plot(pred[:, 0], pred[:, 1], "r--", label="calibrated model (open loop)")
    ax[1].set_aspect("equal"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"Model validation, 10 s open loop: {100*e[-1]:.1f} cm drift")
    names = [k for k in obs if k.startswith("driving")]
    vals = [max(obs[k]["eig_min"], 1e-12) for k in names]
    ax[2].barh([k.replace("driving: ", "") for k in names], vals, color="tab:blue", label="driving")
    names_s = [k for k in obs if k.startswith("stationary")]
    vals_s = [max(obs[k]["eig_min"], 1e-12) for k in names_s]
    ax[2].barh([k.replace("stationary: ", "") for k in names_s], vals_s, color="tab:orange",
               alpha=0.6, label="stationary")
    ax[2].set_xscale("log"); ax[2].set_xlabel("smallest eigenvalue of observability Gramian")
    ax[2].set_title("Observability by sensor set (1e-12 = unobservable)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "model_analysis.png", dpi=120); plt.close(fig)
    print("wrote results/model_analysis.json and results/figures/model_analysis.png")


if __name__ == "__main__":
    main()
