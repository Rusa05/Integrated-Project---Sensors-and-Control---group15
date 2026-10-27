"""Part A (Modelling): checks of the state-space model against the simulated robot.

    * controllability of the linearised tracking error, observability for each sensor combination
    * wheel actuator step response (justifies dropping motor dynamics: kinematic model)
    * odometry drift on an open-loop figure-eight (what the model + encoders alone achieve)

Run from the project root:   python part_A_modelling/model_analysis.py      (~15 s)
Writes figures and results/summary.json next to this script. Ground truth is read only through
eval/ground_truth.py, to score the odometry.
"""
from pathlib import Path
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config, model as rm                                   # noqa: E402
from src.sensors import Sensors, NoiseConfig, Odometry                # noqa: E402
from eval import ground_truth as gt                                   # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
STEPS_PER_TICK = round(rm.DT / 0.002)


def load():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def drive(model, data, v, omega):
    data.ctrl[[model.actuator('wheel_left').id, model.actuator('wheel_right').id]] = rm.body_to_wheels(v, omega)
    mujoco.mj_step(model, data, nstep=STEPS_PER_TICK)


# --------------------------------------------------------------------------
def model_analysis():
    print('\n== Part A: controllability of the linearised tracking error ==')
    ctrb = {}
    for v_r, w_r in [(0.15, 0.0), (0.15, 0.5), (0.0, 0.5), (0.0, 0.0)]:
        A, B = rm.error_dynamics(v_r, w_r)
        rank, _ = rm.controllability_rank(A, B)
        ctrb[f'v={v_r},w={w_r}'] = int(rank)
        print(f'  v_ref={v_r:4.2f} omega_ref={w_r:3.1f}: rank {rank}/3')

    print('\n== Part A: observability over 50 steps (1 s) ==')
    m = np.array(config.MARKER_XY)
    # the two small dock tags, 10 cm apart (src/config.py TAGS 7 and 8)
    m7, m8 = (np.array([config.TAGS[i].x, config.TAGS[i].y]) for i in (7, 8))
    x0 = np.array([0.8, -0.8, 0.0])
    obs = {}
    for label, u, meas in [
        ('odometry only, moving', (0.15, 0.2), None),
        ('range+bearing, stationary', (0.0, 0.0), 'rb'),
        ('range+bearing, moving', (0.15, 0.2), 'rb'),
        ('range only, moving', (0.15, 0.2), 'r'),
        ('bearing only, moving', (0.15, 0.2), 'b'),
        ('range+bearing+marker yaw, stationary', (0.0, 0.0), 'rby'),
        ('two tags 10 cm apart, range+bearing, stationary', (0.0, 0.0), 'rb2'),
    ]:
        x, Fs, Hs = x0.copy(), [], []
        for _ in range(50):
            H = None
            if meas == 'rb':
                H = rm.H_range_bearing(x, m)
            elif meas == 'r':
                H = rm.H_range_bearing(x, m)[:1]
            elif meas == 'b':
                H = rm.H_range_bearing(x, m)[1:]
            elif meas == 'rby':
                H = rm.H_range_bearing_yaw(x, m)
            elif meas == 'rb2':
                H = np.vstack([rm.H_range_bearing(x, m7), rm.H_range_bearing(x, m8)])
            Fs.append(rm.jacobian_state(x, u))
            Hs.append(H)
            x = rm.f_discrete(x, u)
        rank, O = rm.observability_rank(Fs, Hs)
        sv = np.linalg.svd(O, compute_uv=False) if O.size else np.array([0.0])
        cond = float(sv[0] / sv[-1]) if rank == 3 else float('inf')
        obs[label] = {'rank': int(rank), 'condition': cond}
        print(f'  {label:48s} rank {rank}/3   cond {cond:.1f}')
    return {'controllability_rank': ctrb, 'observability': obs}


# --------------------------------------------------------------------------
def actuator_step():
    model, data = load()
    l, r = model.actuator('wheel_left').id, model.actuator('wheel_right').id
    adr = model.sensor('speed_left').adr[0]
    t, w = [], []
    cmd = 6.0
    for _ in range(250):
        data.ctrl[[l, r]] = cmd
        mujoco.mj_step(model, data)
        t.append(data.time)
        w.append(data.sensordata[adr])
    t, w = np.array(t), np.array(w)
    gain = w[-50:].mean() / cmd
    tau = float(t[np.argmax(w >= 0.632 * w[-50:].mean())])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(t * 1000, w, label='wheel speed (sim)')
    ax.plot(t * 1000, cmd * gain * (1 - np.exp(-t / tau)), '--', label=f'1st order, tau={tau*1000:.0f} ms')
    ax.axvline(rm.DT * 1000, color='grey', lw=0.8, ls=':', label='control period')
    ax.set(xlabel='time (ms)', ylabel='rad/s', title='Wheel speed step response (6 rad/s command)')
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'actuator_step.png', dpi=130); plt.close(fig)
    print(f'\n== Actuator: steady-state gain {gain:.3f}, time constant {tau*1000:.1f} ms ==')
    return {'gain': float(gain), 'tau_s': tau}


# --------------------------------------------------------------------------
def odometry_drift():
    """Open-loop drive (commands only, no feedback) and compare dead reckoning to truth."""
    # Two figure-eights of radius 0.3 m around the start; clear of both obstacles.
    loop = 2 * math.pi / 0.4
    schedule = [(0.12, 0.4, loop), (0.12, -0.4, loop)] * 2 + [(0.0, 0.0, 0.5)]
    runs = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, noise in [('ideal encoders', NoiseConfig.ideal()), ('noisy encoders', NoiseConfig())]:
        model, data = load()
        sens = Sensors(model, data, noise, seed=1)
        odo = Odometry(sens.encoders(), gt.true_pose(model, data))
        truth, est, t = [], [], []
        for v, w, dur in schedule:
            for _ in range(int(round(dur / rm.DT))):
                drive(model, data, v, w)
                odo.update(sens.encoders(), rm.DT)
                truth.append(gt.true_pose(model, data)); est.append(odo.pose.copy()); t.append(data.time)
        truth, est, t = np.array(truth), np.array(est), np.array(t)
        pos_err = np.hypot(*(est[:, :2] - truth[:, :2]).T)
        yaw_err = np.array([rm.wrap(a) for a in est[:, 2] - truth[:, 2]])
        dist = np.sum(np.hypot(*np.diff(truth[:, :2], axis=0).T))
        runs[name] = {'distance_m': float(dist), 'final_pos_err_m': float(pos_err[-1]),
                      'final_yaw_err_deg': float(math.degrees(yaw_err[-1])),
                      'drift_pct_of_distance': float(100 * pos_err[-1] / dist)}
        axes[0].plot(est[:, 0], est[:, 1], '--', label=f'odometry ({name})')
        axes[1].plot(t, pos_err * 100, label=name)
        if name == 'ideal encoders':
            axes[0].plot(truth[:, 0], truth[:, 1], 'k', lw=2, label='ground truth')
    axes[0].set(aspect='equal', xlabel='x (m)', ylabel='y (m)', title='Open-loop figure-eight')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='time (s)', ylabel='position error (cm)', title='Dead-reckoning drift')
    axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / 'odometry_drift.png', dpi=130); plt.close(fig)
    print('\n== Odometry drift ==')
    for k, r in runs.items():
        print(f'  {k:15s} {r["distance_m"]:.2f} m driven, final error {r["final_pos_err_m"]*100:.1f} cm '
              f'({r["drift_pct_of_distance"]:.2f}%), yaw {r["final_yaw_err_deg"]:.2f} deg')
    return runs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {'model': model_analysis(), 'actuator': actuator_step(), 'odometry': odometry_drift()}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(f'\nWrote figures and summary.json to {OUT}')


if __name__ == '__main__':
    main()
