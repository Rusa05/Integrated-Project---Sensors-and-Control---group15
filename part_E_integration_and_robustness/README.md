# Part E: Integration and robustness

> **Marking guide, Part E (20 marks):** "performance under disturbance and noise".
> Distinction band: "performance is reported over repeated trials with varied conditions, and failures are diagnosed and traced to specific causes in the model, sensing, estimator or controller".

This part covers the complete pipeline in one run (start → navigate → dock), the video recorder, and 80 randomised trials under disturbances, with every failure traced to its cause.

## Steps, in order

| Step | Script | What it does | Time |
|---|---|---|---|
| E1 | `run_mission.py` | Full mission, 3 scenarios × 3 seeds, headless | 4 min |
| E1 | `run_mission.py --viewer` | Watch one mission live in MuJoCo | n/a |
| E1 | `run_mission.py --record mission.mp4` | Record one mission with overlays (demo video) | 1 min |
| E2 | `run_trials.py` | 8 conditions × 10 random trials, 4 processes | 7 min |
| E2 | `run_trials.py --record run.mp4 --only combined --trial 7` | Record one disturbed trial | 1 min |
| E2 | `run_trials.py --report-only` | Redraw the plots from saved results | 5 s |

Run from the project root, for example `python part_E_integration_and_robustness/run_mission.py --viewer`.

- `run_mission.py` options: `--scenario A_cylinder_in_way | B_box_in_way | C_far_corner`, `--seed N`, `--start X Y THETA`.
- `run_trials.py` options: `--trials N`, `--only CONDITION`.
- On macOS use `mjpython` for `--viewer`.

Results: `results/mission/` (E1) and `results/trials/` (E2).

---

## How the pieces fit

- [`src/robot.py`](../src/robot.py) is the robot's hardware interface. It samples encoders every 20 ms, LiDAR at 5 Hz and the camera at 10 Hz, timestamps each camera frame, and sends wheel speeds. It never reads the true pose.
- [`src/mission.py`](../src/mission.py) runs the EKF ([Part C](../part_C_estimation/)), the map, planner and LQR ([Part D](../part_D_control/)), and the state machine ALIGN → TRACK → FINAL_ALIGN → DOCK → DOCKED.
- Ground truth is read only in [`eval/ground_truth.py`](../eval/ground_truth.py), to place the robot, score runs, detect collisions and diagnose failures.

## E1. Full mission (`results/mission/`)

![Mission video frame](results/mission/mission_frame.jpg)

| Metric | 9 runs (3 scenarios × 3 seeds) |
|---|---|
| Docked within 2 cm and 3°, no collisions | **9 / 9** |
| Final position error (truth vs dock goal) | **0.58 cm mean**, 0.60 cm worst |
| Lateral / depth error | 0.0–0.16 cm / −0.51 to −0.60 cm (always slightly short) |
| Final heading error | **0.15° mean**, 0.39° worst |
| EKF error at the dock | 0.29–0.36 cm, against a 3σ of about 0.65 cm (consistent) |
| Mission time | 33.6 s mean |

- **Always slightly short:** docking stops once the estimate is within 1 cm of the goal, approaching from in front.
- **Lateral and heading error so small:** the last 0.5 m ends on the dock axis with the marker in view for most of it.

**Video overlays** (the brief asks for "the estimate and its uncertainty alongside the true state"):

| On screen | Meaning |
|---|---|
| Blue arrow + ellipse | EKF estimate and its 3σ position uncertainty |
| Blue / black dots | Estimated / true path so far |
| Orange dots | Current planned path |
| Top-left panel | State, EKF pose, true pose, error, 3σ, marker in view |
| Top-right inset | The robot's camera, detected marker outlined in green (greyed out while occluded) |
| Banner (trials only) | Active disturbances: push, occlusion, latency, slip |

Recorded examples:

- `results/mission/mission_A.mp4`: nominal, 35 s.
- `results/trials/combined_trial7.mp4`: pushed at 8 s, slippery tyres, 0.1 s latency, marker blocked for 10 s during the approach. Docks at 0.64 cm.

## E2. Robustness trials (`results/trials/`)

**Every trial** gets:

- a random start pose (anywhere with clearance, any heading);
- random positions for the box (and its rotation) and the cylinder;
- a random initial-estimate error drawn from P0.

Trial *n* uses the same layout in every condition, so conditions compare like for like. The robot's code never knows which condition it's in.

| Condition | What changes | Where it acts |
|---|---|---|
| nominal | nothing else | n/a |
| push | 8 N sideways shove for 0.2 s at a random time (6–20 s), random direction | world |
| slip | tyre friction 1.0 → 0.1 | world |
| occlusion | marker detections dropped for a random 10 s window | sensor ([`src/robot.py`](../src/robot.py)) |
| latency | each marker measurement delivered 0.2 s after its frame | sensor |
| latency_nocomp | the same, with the EKF's latency compensation off | for comparison |
| noise_x2 | every sensor's noise doubled | sensor |
| combined | push + slip + occlusion + 0.1 s latency | both |

### Results: final code, 80 trials

| Condition | Docked (2 cm / 3°) | Dock error mean / worst | Worst EKF error during run (median) |
|---|---|---|---|
| nominal | **10 / 10** | 0.53 / 0.69 cm | 8.0 cm |
| push | **10 / 10** | 0.52 / 0.63 cm | 7.6 cm |
| slip | **10 / 10** | 0.52 / 0.65 cm | 8.2 cm |
| occlusion | **10 / 10** | 0.61 / 0.99 cm | 7.8 cm |
| latency 0.2 s | **10 / 10** | 0.57 / 0.64 cm | 9.7 cm |
| latency 0.2 s, no compensation | 10 / 10 | 0.61 / **1.24** cm | 9.1 cm |
| noise ×2 | **10 / 10** | 0.56 / 0.63 cm | 10.5 cm |
| combined | **10 / 10** | 0.57 / 0.86 cm | 9.6 cm |

No collisions; mission time about 30 s in every condition. Mid-run EKF errors of 5–35 cm happen (pushes, blind stretches), but the robot always recovers before docking.

![Success by condition](results/trials/success_by_condition.png)
![Example runs](results/trials/example_runs.png)

### Failures found, diagnosed and fixed

The first full batch docked **59 / 70** in the seven real conditions, and **1 / 10** with uncompensated latency. Each failure was classified automatically (`run_trial` in `run_trials.py`):

- **Collision:** was the EKF more than 8 cm off at the moment of contact?
- **Docking miss:** was the EKF's position or heading off at the dock?

Failing runs were then replayed with a full EKF log.

| # | Symptom | Diagnosis | Part | Fix |
|---|---|---|---|---|
| 1 | Latency: "docked" 6–44 cm from the pad, confident it was right (9 / 10 failed) | Measurements fused as if they described the robot now; they described it 0.2 s earlier (about 9° of heading at 0.8 rad/s) | **Estimation** | **Latency compensation:** timestamped frames; the EKF rewinds 1 s of history to the capture time, updates there, replays odometry. Unit-tested to match exactly |
| 2 | Noise ×2: docked 16 cm off | Three range + bearing updates at 2.6–2.8 m were accepted and each made the error worse (5 → 19 cm). Without the yaw, one marker can't fix heading: rank 2, [Part A](../part_A_modelling/) | **Estimation** (observability) | Fuse the marker only within **2 m**, where the full pose is used |
| 3 | Occlusion, noise ×2: EKF 17–49 cm off but rejecting every reading | **Gate lock-out:** a bad start (13 cm, 7°) made every reading fail the χ² gate; the error grew while the filter still claimed ±15 cm | **Estimation** (consistency) | After 5 rejections in a row (1 s), inflate P on each further rejection. That run recovered after 4 inflations, docked at 0.65 cm |
| 4 | Six runs on one layout: right place, 3.5–8.6° crooked | The EKF heading was right; the docking path was a straight line from wherever the robot stopped | **Planning** | Hermite curve ending on the dock axis. Now 0.15° |

**Is latency compensation still needed now the EKF has safeguards?** With fixes 2 and 3, uncompensated latency also docks 10/10, but its worst error doubles (1.24 vs 0.64 cm), and before the safeguards it failed 9 times in 10. The safeguards are a second layer of defence; compensation is the correct fix.

**Knock-on effects, re-checked:**

- **Part C:** EKF test RMSE rose from 1.5 to 2.2 cm (the test reacquires the marker beyond 2 m); consistency unchanged.
- **Part D:** mid-route EKF error rose from 4.1 to 5.8 cm.
- **Docking:** improved from 0.41° to 0.15°.

## Limitations

- **Sample size:** 10/10 per condition bounds the true success rate above about 69 % (95 % confidence); 80/80 overall, above about 95 %.
- **Static obstacles:** the replanning and safety stop were never needed in 80 runs.
- **Push size:** limited to 8 N for 0.2 s. A large "kidnap" isn't recoverable without a marker-search behaviour.
