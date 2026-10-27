# Part D: Control (mapping, planning, LQR tracking, docking)

> **Marking guide, Part D (20 marks):** "controller design, tuning and tracking results".
> Brief: "Navigate a room with obstacles using the LiDAR … state-feedback or LQR trajectory tracker. Constraint: the controller must use the estimated state, never ground truth."

This part covers the LiDAR map, the A* planner and trajectory timing, the LQR tracker, and the docking controller, all driven only by the EKF estimate. The full start-to-dock runs and the robustness tests are in [Part E](../part_E_integration_and_robustness/).

## Steps, in order

| Step | Script | What it does | Time |
|---|---|---|---|
| D1 | `run_navigation.py` | Start → pre-dock point around obstacles, 3 scenarios × 3 seeds, plus the LQR weight study | 4 min |
| D2 | (docking) | Docking controller: design below; results come from the full mission in Part E | n/a |

```bash
python part_D_control/run_navigation.py           # full test
python part_D_control/run_navigation.py --quick   # one run per scenario
```

Code: [`src/mapping.py`](../src/mapping.py), [`src/planner.py`](../src/planner.py), [`src/controller.py`](../src/controller.py), [`src/mission.py`](../src/mission.py).

---

## Pipeline (every 20 ms)

```
encoders ──► EKF predict ◄── camera 10 Hz → marker → EKF update (≤5 Hz)
                │
LiDAR 5 Hz ──► occupancy grid (placed with the EKF pose) ──► A* ──► smoothed, timed reference
                │                                                       │
                └──────────── EKF pose ──► LQR tracker ◄────────────────┘ ──► wheel speeds
```

## Design and parameter values

| Component | Choice | Why |
|---|---|---|
| Map | Log-odds occupancy grid, 5 cm cells; hit +0.85, pass-through −0.4; no-return beams ignored | Single noisy or dropped beams can't flip a cell. Nothing about the room is known in advance |
| Beam placement | LiDAR offset (−0.035 m) + EKF pose | Frame chain from [Part B](../part_B_sensing/) |
| Planner | A*, 8-connected; cells within **0.25 m** of an obstacle forbidden, extra cost within a further 0.2 m | 0.25 m = radius of the circle containing the Waffle Pi footprint around its axle |
| Smoothing | Line-of-sight shortcuts that keep 0.45 m clearance where possible, then iterative smoothing | Leaves room to round corners |
| Timing | Cruise 0.18 m/s, lateral acceleration ≤ 0.15 m/s², acceleration ≤ 0.2 m/s², wheel limits | Leaves headroom below the 0.26 m/s top speed for feedback |
| Heading and curvature | 41 cm Savitzky-Golay derivative | Removes grid ripple from the feed-forward turn rate |
| LQR | Discrete LQR on the linearised tracking error ([Part A](../part_A_modelling/)), zero-order hold at 50 Hz, gain-scheduled on reference speed. u = [v_r cos e₃, ω_r] − K e | Standard state-feedback tracker for a unicycle |
| LQR weights | **Bryson's rule:** Q = diag(1/0.05², 1/0.05², 1/0.15²), R = diag(1/0.10², 1/0.8²) | Allowed error 5 cm along, 5 cm across, 0.15 rad heading; allowed correction 0.1 m/s and 0.8 rad/s |
| Saturation | Scale v and ω together if a wheel exceeds 7.88 rad/s | Keeps the path curvature |
| Safety | Replan if the path becomes blocked (checked every 1 s); stop and replan if LiDAR sees something within 0.18 m ahead | n/a |

### Why turn on the spot first?

When the reference is stationary, the tracking error loses controllability: rank 2 ([Part A](../part_A_modelling/)). A wheeled robot can't correct a sideways error without moving forward, and the Riccati equation has no stabilising solution at v = ω = 0. So the mission turns on the spot to the path heading (ALIGN state) before the LQR takes over, and the gain uses a slow crawl speed whenever the reference stops.

## D1. Navigation results: 9 runs, final code (`results/`)

| Scenario | Start | In the straight-line path |
|---|---|---|
| A | (−1.5, −0.8), facing east | Green cylinder |
| B | (0.5, 1.5), facing south | Orange box |
| C | (−1.5, 1.5), facing −34° | None (longest path) |

Goal: the pre-dock pose (1.30, −1.00), 0.6 m in front of the marker, facing it.

| Metric | Mean | Worst |
|---|---|---|
| Reached pre-dock without collision | **9 / 9** | n/a |
| Final position error (truth) | **0.65 cm** | 0.8 cm |
| Final heading error (truth) | **0.6°** | 0.8° |
| Cross-track error, path vs **EKF** (what the controller can see) | **1.6 cm RMS** | 18 cm peak |
| Cross-track error, path vs **truth** | 4.7 cm RMS | 19 cm peak |
| EKF position error | 5.8 cm RMS | n/a |
| Time | 24.3 s | 27.7 s |

![Navigation, scenario B](results/nav_B_box_in_way.png)

**The gap between the two tracking rows is estimation error, not control error.** The plots show this directly:

- **Before the marker is in range:** the robot starts 2.9–3.6 m from it, and the EKF only fuses it within 2 m. The initial heading uncertainty (σ = 3°) grows into a sideways error of up to about 9 cm. The controller can't see this, so it keeps the *estimate* on the path.
- **When the marker comes into range:** the EKF corrects in one step, the controller sees the true error, and it removes it in about 4 s.

The controller is accurate; the limit is how well the robot knows where it is before it can see the marker.

**Trade-off of the 2 m marker range limit** (added after the robustness trials in Part E):

- With the first EKF, which used readings out to about 2.7 m, the same runs gave 0.9 cm control error, 2.9 cm true cross-track error and 4.1 cm EKF error.
- The limit costs that mid-route accuracy, but it removed docking failures where far updates made the estimate worse. Docking accuracy is unchanged.

### LQR weights: control effort vs tracking error (`lqr_tradeoff.png`)

| R scale | Feedback turn effort (RMS) | Cross-track error vs truth (RMS) |
|---|---|---|
| ×0.1 (aggressive) | 0.144 rad/s | 5.03 cm |
| **×1 (chosen)** | **0.096 rad/s** | **5.13 cm** |
| ×10 (gentle) | 0.062 rad/s | 5.23 cm |

- 10× more aggressive: +50 % effort for −2 % error.
- 10× gentler: −35 % effort for +2 % error.

The curve is flat because most of the remaining error is estimation error, which no control gain can remove. The chosen weights sit at the knee.

## D2. Docking controller

States in [`src/mission.py`](../src/mission.py): ALIGN → TRACK → **FINAL_ALIGN → DOCK** → DOCKED.

1. **FINAL_ALIGN:** at the pre-dock pose, turn on the spot to face the marker (within 0.5°). The marker is well in view, so the EKF gets good pose updates.
2. **DOCK:** a 0.48 m reference to the dock goal (1.78, −1.0) at 0.06 m/s, tracked by **the same LQR**. The reference is a cubic Hermite curve that starts along the robot's heading and **ends along the dock axis**, so the robot finishes square even if it reached the pre-dock point a few cm off-axis.
3. **Blind finish:** the marker leaves the camera view when the axle is about 0.30 m from it. The last ~0.17 m runs on odometry, which drifts under 1 mm over that distance.
4. **Stop** when the reference has ended and the estimate is within 1 cm of the goal.
5. **LiDAR backstop:** at the dock the LiDAR should read 0.165 m to the backplate. Below 0.13 m the robot stops at once. It's an independent check that doesn't rely on the estimate.

**Why reuse the LQR for docking:**

- It's already tuned.
- A moving reference keeps v_r ≠ 0, so the error stays controllable (rank 3) all the way in.
- A pose-stabilising controller would face the rank-2 problem at standstill.

**Results** (full mission, [Part E](../part_E_integration_and_robustness/)): 9 / 9 docked, **0.58 cm and 0.15°** from the goal; 80 / 80 under disturbances.

## Problems found and fixed during development

1. **Sharp corner at the obstacle.** Shortcuts ran exactly along the inflation edge, leaving no room to round corners: turn spikes of 1.4 rad/s. Shortcuts now keep 0.45 m clearance where possible.
2. **Turn-rate wobble** (±0.04 rad/s at about 3 Hz). The feedback term was near zero, so the source was the feed-forward. Curvature is a second derivative of the path, so 5 cm grid kinks were amplified. Fixed with the 41 cm Savitzky-Golay derivative, after first ruling out the gain schedule.
3. **Final heading always 2° off.** The turn-to-face stopped at the edge of its 2° tolerance. Tightened to 0.5°.
4. **Crooked docking** (3.5–8.6°, found in Part E). The docking path was a straight line from wherever the robot stopped. Now a Hermite curve ending on the dock axis.

## Limitations

- Obstacles are static. Replanning and the safety stop exist but were never triggered.
- Unknown cells count as free; fine here, as the first scan sees most of the room.
- The reference is time-indexed; a governor pauses it if the robot falls more than 10 cm behind.
