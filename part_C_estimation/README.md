# Part C: Estimation (EKF)

> **Marking guide, Part C (20 marks):** "KF/EKF or observer design, tuning and reported performance".
> Brief: "EKF fusing wheel-encoder odometry with range and bearing, or pose, measurements from the camera … the controller must use the estimated state, never ground truth."

**Goal:** one best estimate of the robot pose [x, y, θ], with its uncertainty, from wheel odometry and the camera's view of the ArUco tags.

- Odometry is smooth but drifts and never recovers.
- The tags give an absolute pose, but with noise that grows with distance, and only when one is in view.

The EKF blends the two, trusting each according to its *measured* noise.

## Steps, in order

| Step | Script | What it does | Time |
|---|---|---|---|
| C1 | `run_ekf_test.py` | 20 Monte Carlo runs of a fixed drive: accuracy, NEES/NIS consistency, and the R ×0.1 / ×1 / ×10 study | 6 min |
| C2 | `tradeoff_studies.py` | On the full mission (Part E random layouts): Study A, process-noise scale under pushes; Study D, which measurements matter | 3 min |

```bash
python part_C_estimation/run_ekf_test.py            # add --quick for 4 runs
python part_C_estimation/tradeoff_studies.py
```

Code: [`src/ekf.py`](../src/ekf.py).

---

## The filter

| Step | When | What |
|---|---|---|
| Predict | Every tick (50 Hz) | Unicycle model driven by encoder increments. Q is built from wheel slip, σ = 1 % of each wheel's travel, pushed through the Jacobian |
| Update | Each known tag in view, at most 5 frames/s | z = [range, bearing, tag yaw relative to the robot]; the tag's map pose comes from `src/config.py`. Several tags in one frame are all fused |
| IPPE ambiguity | Each update | The vision keeps both planar pose solutions; the filter uses the one that agrees with its estimate |
| Gate | Each update | χ² test (3 dof, 99 %). If the full pose fails, it retries with range + bearing only (2 dof) |
| Range limits | Each update | Scaled by tag size (k = size / 10 cm): range + bearing fused to 2k m; yaw only to **1.75k m** and only from tags ≥ 5 cm |
| New information only | Each frame | A frame is fused only once the robot has moved 2 cm or turned 2° since the last fused frame (or after 2 s) |
| Lock-out recovery | After 5 rejections in a row | P inflated on each further rejection until readings get through |
| Latency | Late frames | Rewinds to the frame's capture time, updates there, replays odometry |

### Parameter values and why

| Parameter | Value | Why |
|---|---|---|
| Q: `SLIP_FRAC` | 1 % of wheel travel | Part A drift was 0.35 %. 0.5 % made the filter over-confident when blind; 2 % pessimistic. Study A below confirms 1 % |
| Encoder rounding/jitter in Q | **Not included** | Errors on an *absolute* angle cancel between increments, so they don't accumulate. Including them at 50 Hz made the filter far too pessimistic |
| R: range, bearing, yaw | Part B fits (σ = a·r^p, scaled by tag size), floors 2 cm / 0.5° / 2°, **× 3** | The fits capture frame-to-frame noise. The floors and the ×3 cover the viewpoint-dependent error that repeats frame after frame (below) |
| Yaw range | 1.75 × the reference range; tags ≥ 5 cm only | Part B: yaw is 12–16° RMS at 2× the reference range, and small tags read a consistently biased yaw |
| P0 | (5 cm, 5 cm, 3°)² | Uncertainty of placing the robot at the start |

### How R was tuned with several tags (a failure found and fixed)

Moving from one marker to nine tags made the filter **about 3× over-confident**: mean NEES 6.4 against an ideal 3, over 6 random missions. Breaking NEES down by mission state and axis localised it:

- it sat at the dock and wherever the robot stood still or turned on the spot (NEES 14–24);
- it was mostly in **heading**, with a constant 0.3–0.5° error while the filter claimed 0.12°.

Three causes, each measured:

1. **Small tags give a biased yaw** (Part B: ±0.4° to ±5°, consistent): they now contribute range and bearing only.
2. **Same viewpoint, same error:** frames are now fused only after the robot actually moves.
3. **One poorly constrained direction:** the dock tags sit within 10 cm of each other, so a sideways offset paired with a heading error looks almost the same from 0.6 m (Part A: condition number 40 against 2.8 with yaw). Each tag carries less independent information than its noise model suggests.

Fixes 1 and 2 are principled but barely moved NEES (6.3–6.4). Scaling R by 3 gave **mean NEES 3.0 with unchanged accuracy** (1.47 vs 1.44 cm), so R ×3 is the default (`R_INFLATION` in `src/ekf.py`).

## C1. Monte Carlo results (20 runs, `results/`)

The fixed open-loop drive:

1. approach a wall tag;
2. turn away and drive 0.9 m;
3. turn back and reacquire.

Each run draws new sensor noise and a new initial error from P0; ground truth is used only for scoring.

| Metric | EKF | Odometry only |
|---|---|---|
| Position RMSE (whole run, including start-up) | **2.4 cm** | 8.1 cm |
| Heading RMSE | **0.79°** | 2.6° |
| Final position error | **0.23 cm** | 9.0 cm |
| Truth inside ±3σ | **100 %** of time steps | n/a |
| Average NEES (ideal 3; 95 % bounds 2.0–4.2) | **3.3** | n/a |
| Updates: full pose / range + bearing / rejected | 1652 / 603 / 0 | n/a |

- **Odometry-only error** is mostly the initial placement error, which odometry can never correct.
- **NIS averages 0.16:** the expected sign of R being sized for repeating errors, not frame-to-frame noise.

**What if R were 10× smaller or larger?** (`r_scaling.png`, 10 runs each)

| R scale | Position RMSE | Heading RMSE | NEES | Rejected updates |
|---|---|---|---|---|
| ×0.1 | 2.04 cm | 0.79° | **19.6** (badly over-confident) | 12 / 1128 |
| **×1 (chosen)** | **2.12 cm** | **0.82°** | **3.4** (consistent) | 0 / 1128 |
| ×10 | 2.38 cm | 0.92° | 1.2 (pessimistic) | 0 / 1128 |

- **Trusting the camera 10× more** gains 1 mm of accuracy, but the filter becomes six times more certain than it should be and starts rejecting good readings. That is dangerous, because the gate, planner and docking logic all rely on P.
- **Trusting it 10× less** costs accuracy and makes the filter slow to correct.

## C2. Studies on the full mission (`results/tradeoff_studies.png`)

Each uses 6 random layouts from Part E.

**Study A: consistency against responsiveness.** Process noise scaled, with the robot pushed (8 N, 0.2 s) part-way through:

| Q scale | Docked | Mean NEES (ideal 3) | Max NEES | EKF RMSE | Dock error |
|---|---|---|---|---|---|
| ×0.3 | 6 / 6 | 4.3 (over-confident) | 39 | 1.47 cm | 0.55 cm |
| **×1 (chosen)** | 6 / 6 | **3.2** | 22 | 1.47 cm | 0.56 cm |
| ×3 | 6 / 6 | 2.0 | 10 | 1.47 cm | 0.56 cm |
| ×10 | 6 / 6 | 1.3 (pessimistic) | 6 | 1.48 cm | 0.53 cm |

- The chosen Q is the one that's consistent under pushes.
- With tags in view almost everywhere, the next camera frame corrects a push before the error passes 3 cm, so accuracy barely depends on Q. **Q mainly decides whether the filter's uncertainty is honest.**
- With the earlier single marker, the same push left the estimate 5–35 cm off until the marker came back into view.

**Study D: which measurements matter** (nominal conditions):

| Measurements used | Docked | EKF RMSE | Dock error mean (worst) |
|---|---|---|---|
| **Range + bearing + tag yaw** | **6 / 6** | **1.47 cm** | **0.53 cm (0.68)** |
| Range + bearing only | 6 / 6 | 1.68 cm | 0.67 cm (1.52) |
| Odometry only (no camera) | **0 / 6** | 7.58 cm | 8.08 cm (13.5) |

- **The camera is essential:** dead reckoning alone never docks.
- **Yaw helps.** Without it the worst dock is 2× worse. Claudia found yaw added nothing in her room, where several tags are usually in view at once. Ours has fewer tags visible at a time, so the yaw of a single tag still carries information. The difference is a real design point, not a contradiction.

## Problems found and fixed

| Problem | Cause | Fix |
|---|---|---|
| Too pessimistic when blind (first tuning) | Encoder rounding put in Q, but it doesn't accumulate | Left out of Q |
| Over-confident after watching the marker (first tuning) | Same viewpoint repeats the same error | R floors, at most 5 frames/s |
| Far updates made the estimate worse (Part E) | Range + bearing to one tag can't fix heading | Size-scaled range limit |
| Gate lock-out (Part E) | A bad start made every reading fail the gate | P inflation after 5 rejections |
| Stale measurements with camera latency (Part E) | Fused as if current | Rewind-and-replay |
| 3× over-confident with nine tags | Biased small-tag yaw, repeating viewpoint errors, clustered dock tags | Small tags: range + bearing only; fuse only after moving; R ×3 |

## Limitations

- The camera simulation has no lens distortion, blur or rolling shutter, so R would need re-measuring on hardware.
- The R ×3 is an empirical correction for errors that repeat across frames. A filter that models that correlation directly (for example by adding a per-tag bias state) would be the principled next step.
- NEES is consistent on average, but individual moments still go over-confident (max NEES 22 under pushes).
