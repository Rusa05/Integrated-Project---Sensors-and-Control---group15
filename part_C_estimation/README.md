# Part C: Estimation (EKF)

> **Marking guide, Part C (20 marks):** "KF/EKF or observer design, tuning and reported performance".
> Brief: "EKF fusing wheel-encoder odometry with range and bearing, or pose, measurements from the camera … the controller must use the estimated state, never ground truth."

**Goal:** one best estimate of the robot pose [x, y, θ], with its uncertainty, from wheel odometry and the camera.

**Why:**

- Odometry is smooth but drifts and never recovers.
- The camera gives an absolute position, but only when the marker is in view, and its noise grows with distance.

The EKF blends the two, trusting each according to its measured noise.

## Run it

```bash
python part_C_estimation/run_ekf_test.py
```

This does 20 Monte Carlo runs plus the R-scaling study and takes about 15 minutes. Add `--quick` for 4 runs in about 1 minute. Figures and `summary.json` go to `results/`.

## The filter ([../src/ekf.py](../src/ekf.py))

| Step | When | What |
|---|---|---|
| Predict | Every tick (50 Hz) | Unicycle model driven by encoder increments. Q is built from wheel slip: σ = 1 % of each wheel's travel, pushed through the Jacobian to x, y, θ |
| Update | Marker seen (at most 5 Hz) | z = [range, bearing, marker yaw relative to the robot]. Range has the Part B bias removed. R comes from the Part B noise fits, with floors |
| Gate | Each update | χ² test on the innovation (3 dof, 99 %). If the full pose fails, it retries with range + bearing only (2 dof). Rejects pose flips |
| Range limit | Each update | Readings further than **2 m** are ignored: without the yaw, one marker can't fix heading and far updates did harm (found in Part E) |
| Lock-out recovery | After 5 rejections in a row | P is inflated on each further rejection until readings get through again (Part E) |
| Latency | Late frames | Rewinds to the frame's capture time, updates there, replays odometry (Part E) |
| Blind | Marker not visible | Predict only; the uncertainty grows |

`EKF.pose_from_marker()` turns a single detection into a pose, for starting without a prior (used later for random starts).

### Parameter values and why

| Parameter | Value | Why |
|---|---|---|
| `SLIP_FRAC` | 1 % of wheel travel | Part B drift was 0.35 %. 0.5 % left NEES too high (6.9) when blind; 1 % passes; 2 % is pessimistic |
| Encoder rounding/jitter in Q | **Not included** | Errors on an *absolute* angle cancel between successive increments, so they don't accumulate. Including them at 50 Hz made the filter far too pessimistic when blind |
| R range | 1.1 mm·r^3.3, floor **2 cm** | The fit is from Part B. The floor covers the systematic per-viewpoint error (about 0.5 cm, the same every frame) |
| R bearing | 0.007°·r^1.1, floor **0.5°** | The simulated camera has no distortion or blur, so the raw fit is optimistic |
| R marker yaw | 0.63°·r^2.7, floor **2°** | Used only within 2 m (IPPE ambiguity grows beyond that) |
| Update rate | **5 Hz** (of 10 Hz frames) | Consecutive frames repeat nearly the same error. Fusing all 10 Hz counts it twice |
| P0 | (5 cm, 5 cm, 3°)² | Uncertainty of placing the robot at the start |

These numbers are from the final filter. The first version (without the 2 m range limit) gave 1.5 cm / 0.43° on the same test. This drive reacquires the marker at about 2.2 m, so the final filter waits slightly longer for its first update. The robustness trials (Part E) showed the range limit is worth that: far updates caused docking failures.

## The test

An open-loop drive, so no feedback and no ground truth in the loop:

1. Face the marker and approach to about 1.4 m.
2. Turn 180° and drive 0.9 m away. The marker is lost for about 15 s.
3. Turn back, approach, reacquire the marker.
4. Drift out of view for the last 4 s.

Each run draws new sensor noise and a new initial error sampled from P0. Ground truth (`eval/ground_truth.py`) is used only for scoring.

## Results: 20 runs (`results/`)

| Metric | EKF | Odometry only |
|---|---|---|
| Position RMSE (whole run, including start-up) | **2.2 cm** | 8.1 cm |
| Heading RMSE | **0.77°** | 2.6° |
| Final position error | **0.43 cm** | 9.0 cm |
| Truth inside ±3σ | **100 % of time steps** | n/a |
| Average NEES (ideal 3; 95 % bounds 2.0–4.2) | **2.3** | n/a |
| Updates: full pose / range + bearing / rejected | 1084 / 8 / 0 | n/a |

The odometry-only error is mostly the initial placement error, which odometry can never correct.

- `ekf_run.png`: one run. The 3σ ellipses grow while blind and shrink when the marker returns. The error stays inside the band.
- `ekf_consistency.png`:
  - **NEES**:
    - The average over the run is 2.3, inside the bounds.
    - While the marker is in view it sits *below* the band, so the filter is cautious there: the price of the R floors.
    - It peaks at about 6.5 just before reacquisition (t ≈ 22 s), after two blind 180° turns, so turning-on-the-spot slip is slightly under-modelled.
  - **NIS**: averages 0.33 (ideal 3), all below the 95 % line. Low NIS is the expected sign of the R floors. Innovations are small because the camera's error is consistent frame to frame, while R is sized for the systematic part.

### What if R were ten times smaller or larger? (`r_scaling.png`, 10 runs each)

| R scale | Position RMSE | Heading RMSE | NEES | Rejected updates |
|---|---|---|---|---|
| ×0.1 | 1.89 cm | 0.77° | **8.0** (over-confident) | 11 / 545 |
| **×1 (chosen)** | **1.94 cm** | **0.79°** | **2.34** (in bounds 1.7–4.7) | 0 / 545 |
| ×10 | 3.11 cm | 1.14° | 1.58 (pessimistic) | 0 / 545 |

- **R ×0.1:** the filter believes the camera too much. Accuracy is about the same (1.89 vs 1.94 cm), but its uncertainty is now badly wrong: NEES 8, nearly three times what it should be, and its gate starts rejecting good measurements (11). An over-confident filter is dangerous because the planner, the gate and the docking logic all trust its 3σ. (Before the Part E safeguards, R ×0.1 also had *worse* accuracy, 1.75 vs 1.61 cm, with 61 rejections.)
- **R ×10:** the filter barely listens to the camera, so it corrects slowly and accuracy roughly halves.
- The chosen R sits between: as accurate as ×0.1 and the only one that is consistent.

## Limitations

- The camera simulation has no lens distortion, blur or latency, so real-world R would need re-measuring on hardware.
- A single marker means the pose is only observable thanks to the marker-yaw measurement, which degrades beyond about 1.5–2 m.
- NEES peaks around blind spins; a separate rotational slip term in Q could fix it.
