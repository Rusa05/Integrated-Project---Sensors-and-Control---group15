# Step-by-step guide — Track A (TurtleBot3 visual waypoint navigation and docking)

This guide walks the whole project from a fresh clone to the submission. Every step lists the
command, what it does, what to look at in the output, and which marking criterion (A–E) it
serves. Steps 2 and 3 write `results/calibration.json`, which every later step reads, so run
the steps **in order** the first time.

| Step | Script | Criterion | Typical time (laptop) |
|---|---|---|---|
| 0 | `scripts/00_frame_diagram.py` | B (frame management) | seconds |
| 1 | `scripts/01_model_analysis.py` | A (model, controllability, observability) | ~1 min |
| 2 | `scripts/02_calibrate_odometry.py` | B (calibration) | ~1 min |
| 3 | `scripts/03_calibrate_camera.py` | B (calibration, noise handling) | 1–5 min |
| 4 | `scripts/04_run_mission.py` | C, D (one mission, figure, video) | ~1 min (real time with `--viewer`) |
| 5 | `scripts/05_monte_carlo.py` | E (robustness, repeated trials) | 5–60 min (see note) |
| 6 | `scripts/06_tradeoffs.py` | C, D, Distinction/HD analysis | 5–40 min |

Timings: on a Mac or a PC with a GPU, camera rendering takes a few milliseconds, so each mission
runs faster than real time. In a headless Linux container with software rendering (OSMesa)
each frame takes ~0.15 s, and a mission takes ~90 s of wall time. Steps 5 and 6 run missions
in parallel on all CPU cores and cache every result in `results/batch/`, so they can be
interrupted and resumed.

---

## Step −1: set up the environment

```bash
git clone https://github.com/Rusa05/Integrated-Project---Sensors-and-Control---group15.git
cd Integrated-Project---Sensors-and-Control---group15
git checkout Claudia
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

* **macOS**: the live viewer needs `mjpython` (installed with MuJoCo):
  `mjpython scripts/04_run_mission.py --viewer`. All other scripts use plain `python`.
* **Headless Linux** (lab server, CI, WSL without a display): set an offscreen GL backend before
  running anything that renders the camera: `export MUJOCO_GL=egl` (GPU) or
  `export MUJOCO_GL=osmesa` (CPU; needs `sudo apt install libosmesa6`).
* The original milestone script `simulate_turtlebot.py` still runs unchanged; it uses the
  menagerie model directly (open-loop actuator test).

---

## Step 0: frames (Part B, frame management)

```bash
python scripts/00_frame_diagram.py        # -> results/figures/frames.png
```

**What it is.** Every coordinate frame in the pipeline and every transform between them, drawn
from the actual numbers in `tb3nav/frames.py` and `models/turtlebot3_waffle_pi_sensors.xml`:

* `W` world: origin at the room's south-west inner corner, x east, y north, z up.
* `B` base_link: midpoint of the wheel axle on the floor, x forward. **This is the state the
  EKF estimates and the LQR controls.**
* `C` camera optical frame (OpenCV convention: z forward, x right, y down); `R_BC` maps it
  into `B`. It is 85.5 mm ahead of the axle and 107 mm up.
* `L` LiDAR at (−49, 0, 172) mm, axes parallel to `B`.
* `M_i` marker frames: x along the wall, y up, z the outward normal.

**The chain for one camera measurement** (`tb3nav/sensing.py` → `frames.marker_in_base`):
pixel corners → `solvePnPGeneric(IPPE_SQUARE)` gives `T_CM` → `T_BM = T_BC · T_CM` →
range = ‖(x, y) of the tag in `B`‖, bearing = atan2(y, x), relative yaw = yaw of `z_M` in `B`.
The EKF compares this with `h(x)` computed from the map position of `M_i` in `W`.

**Say in the viva:** why the measurement is planar (tags and camera are at different heights, so
only the x–y projection is used), and why the camera frame is rotated relative to `B`.

---

## Step 1: platform model, controllability, observability (Part A)

```bash
python scripts/01_model_analysis.py   # prints a report; results/model_analysis.json, figures/model_analysis.png
```

**The model** (derivation in `tb3nav/model.py`):

* Full 5-state model `X = [x, y, θ, ω_L, ω_R]`, input = wheel-speed set-points. Kinematics of a
  differential drive plus a first-order motor loop `ω̇ = (ω_cmd − ω)/τ`.
* τ is identified from wheel step responses (≈ 7 ms). Because τ is far below the 50 ms control
  period, the controller and estimator use the reduced **unicycle** model `s = [x, y, θ]`,
  `u = [v, ω]`. This is a stated, checked assumption, not a guess.
* Platform-specific parameters (`r_eff(ω)`, `g(ω)`, `x_ICR`) come from Step 2, so the model is
  derived for the actual robot rather than the datasheet.
* Tracking-error model for the LQR: `e = [e_x, e_y, e_θ]` in the robot frame, linearised to
  `A(v_r, ω_r) = [[0, ω_r, 0], [−ω_r, 0, v_r], [0, 0, 0]]`, `B = [[1, 0], [0, 0], [0, 1]]`.

**What the output shows:**

1. **Controllability.** The error model has rank 3 whenever `v_r ≠ 0` or `ω_r ≠ 0`, and rank 2 at
   `v_r = ω_r = 0`. That is why pure rotations use a separate heading controller
   (`controller.heading_controller`) and the LQR schedule clamps `v_r ≥ 1 cm/s`. The unicycle
   linearised at standstill has rank 2. That is Brockett's condition: no smooth static
   feedback can stabilise a point, so we track a *trajectory* instead.
2. **Observability.** Odometry alone: rank 0, because absolute pose is unobservable and dead
   reckoning drifts. One tag with range + bearing: **rank 2 even while driving**, because rotating
   the robot's whole pose about the tag changes neither range nor bearing. Range + bearing + tag
   yaw, or two tags: rank 3.
3. **Design consequence.** The final docking approach sees only the dock, and there the tag yaw
   is unusable (head-on views, see Step 3). The dock therefore carries **two** small tags 10 cm
   apart (IDs 10 and 11), which restores full observability. Without them the heading
   estimate drifted confidently during docking (NEES ≈ 14).

---

## Step 2: odometry calibration (Part B)

```bash
python scripts/02_calibrate_odometry.py   # updates results/calibration.json, figures/odometry_calibration.png
```

The simulated robot does **not** match the datasheet (r = 33 mm, b = 287 mm), just as a real one
would not. The script finds, from steady-state runs against an external reference (the
simulator's true pose plays the role of a tape measure or overhead tracker, used **offline
only**):

| Effect | Finding | Model term |
|---|---|---|
| Rolling radius | 28.6 mm at cruise, 32 mm slow or reversing (slip against the casters) | lookup table `r_eff(ω_mean)` |
| Turning gain | 27 % below r/b when turning on the spot, ~18 % below when rolling | table `g(ω_mean)` |
| Centre of rotation | turns about a point **27.7 mm behind the axle** | `x_ICR`; lateral motion `−x_ICR·Δθ` |
| Random error | k_s ≈ 0.02 m^½ per wheel, plus rotation noise | EKF process noise Q |

**Validation (UMBmark-style 1 m square on odometry only):** the end-point error drops from about
110–125 cm with datasheet values to about 10 cm with the calibrated model. Modelling the ICR alone
cuts the position error per in-place turn from 2.95 cm to 0.96 cm.

The step also documents one model change (`models/turtlebot3_waffle_pi_sensors.xml`): the
menagerie actuator gain `kv = 0.1` left a ~2 rad/s dead band against the 0.1 N·m joint
friction, unlike a real Dynamixel in velocity mode. `kv` was raised to 1.0, and a firmware-like
integral term in `sim.py` removes the residual offset.

---

## Step 3: camera calibration and measurement-noise model (Part B)

```bash
python scripts/03_calibrate_camera.py           # full run (renders ~5000 images)
python scripts/03_calibrate_camera.py --refit   # re-fit saved samples only (seconds)
```

**Part 1, intrinsics.** A 9×6 checkerboard is shown to the robot camera in 60 random poses. Views
with mis-localised corners are rejected iteratively (> 3× the median error). The lens model is
chosen on **held-out** reprojection error. The simplest model wins (no distortion, 0.059 px),
which is correct for MuJoCo's ideal pinhole. The calibrated K matches the analytic K from
`fovy` to 0.2 px. Lesson worth telling: without a generic initial guess and outlier rejection,
`calibrateCamera` converged to a wrong K (f off by 4 %, cx off by 17 px), which produced a
−0.03 rad bearing bias.

**Part 2, ArUco noise.** The robot is placed at 15 ranges × 3 bearings × 5 viewing angles from a
tag, with 24 noisy frames each. Findings, all used in the EKF:

* **Range bias** grows ≈ quadratically (≈ 4 cm at 3 m and 9 cm at 4.4 m for a 15 cm tag). It is fitted as
  `c1·r + c2·r²` and subtracted online.
* **Bearing** is very precise (σ < 0.001 rad). Range σ ≈ 0.7 mm + 2.2 % of range.
* **Relative yaw** is 3–4× worse **head-on** than at oblique views. Planar PnP is weakest
  frontally, where the two IPPE solutions nearly coincide. Yaw R is therefore a function of
  the viewing angle, and yaw is only used while its σ < 0.08 rad.
* **IPPE flips** (the two-fold planar ambiguity) occur in 5–30 % of frames beyond 2.5 m. Both
  solutions are kept and the EKF picks the one consistent with its prior.
* **Different tag sizes** (15 cm wall tags, 4 cm dock tags) use the same model through pinhole
  similarity: a tag of size s at range r looks like the reference tag at range r·s_ref/s.

---

## Step 4: run one mission (Parts C and D, and the demo video)

```bash
python scripts/04_run_mission.py                              # nominal; figure in results/runs/
python scripts/04_run_mission.py --scenario push --seed 3     # a disturbed trial
python scripts/04_run_mission.py --record results/runs/demo.mp4   # composite video
mjpython scripts/04_run_mission.py --viewer                   # live 3D view (macOS)
```

The mission state machine (`tb3nav/mission.py`) runs as follows:

`LOCALISE` (spin until the EKF initialises from two-tag triangulation) → `PLAN` (A* on the
LiDAR occupancy grid to a pre-dock pose 0.7 m in front of the dock) → `ALIGN` (turn to the
path) → `TRACK` (LQR on the smoothed, time-parameterised path) → `DOCK_ALIGN` → `DOCK` (5 cm/s
straight approach, still LQR) → `DONE`.

Replanning happens when new LiDAR obstacles block the path, on a LiDAR safety stop, or when the
tracking error exceeds 35 cm. The estimator's divergence detector re-localises after 6
consecutive gated updates, for example after a push.

**Read the figure** (`results/runs/<scenario>_seed<k>.png`): map with true, estimated and
reference paths plus 3σ ellipses; estimate − truth with the ±3σ band for x, y and θ (the "estimate
and its uncertainty alongside the true state" the brief asks for); true versus believed
cross-track error; NIS of every camera update, with gated ones marked.

**The ground-truth rule.** `sim.Simulator.ground_truth_for_evaluation()` is the only accessor of
the true pose. It is called in `mission.run_mission()` for logging, in the video recorder for
the on-screen "true" line, and in the offline calibration scripts. The `Robot` class (EKF,
planner, controller) receives only `SensorPacket`s. Check with
`grep -n ground_truth tb3nav/*.py`.

---

## Step 5: Monte Carlo robustness (Part E)

```bash
python scripts/05_monte_carlo.py --workers 4     # -> results/monte_carlo.md, figures/mc_*.png
python scripts/05_monte_carlo.py --report        # rebuild tables and figures from the cache
```

There are 43 trials over 7 scenarios: nominal, random layouts, a 25 N push, a slippery floor
(μ = 0.25 against the 1.0 used for calibration), encoder-scale error, a degraded camera
(×3 pixel noise, 40 % dropped frames) and everything combined. The outputs are the success rate,
docking error, estimation RMSE, NEES consistency, true tracking error, collisions,
re-localisations and push recovery time. Failures are listed by name, so you can re-run any
one with `04_run_mission.py --scenario … --seed …` and diagnose it.

---

## Step 6: trade-offs and ablations (Distinction / HD analysis)

```bash
python scripts/06_tradeoffs.py --workers 4       # -> results/tradeoffs.md, figures/tradeoff_*.png
```

* **A.** EKF process-noise scale: consistency (NEES) against responsiveness (push recovery time).
* **B.** EKF measurement-noise scale: trusting the camera against consistency.
* **C.** LQR control weight: tracking error against control effort (a Pareto front).
* **D.** Measurement model ablation: range + bearing + yaw against range + bearing only against
  odometry only.

---

## Making the 5-minute demo video

The brief requires the system completing the task, at least one varied or disturbed trial, and
the estimate with its uncertainty alongside the true state. A suggested timeline:

| Time | Content | Source |
|---|---|---|
| 0:00–0:30 | Problem, room, robot, sensors, frame diagram | `figures/frames.png`, `--viewer` |
| 0:30–1:45 | Nominal mission: localise, plan, track, dock | `--record demo_nominal.mp4` |
| 1:45–2:45 | Disturbed trial: push, divergence detected, re-localise, still docks | `--scenario push --seed 0 --record …` |
| 2:45–3:30 | Calibration highlights (odometry square, marker noise, dock observability) | `figures/odometry_calibration.png`, `marker_noise_vs_range.png`, `model_analysis.png` |
| 3:30–4:30 | Monte Carlo table, consistency plot, trade-offs | `results/monte_carlo.md`, `figures/mc_*.png`, `tradeoff_*.png` |
| 4:30–5:00 | Limits and what you would do next | README "Limitations" |

The recorded video shows a tracking overview with the green estimate, 3σ ellipse and heading
fan, the blue reference and yellow LiDAR points placed at the *estimated* pose. Next to it are
the robot camera with detected tags and a live text panel of true pose, estimate, 3σ and error.
On a Mac, also screen-record the `--viewer` window for a smoother 3D view.

---

## Splitting the work and the individual viva (Part F, 25 %)

Each member must explain their own part and the whole pipeline. A natural split for four people:

| Member | Owns | Must be able to explain |
|---|---|---|
| 1 | Model and control: `model.py`, `controller.py`, Step 1, Study C | error-model linearisation, why the LQR fails at v_r = 0, Bryson's rule, gain scheduling, time-scaling of the reference |
| 2 | Odometry and LiDAR: `kinematics.py`, `planner.py`, Step 2 | speed-dependent radius, ICR offset, UMBmark result, occupancy log-odds, inflation, A*, replanning |
| 3 | Camera: `sensing.py`, `frames.py`, Step 3 | intrinsic calibration, IPPE ambiguity, range-bias correction, view-dependent yaw noise, frame chain |
| 4 | Estimation and integration: `ekf.py`, `mission.py`, Steps 5–6 | Jacobians F, G, H; Q and R choices; χ² gating; NEES and NIS; divergence detection; Monte Carlo results |

For a 3-person team, merge members 2 and 4. Record the real split in the README's contribution
table. **Everyone should be ready for:** "Where exactly does the controller get its pose?" (the
EKF state, `mission.Robot._track`) and "What happens if you remove the camera?" (Study D).

Likely viva questions to rehearse:

1. Why is the system unobservable with odometry only, and with a single tag?
2. Your NEES is below 3 in some states. Is that good? (The filter is conservative; Study A
   shows what tighter Q costs in push recovery.)
3. Why use range/bearing rather than the tag's full pose? (Bearing is ~100× more precise, and
   yaw is ambiguous head-on.)
4. Why did you change `kv` in the robot model? (Dead band; justified by Dynamixel velocity mode.)
5. What would change on the real TurtleBot3? (See below.)

---

## Optional: real hardware (Week 12 induction, HD evidence)

The `Robot` class only needs a `SensorPacket` (encoder ticks, LiDAR scan, camera image) and
returns wheel-speed commands. On a real TurtleBot3 (ROS 2), you would:

1. Subscribe to `/joint_states` (wheel positions, the encoder ticks equivalent), `/scan`
   (LDS-01) and `/image_raw` (Pi camera); build a `SensorPacket` every 50 ms.
2. Publish `(v, ω)` on `/cmd_vel` directly from `Robot` (skip `DiffDrive.inverse`, because the
   OpenCR board does the wheel conversion).
3. Repeat Steps 2 and 3 on the real robot: checkerboard calibration with the real lens (expect
   distortion to be selected this time), odometry with a tape measure, and marker noise
   with printed tags at measured positions.
4. Compare sim against hardware quantitatively on the same metrics: docking error, estimation
   NIS, tracking error and success rate over at least 5 runs.

---

## Before submitting: checklist

* [ ] README: group number, members, contribution statement, video link filled in.
* [ ] Fresh clone → `pip install -r requirements.txt` → Steps 0–4 run as described.
* [ ] Results tables in the README match `results/monte_carlo.md` and `results/tradeoffs.md`.
* [ ] Generative-AI declaration updated (ChatGPT for setup; Claude for this pipeline and these docs).
* [ ] Video ≤ 5 min, includes a disturbed trial and estimate ± uncertainty against truth.
