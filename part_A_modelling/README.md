# Part A: Modelling

> **Marking guide, Part A (20 marks):** "correctness of the state-space model, observability and controllability analysis".
> Brief: "Required: state-space model of the platform."

This part covers the robot platform, its state-space model, the assumptions behind it, and the controllability and observability analysis that drives later design choices.

## Steps, in order

| Step | Script | What it shows | Time |
|---|---|---|---|
| A1 | `turtlebot_sim.py` | Drive the robot yourself: the platform, its sensors and the room | n/a |
| A2 | `open_loop_test.py --headless` | Open-loop wheel-speed sequence (forward, turn, stop) runs cleanly | 5 s |
| A3 | `official_model_checks.py` | The official robot model slips badly; three fixes, applied one at a time | 20 s |
| A4 | `model_analysis.py` | Controllability and observability ranks, actuator step response, odometry drift | 15 s |

Run from the project root, for example `python part_A_modelling/model_analysis.py`. On macOS, run viewer scripts (A1, A2 without `--headless`) with `mjpython`.

Model code: [`src/model.py`](../src/model.py).

---

## A1. The platform

We use the **official ROBOTIS TurtleBot3 Waffle Pi model**: it is the TurtleBot3 variant with a camera on the real hardware, which the brief requires. We added the sensors it was missing (camera, LiDAR rays, encoders), all marked `TRACK A` in [`turtlebot3/turtlebot3_waffle_pi_track_a.xml`](../turtlebot3/turtlebot3_waffle_pi_track_a.xml). The untouched original is kept alongside.

| | |
|---|---|
| Body | Official meshes, masses and inertias (1.86 kg) |
| Drive | Two wheels, radius r = 0.033 m, separation b = 0.288 m; velocity servos limited to ±7.88 rad/s (0.26 m/s). Two rear ball casters |
| Encoders | Wheel angle and speed (added) |
| LiDAR | 72 rays at 5° from the LDS-01 turret, 3.5 m range (added) |
| Camera | `front_camera` at the Pi camera mount, 0.086 m ahead of the axle and 0.107 m up (added) |
| Room | 4 m × 4 m, two obstacles, six 15 cm wall tags and an ArUco-marked charging station with three tags ([Part B](../part_B_sensing/)) |
| Physics | Timestep 0.002 s |

**Keyboard driving (A1):** W/S forward and reverse, A/D turn on the spot, Space stop, R reset, C toggle the robot's camera view, L show LiDAR rays.

| File | Origin |
|---|---|
| `turtlebot_sim.py` | Codex-generated starter, adapted to the Waffle Pi |
| `open_loop_test.py` | The group repo's `simulate_turtlebot.py` (loop adapted from the UR3 tutorial) |

## A3. Fixing the official model so it moves like the real robot

Driving the official model, the wheels turned 11 % further than the robot moved, and 24 % further when turning on the spot. A real TurtleBot3 doesn't do this, and odometry built on it would be useless.

| Model | Straight: true ÷ odometry | Turn: true ÷ odometry | Wheel speed ÷ command | Chassis carries |
|---|---|---|---|---|
| Official | 0.888 | 0.763 | 49 % | 35 % of weight |
| + chassis collision box raised 1 cm | 0.890 | 0.765 | 49 % | 0 % |
| + casters frictionless (contact priority) | 0.999 | 0.998 | 75 % | 0 % |
| + wheel servo gain kv 0.1 → 1.0 (**used**) | 0.997 | 1.000 | 97 % | 0 % |

1. **The chassis dragged on the floor.** The official collision mesh reaches the floor and carried 35 % of the weight. It was replaced with a box over the same footprint, 1 cm up.
2. **The casters weren't frictionless.** MuJoCo combines two touching surfaces using the *higher* friction of the pair, so the "frictionless" casters took the floor's friction and dragged in every turn. `priority="1"` makes their own setting win.
3. **The wheels were too slow.** With kv = 0.1 against 0.1 N·m of joint friction, wheels reached only 49–75 % of the command. The real XL430 servo runs its own velocity loop; kv = 1.0 gives 97 %.

## The state-space model

State **x** = [x, y, θ], the pose of `base_link` (on the floor under the wheel axle) in the map frame. Input **u** = [v, ω].

```
ẋ = v cos θ          v = r (ω_R + ω_L) / 2
ẏ = v sin θ          ω = r (ω_R − ω_L) / b
θ̇ = ω
```

| Assumption | Justified by |
|---|---|
| Flat floor, rigid body | The simulated room |
| No sideways wheel slip (non-holonomic) | A3: after the fixes, true motion matches the wheels within 0.3 % |
| Motor dynamics negligible (kinematic model) | A4: wheel time constant 14 ms against a 20 ms control period. Close, so stated as a limitation |
| Nominal r and b | A3/A4: odometry drift 0.35 % |

- **Discretisation:** 50 Hz, midpoint rule (exact for straight motion). [`src/model.py`](../src/model.py) also has the Jacobians F = ∂f/∂x and G = ∂f/∂u used by the EKF.
- **Linearised tracking error** (used by the LQR in [Part D](../part_D_control/)). Error **e** is the reference pose relative to the robot, in the robot frame, linearised about (v_r, ω_r) (Kanayama et al. 1990):

```
ė = A e + B u_fb,   A = [[0, ω_r, 0], [−ω_r, 0, v_r], [0, 0, 0]],   B = −[[1,0],[0,0],[0,1]]
```

## A4. Controllability and observability

| Question | Rank | Design consequence |
|---|---|---|
| Tracking error, reference moving (v_r or ω_r ≠ 0) | **3 / 3** | LQR can correct any error |
| Tracking error, reference at rest | 2 / 3 | A wheeled robot can't fix a sideways error standing still. So the robot turns on the spot before tracking, and docking uses a moving reference |
| Odometry only | 0 / 3 | Absolute pose unobservable: it drifts forever |
| Range + bearing to one marker, even moving | **2 / 3** | Rotating the whole path about the marker gives identical readings |
| Range + bearing + **tag yaw** | **3 / 3** (condition 2.8) | The EKF uses this pose measurement ([Part C](../part_C_estimation/)) |
| **Two tags 10 cm apart**, range + bearing only, stationary | **3 / 3** (condition 40) | Why the dock has two small tags: they make the pose observable without the tag's yaw. The high condition number means one direction (sideways offset paired with heading) is only weakly constrained, which mattered when tuning R (Part C) |

The rank-2 result for range + bearing to one tag came back as a real failure in [Part E](../part_E_integration_and_robustness/). Range + bearing updates from far away, where the yaw isn't usable, made the estimate worse, so the EKF limits how far away each tag is fused (scaled by tag size).

## Results (`results/`)

| File | What it shows |
|---|---|
| `actuator_step.png` | Wheels reach 63 % of a speed step in **14 ms**, steady-state gain 0.98 |
| `odometry_drift.png` | Dead reckoning over a 7.3 m figure-eight drifts **2.5 cm (0.35 %)**, 0.1° heading. Encoder noise barely matters; slip dominates. Small, but unbounded, which is why the camera is needed |
| `summary.json` | All numbers above |
