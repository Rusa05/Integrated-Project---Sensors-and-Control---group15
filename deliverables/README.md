# Deliverables (marking guide, section 3)

The project page is the [top-level README](../README.md); GitHub shows it on the repository's front page. The video plan is in [video_plan.md](video_plan.md).

## Checklist from the marking guide (section 3)

| Item | Status | Where / what's left |
|---|---|---|
| Project page states the track | ✅ | README header |
| Group number and team members | ⬜ **team to fill in** | README header and contributions table |
| Link to the GitHub repository | ✅ | README header |
| Model, controllability and observability | ✅ | README Part A; `part_A_modelling/` |
| Frame diagram | ✅ | `part_B_sensing/results/frame_diagram.png` (regenerate: `python part_B_sensing/frame_diagram.py`) |
| Estimator and controller design with parameter values, and why | ✅ | README Parts C and D |
| Quantitative results | ✅ | README results table, Parts C–E |
| Install and run instructions; runs from a fresh environment | ✅ | README Quick start; tested in a clean virtual environment |
| Code organised and readable | ✅ | One folder per marking part (A–E), each with a README and its own steps; shared code in `src/` |
| Ground truth used only for evaluation, visibly | ✅ | `eval/ground_truth.py` only; `grep -rn "ground_truth\|qpos" src/` returns nothing |
| Demo video, 5 min: task done, a disturbed trial, estimate + uncertainty with truth | ⬜ **team to record and edit** | Clips are ready (`part_E_integration_and_robustness/results/mission/mission_A.mp4`, `.../results/trials/combined_trial7.mp4`); timeline in `video_plan.md`; then add the link to the README |
| Individual contribution statement | ⬜ **each member to write** | README contributions table |
| Generative AI declaration | ✅ draft, ⬜ **team to review** | README; check it matches how each member used AI tools |
| Acknowledge reused code | ✅ | README Sources (ROBOTIS model, UR3 tutorial loop, libraries) |
| Viva preparation | ⬜ | Each member: explain own part + one other; questions in `docs/Track_A_What_Needs_To_Be_Done.docx` section 8 |

## Before pushing to GitHub

- The marking-guide PDF (`docs/Sensors_Control_Integrated_Project v2.pdf`) is course material, so it's in `.gitignore` and stays local.
- The two demo videos (about 7 MB each) are small enough to commit, which lets markers watch them without running anything.
- Optional, for the hardware option and High Distinction: after the Week 12 hardware induction, run the same pipeline on the real robot and add a sim-vs-hardware comparison.
