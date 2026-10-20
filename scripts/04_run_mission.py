"""Step 4 - Run one closed-loop mission (navigate + dock) and save its figure / video.

Examples
  python scripts/04_run_mission.py                          # nominal, headless, figure only
  python scripts/04_run_mission.py --scenario push --seed 3
  python scripts/04_run_mission.py --record demo.mp4        # composite video for the demo
  mjpython scripts/04_run_mission.py --viewer               # macOS live viewer (use python on Linux/Win)

Outputs go to results/runs/<scenario>_seed<k>.png (+ .mp4 if --record).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from tb3nav import metrics
from tb3nav.mission import RobotOptions, run_mission
from tb3nav.plots import run_overview
from tb3nav.scenarios import SCENARIOS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="nominal", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", action="store_true", help="live MuJoCo viewer with overlays")
    ap.add_argument("--record", metavar="MP4", help="write a composite demo video")
    ap.add_argument("--no-yaw", action="store_true", help="range-bearing only (no marker yaw)")
    ap.add_argument("--t-max", type=float, default=240)
    args = ap.parse_args()

    spec, dist = SCENARIOS[args.scenario](args.seed)
    opts = RobotOptions(use_yaw=False if args.no_yaw else None)
    out = ROOT / "results" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.scenario}_seed{args.seed}"

    rec = None
    kw = {}
    if args.record:
        from tb3nav.viz import Recorder
        rec = Recorder(args.record)
        rec.title = f"Track A | {args.scenario} | seed {args.seed}"
        kw.update(frame_cb=rec, fast_camera=False)
    if args.viewer:
        import mujoco.viewer
        from tb3nav.viz import viewer_callback_factory
        holder = {}

        def view_cb(sim, robot, pkt):
            if "v" not in holder:
                holder["v"] = mujoco.viewer.launch_passive(sim.model, sim.data)
                v = holder["v"]
                v.cam.lookat[:] = [3, 2, 0]; v.cam.distance = 6.5; v.cam.elevation = -60
                holder["cb"] = viewer_callback_factory(v)
            holder["cb"](sim, robot, pkt)
        kw["viewer"] = view_cb
    try:
        res = run_mission(spec, dist, seed=args.seed, opts=opts, t_max=args.t_max, **kw)
    finally:
        if rec:
            rec.close()
    m = metrics.summarise(res)
    for e in res.events:
        print(f"{e[0]:7.2f}  {e[1]}")
    print(json.dumps(m, indent=2))
    run_overview(res, out / f"{name}.png", title=f"{args.scenario}, seed {args.seed}: "
                 f"{'DOCKED' if res.success else 'FAILED'}  (dock error {100*m['dock_pos_err_m']:.1f} cm, "
                 f"{m['dock_yaw_err_deg']:.1f} deg)")
    print("figure:", out / f"{name}.png")


if __name__ == "__main__":
    main()
