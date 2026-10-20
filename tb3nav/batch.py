"""Parallel batch runner for Monte Carlo trials and parameter sweeps."""
from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _job(job):
    """job = dict(name, scenario, seed, opts, cfg_overrides, out_dir). Runs in a worker."""
    from . import metrics
    from .config import load_config
    from .mission import RobotOptions, run_mission
    from .scenarios import SCENARIOS
    try:
        spec, dist = SCENARIOS[job["scenario"]](job["seed"])
        cfg = load_config()
        for path, val in job.get("cfg", {}).items():
            obj = cfg
            *parents, leaf = path.split(".")
            for p in parents:
                obj = getattr(obj, p)
            setattr(obj, leaf, val)
        opts = RobotOptions(**job.get("opts", {}))
        res = run_mission(spec, dist, seed=job["seed"], cfg=cfg, opts=opts, t_max=job.get("t_max", 240))
        m = metrics.summarise(res)
        m.update(metrics.disturbance_metrics(res, dist))
        m.update({k: job[k] for k in ("name", "scenario", "seed")})
        m["opts"], m["cfg"] = job.get("opts", {}), job.get("cfg", {})
        out = Path(job["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        L = res.log
        st = {s: i for i, s in enumerate(sorted(set(L["state"])))}
        np.savez_compressed(out / f"{job['name']}.npz", t=L["t"].astype(np.float32),
                            est=L["est"].astype(np.float32), true=L["true"].astype(np.float32),
                            P=L["P"].astype(np.float32), ref=L["ref"].astype(np.float32),
                            cmd=L["cmd"].astype(np.float32),
                            state=np.array([st[s] for s in L["state"]], np.int8),
                            state_names=np.array(sorted(st, key=st.get)))
        (out / f"{job['name']}.json").write_text(json.dumps(m, indent=1, default=float))
        return m
    except Exception:
        return {"name": job["name"], "error": traceback.format_exc()}


def run_jobs(jobs, workers=None, skip_existing=True, verbose=True):
    workers = workers or max(1, (os.cpu_count() or 2))
    results, todo = [], []
    for j in jobs:
        f = Path(j["out_dir"]) / f"{j['name']}.json"
        if skip_existing and f.exists():
            results.append(json.loads(f.read_text()))
        else:
            todo.append(j)
    if verbose:
        print(f"{len(results)} cached, {len(todo)} to run on {workers} workers")
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_job, j): j for j in todo}
        for k, f in enumerate(as_completed(futs)):
            m = f.result()
            results.append(m)
            if verbose:
                if "error" in m:
                    print(f"[{k+1}/{len(todo)}] {m['name']} ERROR\n{m['error']}")
                else:
                    print(f"[{k+1}/{len(todo)}] {m['name']}: success={m['success']} "
                          f"dock={100*m['dock_pos_err_m']:.1f}cm est_rmse={100*m['est_rmse_pos_m']:.1f}cm "
                          f"NEES={m['mean_nees']:.2f} t={m['mission_time_s']:.0f}s", flush=True)
    return results
