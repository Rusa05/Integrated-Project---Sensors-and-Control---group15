"""Quantitative metrics for one run (all computed after the run, from the log)."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .frames import wrap
from .plots import init_mask


def nees_series(log):
    m = init_mask(log)
    e = log["est"][m] - log["true"][m]
    e[:, 2] = wrap(e[:, 2])
    P = log["P"][m]
    return np.einsum("ni,nij,nj->n", e, np.linalg.inv(P), e)


def tracking_errors(log):
    """True cross-track / along-track / heading error w.r.t. the reference (reference frame)."""
    tr = ~np.isnan(log["ref"][:, 0])
    ref, tru = log["ref"][tr], log["true"][tr]
    c, s = np.cos(ref[:, 2]), np.sin(ref[:, 2])
    d = tru[:, :2] - ref[:, :2]
    along = c * d[:, 0] + s * d[:, 1]
    lat = -s * d[:, 0] + c * d[:, 1]
    return along, lat, wrap(tru[:, 2] - ref[:, 2]), tr


def summarise(res) -> dict:
    log = res.log
    m = init_mask(log)
    e = log["est"][m] - log["true"][m]
    e[:, 2] = wrap(e[:, 2])
    pos = np.hypot(e[:, 0], e[:, 1])
    nees = nees_series(log)
    along, lat, hd, tr = tracking_errors(log)
    cmd = log["cmd"]
    dt = np.diff(log["t"]).mean() if len(log["t"]) > 1 else 0.05
    ek = res.ekf_log
    nis = np.array([u.nis for u in ek]) if ek else np.array([np.nan])
    dof = np.array([u.dof for u in ek]) if ek else np.array([3])
    acc = np.array([u.accepted for u in ek]) if ek else np.array([True])
    nis_ok = nis < chi2.ppf(0.95, dof)
    tl = log["true"][:, :2]
    states = log["state"]
    return {
        "success": bool(res.success),
        "reached_done": states[-1] == "DONE",
        "dock_pos_err_m": float(np.hypot(*res.final_err[:2])),
        "dock_along_err_m": float(res.final_err[0]),
        "dock_lat_err_m": float(res.final_err[1]),
        "dock_yaw_err_deg": float(np.degrees(res.final_err[2])),
        "collisions": int(res.collisions),
        "mission_time_s": float(res.t_end),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(tl, axis=0), axis=1))),
        "est_rmse_pos_m": float(np.sqrt(np.mean(pos ** 2))) if m.any() else np.nan,
        "est_max_pos_m": float(pos.max()) if m.any() else np.nan,
        "est_rmse_yaw_deg": float(np.degrees(np.sqrt(np.mean(e[:, 2] ** 2)))) if m.any() else np.nan,
        "mean_nees": float(np.mean(nees)) if m.any() else np.nan,     # ideal = 3
        "nees_in_95": float(np.mean((nees > chi2.ppf(0.025, 3)) & (nees < chi2.ppf(0.975, 3)))),
        "nees_below_95": float(np.mean(nees < chi2.ppf(0.95, 3))),  # ideal = 0.95
        "mean_nis_per_dof": float(np.mean(nis[acc] / dof[acc])),   # ideal = 1
        "nis_below_95": float(np.mean(nis_ok)),
        "n_updates": int(acc.sum()), "n_gated": int((~acc).sum()),
        "track_rmse_lat_m": float(np.sqrt(np.mean(lat ** 2))) if tr.any() else np.nan,
        "track_rmse_along_m": float(np.sqrt(np.mean(along ** 2))) if tr.any() else np.nan,
        "track_max_lat_m": float(np.abs(lat).max()) if tr.any() else np.nan,
        "effort_v2": float(np.sum(cmd[:, 0] ** 2) * dt),
        "effort_w2": float(np.sum(cmd[:, 1] ** 2) * dt),
        "effort_dw": float(np.sum(np.abs(np.diff(cmd[:, 1])))),       # yaw-rate chatter
        "n_replans": sum("planned" in ev[1] for ev in res.events),
        "n_relocalise": sum("re-localising" in ev[1] for ev in res.events),
        "wall_time_s": float(res.wall_time),
    }


def disturbance_metrics(res, dist) -> dict:
    """Push response: peak estimation error after the push and time to re-converge."""
    out = {"push_peak_err_m": np.nan, "push_recovery_s": np.nan, "push_true_disp_m": np.nan}
    if dist is None or dist.push_time is None:
        return out
    log = res.log
    t = log["t"]
    m = init_mask(log)
    err = np.where(m, np.hypot(*(log["est"][:, :2] - log["true"][:, :2]).T), np.nan)
    i0 = np.searchsorted(t, dist.push_time)
    if i0 >= len(t) - 2:
        return out
    pre = err[max(0, i0 - 100):i0]
    thresh = max(0.03, 2 * np.sqrt(np.nanmean(pre ** 2))) if np.isfinite(pre).any() else 0.03
    after = err[i0:]
    j = int(np.nanargmax(after))
    out["push_peak_err_m"] = float(after[j])
    i1 = np.searchsorted(t, dist.push_time + dist.push_duration + 1.0)
    out["push_true_disp_m"] = float(np.hypot(*(log["true"][min(i1, len(t) - 1), :2] - log["true"][i0, :2])))
    below = np.where(after[j:] < thresh)[0]
    out["push_recovery_s"] = float(t[i0 + j + below[0]] - dist.push_time) if len(below) else np.nan
    return out
