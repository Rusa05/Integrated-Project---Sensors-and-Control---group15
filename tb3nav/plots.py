"""Figures for single runs (estimate vs truth with uncertainty, map, tracking, NIS)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from scipy.stats import chi2

from . import world as W
from .frames import wrap


def init_mask(log):
    """Samples after the EKF was initialised (estimate is meaningful)."""
    est = log["est"]
    return ~np.all(est == 0, axis=1)


def est_errors(log):
    m = init_mask(log)
    e = log["est"][m] - log["true"][m]
    e[:, 2] = wrap(e[:, 2])
    return log["t"][m], e, log["P"][m]


def draw_world(ax, spec: W.WorldSpec, grid=None, res=0.05):
    if grid is not None:
        occ = 1 / (1 + np.exp(-grid))
        ax.imshow(occ.T, origin="lower", extent=(0, grid.shape[0] * res, 0, grid.shape[1] * res),
                  cmap="Greys", vmin=0, vmax=1, alpha=0.6)
    ax.add_patch(Rectangle((0, 0), W.ROOM_X, W.ROOM_Y, fill=False, lw=2, color="k"))
    for o in spec.obstacles:
        if o.kind == "box":
            hx, hy = o.size
            c = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
            R = np.array([[np.cos(o.yaw), -np.sin(o.yaw)], [np.sin(o.yaw), np.cos(o.yaw)]])
            ax.add_patch(Polygon(c @ R.T + [o.x, o.y], closed=True, fill=False, ec="tab:red", lw=1.2))
        else:
            ax.add_patch(Circle((o.x, o.y), o.size[0], fill=False, ec="tab:red", lw=1.2))
    for lm in spec.landmarks:
        ax.plot(lm.x, lm.y, "s", ms=6, color="tab:orange" if lm.id == W.DOCK_ID else "k")
        ax.annotate(str(lm.id), (lm.x, lm.y), xytext=(4 * np.cos(lm.yaw) - 2, 4 * np.sin(lm.yaw) - 2),
                    textcoords="offset points", fontsize=7)
    dp = spec.dock_pose()
    ax.plot(dp[0], dp[1], "*", ms=12, color="tab:orange")
    ax.set_xlim(-0.1, W.ROOM_X + 0.1); ax.set_ylim(-0.1, W.ROOM_Y + 0.1)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def cov_ellipse(ax, mu, P, nsig=3, **kw):
    vals, vecs = np.linalg.eigh(P[:2, :2])
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * nsig * np.sqrt(np.maximum(vals[::-1], 0))
    ax.add_patch(Ellipse(mu[:2], w, h, angle=ang, fill=False, **kw))


def run_overview(res, path, title=""):
    log, spec = res.log, res.spec
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[:2, :2])
    draw_world(ax, spec, res.grid)
    m = init_mask(log)
    ax.plot(log["true"][:, 0], log["true"][:, 1], "k-", lw=1.5, label="true")
    ax.plot(log["est"][m, 0], log["est"][m, 1], "g--", lw=1.2, label="EKF estimate")
    r = log["ref"]
    ax.plot(r[:, 0], r[:, 1], ":", color="tab:blue", lw=1.5, label="reference")
    idx = np.where(m)[0][::60]
    for i in idx:
        cov_ellipse(ax, log["est"][i], log["P"][i] * 1, nsig=3, ec="g", lw=0.6)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title or "Mission: true vs estimated path (3σ ellipses every 3 s), LiDAR map")

    t, e, P = est_errors(log)
    labels = ["x error [m]", "y error [m]", "heading error [rad]"]
    for k in range(3):
        a = fig.add_subplot(gs[k, 2])
        sd = np.sqrt(P[:, k, k])
        a.fill_between(t, -3 * sd, 3 * sd, color="g", alpha=0.2, label="±3σ (EKF)")
        a.plot(t, e[:, k], "k", lw=0.8, label="estimate − truth")
        a.set_ylabel(labels[k]); a.grid(alpha=0.3)
        lim = max(4 * np.percentile(np.abs(e[:, k]), 95), 3 * np.median(sd) * 2)
        a.set_ylim(-lim, lim)
        if k == 0:
            a.legend(fontsize=7, loc="upper right")
    a.set_xlabel("time [s]")

    a = fig.add_subplot(gs[2, 0])
    tr = ~np.isnan(log["ref"][:, 0])
    te = log["true"][tr]
    ref = log["ref"][tr]
    c, s = np.cos(te[:, 2]), np.sin(te[:, 2])
    d = ref[:, :2] - te[:, :2]
    lat = -s * d[:, 0] + c * d[:, 1]
    a.plot(log["t"][tr], 100 * lat, ".", ms=2, label="true cross-track error")
    a.plot(log["t"][tr], 100 * log["e"][tr][:, 1], ".", ms=2, label="controller's estimated error")
    a.set_ylabel("cross-track [cm]"); a.set_xlabel("time [s]"); a.legend(fontsize=7); a.grid(alpha=0.3)

    a = fig.add_subplot(gs[2, 1])
    ek = res.ekf_log
    if ek:
        tt = np.array([u.t for u in ek]); nis = np.array([u.nis for u in ek])
        dof = np.array([u.dof for u in ek]); acc = np.array([u.accepted for u in ek])
        a.semilogy(tt[acc], nis[acc], ".", ms=3, label="NIS (accepted)")
        a.semilogy(tt[~acc], nis[~acc], "rx", ms=5, label="gated out")
        for d_ in (2, 3):
            a.axhline(chi2.ppf(0.975, d_), ls="--", lw=0.7, color="gray")
        a.set_ylabel("NIS"); a.set_xlabel("time [s]"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
