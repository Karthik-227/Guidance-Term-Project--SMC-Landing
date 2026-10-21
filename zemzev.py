"""ZEM/ZEV-G feedback guidance (Hawkins, Guo & Wie 2012) with a waypoint, as used in the paper.

a = 6/tgo^2 * ZEM - 2/tgo * ZEV
ZEM = r_target - r_free(tgo),   ZEV = v_target - v_free(tgo)
where (r_free, v_free) is the UNFORCED (a = 0) trajectory propagated through the full rotating-frame
dynamics with the variable gravity field, so Coriolis, centrifugal and spherical-harmonic gravity
are all included ("-G").
"""
import numpy as np
import params as P
from dynamics import g_total

N_PROP = 8          # RK4 sub-steps used to propagate the free trajectory
TGO_MIN = 1.0       # s: in the last TGO_MIN seconds of each phase the previous command is held,
                    #    because 1/tgo^2 would amplify tiny integration mismatches into spikes


def _deriv(r, v):
    return v, g_total(r, v)


def free_propagate(r, v, tgo, n=N_PROP):
    """RK4 propagation of the unforced dynamics over tgo."""
    dt = tgo / n
    for _ in range(n):
        k1r, k1v = _deriv(r, v)
        k2r, k2v = _deriv(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v)
        k3r, k3v = _deriv(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v)
        k4r, k4v = _deriv(r + dt * k3r, v + dt * k3v)
        r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r, v


def zemzev_command(t, r, v, m, ctx, tf):
    """ctx = dict(rfd, rwp, vwp, twp).  Phase 1 aims at the waypoint, phase 2 at (rfd, 0)."""
    if t < ctx["twp"]:
        r_t, v_t, tgo = ctx["rwp"], ctx["vwp"], ctx["twp"] - t
    else:
        r_t, v_t, tgo = ctx["rfd"], np.zeros(3), tf - t
    if tgo < TGO_MIN and "_last" in ctx:
        return ctx["_last"]                      # hold (the law is nearly linear in time here)
    tgo = max(tgo, TGO_MIN)
    r_f, v_f = free_propagate(r, v, tgo)
    zem = r_t - r_f
    zev = v_t - v_f
    a = 6.0 / tgo ** 2 * zem - 2.0 / tgo * zev
    amax = P.TMAX / m
    nrm = np.linalg.norm(a)
    if nrm > amax:
        a = a * (amax / nrm)
    ctx["_last"] = a
    return a    