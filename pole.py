"""
pole.py - Improvement 2: the gravity model near the surface (why LS7 / the pole misbehaves).

The paper's gravity is a 2nd-order spherical-harmonic series (gravity.py). Such a series is only
reliable OUTSIDE the sphere that encloses the body (radius la = 1000 m, the Brillouin sphere).
All landing sites are on or inside it, and at the pole (r = 250 m) the factor (la/r)^2 = 16 makes
the C20 term dominate, so the model's gravity even points OUTWARD.

Here we use the EXACT gravity of a uniform triaxial ellipsoid (valid everywhere outside and on the
surface) as the "true" asteroid, and compare two guidance models:
    paper    : SMC-IO uses the 2nd-order model internally         (what the paper does)
    improved : SMC-IO uses the exact ellipsoid model internally    (our improvement)

Exact field of a uniform ellipsoid (Kellogg / MacMillan), for a point outside or on the surface:
    g_i = -2 pi G rho a b c * x_i * Int_lambda^inf ds / ((a_i^2 + s) * sqrt((a^2+s)(b^2+s)(c^2+s)))
where lambda >= 0 solves  x^2/(a^2+lambda) + y^2/(b^2+lambda) + z^2/(c^2+lambda) = 1  (lambda = 0
on the surface). The integral equals (2/3) * R_D(.,.,.), Carlson's elliptic integral (scipy).

Usage:
    python pole.py            # gravity comparison + LS1..LS7 with both guidance models
    python pole.py LS7        # one case only
"""
import sys
import os
import numpy as np
from scipy.optimize import brentq
from scipy.special import elliprd

import params as P
import smcio
from gravity import grad_U
from dynamics import clip_thrust

os.makedirs("results", exist_ok=True)
os.makedirs("figs", exist_ok=True)

A2 = np.array([P.la, P.lb, P.lc]) ** 2
K_ELL = 2.0 * np.pi * P.G * P.rho * P.la * P.lb * P.lc


# ---------------------------------------------------------------- exact ellipsoid gravity
def _lambda(r):
    """Ellipsoidal coordinate lambda >= 0 of point r (0 on or inside the surface)."""
    f = lambda L: np.sum(r**2 / (A2 + L)) - 1.0
    if f(0.0) <= 0.0:
        return 0.0
    hi = max(np.dot(r, r), 1.0)
    while f(hi) > 0:
        hi *= 2.0
    return brentq(f, 0.0, hi, xtol=1e-10, rtol=1e-12)


def grad_exact(r):
    """Gravity of a uniform ellipsoid at r (outside or on the surface), m/s^2."""
    L = _lambda(r)
    p, q, s = A2 + L
    I = (2.0 / 3.0) * np.array([elliprd(q, s, p), elliprd(p, s, q), elliprd(p, q, s)])
    return -K_ELL * r * I


def grad_point(r):
    """Point-mass gravity (for reference)."""
    return -P.GM * r / np.linalg.norm(r) ** 3


def rotation_terms(r, v):
    w = P.omega
    return -2.0 * np.cross(w, v) - np.cross(w, np.cross(w, r))


MODELS = {"2nd-order (paper)": grad_U, "exact ellipsoid": grad_exact, "point mass": grad_point}


# ---------------------------------------------------------------- 1. how wrong is the paper's model?
def compare_fields():
    print("Gravity at each landing site: radial component (negative = inward) and magnitude, m/s^2")
    print("site   r (m)    2nd-order             exact ellipsoid       point mass")
    for name, cs in P.CASES.items():
        r = cs["rfd"]
        u = r / np.linalg.norm(r)
        row = f"{name}  {np.linalg.norm(r):6.0f}  "
        for f in MODELS.values():
            g = f(r)
            row += f"{g @ u:+.2e} ({np.linalg.norm(g):.2e})  "
        print(row)

    # far field: the series and the exact field must agree
    r = np.array([5000.0, 1000.0, 800.0])
    e = np.linalg.norm(grad_U(r) - grad_exact(r)) / np.linalg.norm(grad_exact(r))
    print(f"\ncheck: far away (r = 5.2 km) 2nd-order vs exact differ by {e*100:.3f} %  (should be small)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, (axis, a_i, lab) in enumerate(((0, P.la, "equator, along x"), (2, P.lc, "pole, along z"))):
        d = np.linspace(a_i, 1600.0, 300)
        for (mname, f), ls in zip(MODELS.items(), ("-", "--", ":")):
            gr = []
            for x in d:
                r = np.zeros(3); r[axis] = x
                gr.append(f(r)[axis])                 # radial component (inward is negative)
            ax[k].plot(d, gr, ls, label=mname)
        ax[k].axhline(0, color="k", lw=.6)
        ax[k].axvline(P.la, color="gray", lw=.6, ls=":")
        ax[k].set_xlabel("distance from centre (m)")
        ax[k].set_ylabel("radial gravity (m/s$^2$), <0 = inward")
        ax[k].set_title(f"From the surface outward: {lab}")
        ax[k].grid(alpha=.3)
        ax[k].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figs/pole_gravity.png", dpi=150)
    plt.close(fig)
    print("figure: figs/pole_gravity.png")


# ---------------------------------------------------------------- 2. landings with the exact asteroid
def simulate(case, guidance_model):
    """SMC-IO landing. TRUTH = exact ellipsoid gravity; guidance uses `guidance_model`."""
    smcio.g_total = lambda r, v: rotation_terms(r, v) + guidance_model(r)   # what the guidance believes
    cs = dict(P.CASES[case])
    r, v, m, t, h = cs["r0"].copy(), cs["v0"].copy(), P.M0, 0.0, P.H
    effort, fallbacks, Rmin = 0.0, 0, np.inf
    while True:
        R = np.linalg.norm(cs["rfd"] - r)
        Rmin = min(Rmin, R)
        if R <= P.R_stop or t > P.t_max or R > Rmin + 2.0:
            break
        a, _, st = smcio.smcio_command(r, v, m, cs)
        fallbacks += int(st == 1)
        a = clip_thrust(a, m)
        g_true = rotation_terms(r, v) + grad_exact(r)
        effort += float(a @ a) * h
        r, v = r + v * h, v + (g_true + a) * h
        m -= m * np.linalg.norm(a) / P.c * h
        t += h
    return dict(err=float(np.linalg.norm(r - cs["rfd"])), vf=float(np.linalg.norm(v)),
                dm=P.M0 - m, tf=t, effort=effort, fallbacks=fallbacks)


def run_cases(cases):
    print("\nTRUE asteroid = exact ellipsoid gravity")
    print("case  guidance model       tf (s)   fuel (kg)  pos err (m)  |vf| (m/s)  effort    fallbacks")
    rows = {}
    for case in cases:
        for label, model in (("paper (2nd-order)", grad_U), ("improved (exact)", grad_exact)):
            o = simulate(case, model)
            rows[(case, label)] = o
            print(f"{case}  {label:18s} {o['tf']:8.2f}  {o['dm']:8.2f}  {o['err']:10.3f}  "
                  f"{o['vf']:9.4f}  {o['effort']:.4f}  {o['fallbacks']:6d}")
    np.save("results/pole_results.npy", rows, allow_pickle=True)


if __name__ == "__main__":
    cases = [sys.argv[1]] if len(sys.argv) > 1 else list(P.CASES)
    compare_fields()
    run_cases(cases)