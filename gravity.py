"""
gravity.py - asteroid gravity: the 2nd-order spherical-harmonic potential, paper Eq. (5).

    U(r) = GM/r * [ 1 + (la/r)^2 * ( 1/2*C20*(3 sin^2(phi) - 1) + 3*C22*cos^2(phi)*cos(2*lam) ) ]

In Cartesian form (sin(phi) = z/r,  cos^2(phi)*cos(2*lam) = (x^2 - y^2)/r^2) this is

    U = GM/r + GM*la^2 * [ C20/2 * (3 z^2 / r^5 - 1/r^3) + 3*C22 * (x^2 - y^2) / r^5 ]

so the gravitational acceleration is  grad U  (it points toward the asteroid).
Used by dynamics.py (both guidance schemes share it).
"""
import numpy as np
from params import GM, la, C20, C22


def potential(r):
    """U(r), m^2/s^2, for position r = (x, y, z) in the asteroid-fixed frame."""
    x, y, z = r
    rn = np.linalg.norm(r)
    return GM / rn + GM * la**2 * (
        0.5 * C20 * (3 * z**2 / rn**5 - 1 / rn**3) + 3 * C22 * (x**2 - y**2) / rn**5
    )


def grad_U(r):
    """Gravitational acceleration = gradient of U, m/s^2 (analytic)."""
    x, y, z = r
    rn = np.linalg.norm(r)
    r2, r3, r5, r7 = rn**2, rn**3, rn**5, rn**7
    k = GM * la**2

    # monopole: GM/r
    g = -GM * r / r3

    # C20 term: (C20/2) * (3 z^2 r^-5 - r^-3)
    g = g + k * 0.5 * C20 * (-15 * z**2 / r7 * r + 3 / r5 * r + np.array([0.0, 0.0, 6 * z / r5]))

    # C22 term: 3*C22 * (x^2 - y^2) r^-5
    q = x**2 - y**2
    g = g + k * 3 * C22 * np.array([
        2 * x / r5 - 5 * q * x / r7,
        -2 * y / r5 - 5 * q * y / r7,
        -5 * q * z / r7,
    ])
    return g


def grad_U_numeric(r, d=1e-3):
    """Central-difference gradient of potential(); only used to check grad_U."""
    g = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = d
        g[i] = (potential(r + e) - potential(r - e)) / (2 * d)
    return g


# ---------------------------------------------------------------- self-test:  python gravity.py
if __name__ == "__main__":
    from params import CASES

    ok = True

    # 1. analytic gradient vs finite differences, at every start and landing point
    worst = 0.0
    for name, cs in CASES.items():
        for key in ("r0", "rfd"):
            r = cs[key]
            a, n = grad_U(r), grad_U_numeric(r)
            rel = np.linalg.norm(a - n) / np.linalg.norm(a)
            worst = max(worst, rel)
    print(f"1. analytic vs numeric gradient, worst relative error = {worst:.2e}")
    ok &= worst < 1e-6

    # 2. size: ~1e-4 m/s^2 at ~1 km (the paper's 'weak gravity')
    r = np.array([1000.0, 0.0, 0.0])
    gmag = np.linalg.norm(grad_U(r))
    print(f"2. |g| at (1000,0,0) = {gmag:.3e} m/s^2   (expect order 1e-4)")
    ok &= 5e-5 < gmag < 5e-4

    # 3. far away it becomes a point mass: GM/r^2, pointing inward
    r = np.array([1e5, 2e4, -3e4])
    g = grad_U(r)
    ref = -GM * r / np.linalg.norm(r)**3
    err = np.linalg.norm(g - ref) / np.linalg.norm(ref)
    print(f"3. far field vs point mass, relative diff = {err:.2e}   (expect ~1e-4 or less)")
    ok &= err < 1e-3

    # 4. gravity pulls inward at LS1-LS6.  LS7 (pole, r = 250 m << la = 1000 m) is a known
    #    quirk of the 2nd-order expansion: (la/r)^2 = 16 there, so the C20 term dominates and
    #    the field points outward.  This is what Eq. (5) gives, so we just report it.
    inward = {n: bool(grad_U(cs["rfd"]) @ cs["rfd"] < 0) for n, cs in CASES.items()}
    print(f"4. gravity points inward at LS1-LS6: {all(v for n, v in inward.items() if n != 'LS7')}")
    print(f"   LS7 (pole, model artefact) points inward: {inward['LS7']}   (expected False)")
    ok &= all(v for n, v in inward.items() if n != "LS7")

    print("\nALL PASSED" if ok else "\nSOMETHING FAILED")