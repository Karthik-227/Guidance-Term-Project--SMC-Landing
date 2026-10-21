"""SMC-IO guidance (Shincy & Ghosh 2023): the command a(t) at one guidance instant.

Each call solves   min |a|^2   s.t.  alpha . a = alpha0        (sliding constraint, Eq. 25)
                                      a_lo <= a <= a_hi         (velocity envelope, Eq. 33)
                                      |a| <= Tmax/m             (thrust limit, Eq. 34)
"""
import numpy as np
import params as P
from dynamics import g_total


# ---------------------------------------------------------------- heading error
def heading_error(r, v, rfd):
    """e = angle between LOS R = rfd - r and velocity v, Eq. (11)."""
    R = rfd - r
    c = (R @ v) / (np.linalg.norm(R) * np.linalg.norm(v))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def alpha_terms(r, v, g, rfd):
    """alpha (3-vector) and alpha4 such that  edot = alpha . a + alpha4   (Eqs. 12-13)."""
    R = rfd - r
    Rn = np.linalg.norm(R)
    vn = max(np.linalg.norm(v), 1e-9)
    N = R @ v                                   # R . v
    D = Rn * vn
    sin_e = max(np.sqrt(max(1.0 - (N / D) ** 2, 0.0)), 1e-9)   # guards e -> 0
    K = -1.0 / (D ** 2 * sin_e)
    alpha = K * (D * R - N * Rn * v / vn)       # alpha_1..alpha_3
    alpha4 = K * (D * (R @ g) - N * Rn * (v @ g) / vn - Rn * vn ** 3 + N ** 2 * vn / Rn)   # Rdot = -v
    return alpha, alpha4


# ---------------------------------------------------------------- velocity envelope
def envelope(r_next, r0, v0, rfd):
    """V_iM at the predicted position, Eq. (26), per axis."""
    zeta = np.maximum((r_next - rfd) / (r0 - rfd), 0.0)         # clip: no overshoot past the site
    return np.abs(v0) * np.log10(1.0 + P.B * zeta + P.CC * zeta ** 2)


def envelope_bounds(VM, r0, v0, rfd):
    """Lower / upper velocity limits per axis, Eq. (32)."""
    s_v = np.sign(v0)
    s_r = np.sign(rfd - r0)
    agree = (s_v == s_r) & (s_v != 0)           # axis already moving towards the site
    Vl = np.where(agree, np.where(s_v > 0, 0.8 * VM, -VM), -VM)
    Vu = np.where(agree, np.where(s_v > 0, VM, -0.8 * VM), VM)
    return Vl, Vu


# ---------------------------------------------------------------- the small optimisation
def min_norm_plane_box(n, beta, lo, hi):
    """Minimise |a|^2 s.t. n.a = beta, lo <= a <= hi  (|n| = 1).

    KKT gives a = clip(lam * n, lo, hi); f(lam) = n . a is piecewise linear and
    non-decreasing in lam, so we find lam exactly from the breakpoints.  None if infeasible.
    """
    nz = np.abs(n) > 1e-15
    bp = np.sort(np.concatenate([lo[nz] / n[nz], hi[nz] / n[nz]]))
    fv = np.array([n @ np.clip(l * n, lo, hi) for l in bp])
    if beta < fv[0] - 1e-12 or beta > fv[-1] + 1e-12:
        return None                             # plane does not cut the box
    lam = np.interp(beta, fv, bp)
    return np.clip(lam * n, lo, hi)


# ---------------------------------------------------------------- one guidance step
def smcio_command(r, v, m, ctx, h=P.H):
    """Return (a, e, status).  ctx = dict(r0, v0, rfd).
    status: 0 = all constraints met, 1 = velocity box dropped (infeasible), 2 = thrust saturated."""
    r0, v0, rfd = ctx["r0"], ctx["v0"], ctx["rfd"]
    g = g_total(r, v)

    # (i) sliding constraint  alpha . a = alpha0
    e = heading_error(r, v, rfd)
    alpha, alpha4 = alpha_terms(r, v, g, rfd)
    alpha0 = -alpha4 - P.EPS_E * np.sign(e) - P.K_E * e
    an = np.linalg.norm(alpha)
    n, beta = alpha / an, alpha0 / an           # normalise: well-conditioned even when e is tiny

    # (ii) velocity envelope at the predicted next position -> box on a
    VM = envelope(r + v * h, r0, v0, rfd)
    Vl, Vu = envelope_bounds(VM, r0, v0, rfd)
    a_lo = (Vl - v) / h - g
    a_hi = (Vu - v) / h - g

    # (iii) minimum-norm a on the plane, inside the box
    status = 0
    a = min_norm_plane_box(n, beta, a_lo, a_hi)
    if a is None:
        a = beta * n                            # fall back: keep sliding constraint only
        status = 1

    # (iv) thrust limit.  The min-norm point is the SMALLEST |a| that satisfies the other
    # constraints, so if it exceeds Tmax/m the problem is infeasible -> saturate.
    amax = P.TMAX / m
    nrm = np.linalg.norm(a)
    if nrm > amax:
        a = a * (amax / nrm)
        status = max(status, 2)
    return a, e, status


if __name__ == "__main__":
    # self-test: finite-difference check of edot = alpha.a + alpha4
    rng = np.random.default_rng(0)
    c = P.CASES["LS1"]; r, v = c["r0"], c["v0"]; a = np.array([0.01, -0.02, 0.005]); dt = 1e-4
    g = g_total(r, v)
    alpha, alpha4 = alpha_terms(r, v, g, c["rfd"])
    r2, v2 = r + v * dt, v + (g + a) * dt
    num = (heading_error(r2, v2, c["rfd"]) - heading_error(r, v, c["rfd"])) / dt
    print("edot analytic:", alpha @ a + alpha4, " numeric:", num)
    a0, e0, st = smcio_command(r, v, P.M0, c)
    print("first command:", a0, " e0 =", e0, " status", st, " (paper: e0 = 0.765)")