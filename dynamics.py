"""
dynamics.py - equations of motion in the asteroid-fixed rotating frame, paper Eqs. (1)-(3).

    r_dot = v
    v_dot = g(r, v) + a
    g     = -2 w x v  -  w x (w x r)  +  grad U(r)        (Euler term is 0: w is constant)
    m_dot = -m |a| / c

Used by both guidance schemes (SMC-IO and ZEM/ZEV-G); only the way `a` is chosen differs.
"""
import numpy as np
from params import omega, c, Tmax, h
from gravity import grad_U

def g_total(r, v):
    """Everything except thrust: Coriolis + centrifugal + gravity, m/s^2."""
    coriolis = -2.0 * np.cross(omega, v)
    centrifugal = -np.cross(omega, np.cross(omega, r))
    return coriolis + centrifugal + grad_U(r)


def clip_thrust(a, m, Tm=None):
    """Scale a down if m|a| > Tmax (thrust limit, Eq. 3)."""
    Tm = Tmax if Tm is None else Tm
    amax = Tm / m
    n = np.linalg.norm(a)
    return a if n <= amax else a * (amax / n)


def step(r, v, m, a, dt=None):
    """One explicit-Euler step (as in the paper, dt = 0.05 s). Returns r, v, m."""
    dt = h if dt is None else dt
    g = g_total(r, v)
    r_new = r + v * dt
    v_new = v + (g + a) * dt
    m_new = m - m * np.linalg.norm(a) / c * dt
    return r_new, v_new, m_new


euler_step = step      # name used by run.py


# ---------------------------------------------------------------- self-test:  python dynamics.py
if __name__ == "__main__":
    from params import CASES, GM, m0
    from gravity import potential

    ok = True

    # 1. spin terms: for r on the x-axis, v = 0, centrifugal = +w^2 r along +x (outward)
    w = omega[2]
    r = np.array([1000.0, 0.0, 0.0])
    cen = -np.cross(omega, np.cross(omega, r))
    print(f"1. centrifugal at 1 km = {cen[0]:.3e} (expect +w^2 r = {w**2*1000:.3e}), "
          f"vs gravity {abs(grad_U(r)[0]):.3e}  -> same order, as the report says")
    ok &= np.isclose(cen[0], w**2 * 1000.0)

    # Coriolis for v along +x: -2 w x v = -2 w * (z x x) = -2 w (+y)  -> negative y
    cor = -2.0 * np.cross(omega, np.array([1.0, 0.0, 0.0]))
    print(f"   Coriolis for v=(1,0,0) = {cor}   (expect (0, {-2*w:.3e}, 0))")
    ok &= np.isclose(cor[1], -2 * w) and np.isclose(cor[0], 0)

    # 2. Jacobi constant is conserved with a = 0 (catches any sign error in the rotation terms)
    #    C = 1/2 |v|^2 - U(r) - 1/2 |w x r|^2
    def jacobi(r, v):
        return 0.5 * v @ v - potential(r) - 0.5 * np.dot(np.cross(omega, r), np.cross(omega, r))

    def rk4(r, v, dt):
        f = lambda r, v: (v, g_total(r, v))
        k1r, k1v = f(r, v)
        k2r, k2v = f(r + dt/2*k1r, v + dt/2*k1v)
        k3r, k3v = f(r + dt/2*k2r, v + dt/2*k2v)
        k4r, k4v = f(r + dt*k3r, v + dt*k3v)
        return (r + dt/6*(k1r + 2*k2r + 2*k3r + k4r),
                v + dt/6*(k1v + 2*k2v + 2*k3v + k4v))

    cs = CASES["LS1"]
    r, v = cs["r0"].copy(), cs["v0"].copy()
    J0 = jacobi(r, v)
    for _ in range(int(300 / 0.5)):       # 300 s coasting, no thrust
        r, v = rk4(r, v, 0.5)
    drift = abs(jacobi(r, v) - J0)
    print(f"2. Jacobi constant drift over 300 s coast (RK4) = {drift:.2e}  (J0 = {J0:.3e})")
    ok &= drift < 1e-9

    # 3. mass flow: constant |a| for time T burns m0 (1 - exp(-aT/c))  (rocket equation)
    a = np.array([0.02, 0.0, 0.0])        # 0.02 m/s^2 -> 28 N at 1400 kg
    r, v, m = cs["r0"].copy(), cs["v0"].copy(), m0
    T = 100.0
    for _ in range(int(T / h)):
        r, v, m = step(r, v, m, a)
    exact = m0 * (1 - np.exp(-0.02 * T / c))
    print(f"3. fuel burned: {m0-m:.5f} kg vs rocket equation {exact:.5f} kg")
    ok &= abs((m0 - m) - exact) < 5e-3

    # 4. thrust limiter
    a_big = np.array([1.0, 1.0, 1.0])
    a_c = clip_thrust(a_big, m0)
    print(f"4. clipped thrust = {m0*np.linalg.norm(a_c):.3f} N (limit {Tmax}); direction kept: "
          f"{np.allclose(a_c/np.linalg.norm(a_c), a_big/np.linalg.norm(a_big))}")
    ok &= np.isclose(m0 * np.linalg.norm(a_c), Tmax)
    ok &= np.allclose(clip_thrust(np.array([0.01, 0, 0]), m0), [0.01, 0, 0])   # small a untouched

    # 5. Euler step (h=0.05) tracks RK4 over 60 s of coasting from LS1
    r1, v1, m1 = cs["r0"].copy(), cs["v0"].copy(), m0
    r2, v2 = cs["r0"].copy(), cs["v0"].copy()
    for _ in range(int(60 / h)):
        r1, v1, m1 = step(r1, v1, m1, np.zeros(3))
    for _ in range(int(60 / 0.5)):
        r2, v2 = rk4(r2, v2, 0.5)
    err = np.linalg.norm(r1 - r2)
    print(f"5. Euler(0.05 s) vs RK4 position after 60 s: {err:.3e} m   (expect << 1 m)")
    ok &= err < 0.5

    print("\nALL PASSED" if ok else "\nSOMETHING FAILED")