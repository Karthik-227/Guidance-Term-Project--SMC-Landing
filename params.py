"""
params.py - every number from Shincy & Ghosh (JSR 2023), in SI units.
Other files do:  from params import *
"""
import numpy as np

# ---------------------------------------------------------------- asteroid (Sec. V)
G = 6.674e-11                      # gravitational constant, m^3 kg^-1 s^-2
la, lb, lc = 1000.0, 500.0, 250.0  # semi-axes, m (half-lengths)
rho = 3000.0                       # bulk density, kg/m^3  (= 3 g/cm^3)

M = 4.0 / 3.0 * np.pi * la * lb * lc * rho   # asteroid mass, kg
GM = G * M                                   # m^3/s^2

C20 = (2 * lc**2 - (la**2 + lb**2)) / (10 * la**2)   # flattening  (paper: -0.1125)
C22 = (la**2 - lb**2) / (20 * la**2)                 # elongation  (paper:  0.0375)

T_rot = 8 * 3600.0                                   # rotation period, s
omega = np.array([0.0, 0.0, 2 * np.pi / T_rot])      # spin vector, rad/s (about z)

# ---------------------------------------------------------------- spacecraft (Table 1)
m0 = 1400.0      # initial mass, kg
Tmax = 80.0      # max thrust, N   (30.0 for the low-thrust study)
c = 2206.0       # exhaust velocity, m/s

# ---------------------------------------------------------------- guidance (Table 1)
ke = 0.006       # reaching-law proportional gain   (0.00225 for the 30 N study)
eps = 1e-4       # reaching-law constant gain
b_env = np.array([9.0, 8.0, 5.0])   # velocity-profile constants b_x, b_y, b_z
c_env = np.array([0.0, 1.0, 4.0])   # velocity-profile constants c_x, c_y, c_z  (1+b+c = 10)

# ---------------------------------------------------------------- simulation
h = 0.05         # time step, s (paper: Euler, 0.05 s)
t_max = 2000.0   # safety stop, s
R_stop = 0.05    # stop when closer than this to the landing site, m

# ---------------------------------------------------------------- landing cases (Table 2)
def _v(*x):
    return np.array(x, dtype=float)

CASES = {
    "LS1": dict(r0=_v(1389.37, -60.78, -30.50), v0=_v(-2.0, -1.25, -0.5),  rfd=_v(1000.0, 0.0, 0.0),
                t_wp=450.0, r_wp=_v(1150, -75, -50), v_wp=_v(-1.0, 0.65, 0.4)),
    "LS2": dict(r0=_v(1373.48, -60.08, 60.78), v0=_v(-1.75, -1.0, -0.5),  rfd=_v(965.93, 0.0, 64.70),
                t_wp=620.0, r_wp=_v(1200, -25, 60),  v_wp=_v(-0.75, 0.75, 0.5)),
    "LS3": dict(r0=_v(1264.0, -55.29, 147.92), v0=_v(-2.0, -1.5, 1.0),    rfd=_v(866.03, 0.0, 125.0),
                t_wp=530.0, r_wp=_v(1100, -25, 135), v_wp=_v(-1.25, 0.5, -0.5)),
    "LS4": dict(r0=_v(1068.38, -46.74, 224.98), v0=_v(-2.0, -0.75, 0.75), rfd=_v(707.11, 0.0, 176.78),
                t_wp=420.0, r_wp=_v(1000, -25, 185), v_wp=_v(-1.5, 0.25, -0.25)),
    "LS5": dict(r0=_v(799.95, -34.99, 286.70), v0=_v(-2.0, -0.5, 0.75),   rfd=_v(500.0, 0.0, 216.51),
                t_wp=380.0, r_wp=_v(750, -25, 230),  v_wp=_v(-1.5, 0.25, -0.25)),
    "LS6": dict(r0=_v(477.0, -20.86, 328.89), v0=_v(-2.5, -0.6, 1.25),    rfd=_v(258.82, 0.0, 241.48),
                t_wp=350.0, r_wp=_v(400, -25, 265),  v_wp=_v(-1.5, 0.75, -0.5)),
    "LS7": dict(r0=_v(121.55, -5.32, 348.67), v0=_v(-1.0, -0.95, -1.35),  rfd=_v(0.0, 0.0, 250.0),
                t_wp=315.0, r_wp=_v(75, -65, 300),   v_wp=_v(-0.6, 0.8, -0.4)),
}

# ---------------------------------------------------------------- paper's results (Tables 2-4), for comparison
PAPER = {
    #        e0 (rad)     tf (s)        fuel used (kg)
    "LS1": dict(e0=0.765, tf=608.55, dm_smcio=2.19, dm_zem=3.59),
    "LS2": dict(e0=0.706, tf=771.45, dm_smcio=1.85, dm_zem=4.94),
    "LS3": dict(e0=0.880, tf=679.95, dm_smcio=2.54, dm_zem=5.36),
    "LS4": dict(e0=0.671, tf=574.95, dm_smcio=2.08, dm_zem=5.68),
    "LS5": dict(e0=0.678, tf=528.95, dm_smcio=2.12, dm_zem=5.29),
    "LS6": dict(e0=0.892, tf=500.65, dm_smcio=3.28, dm_zem=5.24),
    "LS7": dict(e0=0.599, tf=412.30, dm_smcio=4.12, dm_zem=3.84),
}
PAPER_30N = dict(Tmax=30.0, ke=0.00225, tf=683.6, dm_smcio=2.22, dm_zem=3.29)   # Table 4, LS1


# ---------------------------------------------------------------- self-test:  python params.py
if __name__ == "__main__":
    print(f"GM    = {GM:.2f} m^3/s^2   (expect ~104.8)")
    print(f"C20   = {C20:.4f}          (expect -0.1125)")
    print(f"C22   = {C22:.4f}           (expect  0.0375)")
    print(f"omega = {omega[2]:.3e} rad/s (expect  2.18e-04)")
    print("\nCase  site-on-surface   start e0 (ours vs paper)")
    for name, cs in CASES.items():
        s = (cs["rfd"][0] / la)**2 + (cs["rfd"][1] / lb)**2 + (cs["rfd"][2] / lc)**2
        R = cs["rfd"] - cs["r0"]
        v = cs["v0"]
        e0 = np.arccos(np.clip(R @ v / (np.linalg.norm(R) * np.linalg.norm(v)), -1, 1))
        print(f"{name}   {s:.3f}             {e0:.3f} vs {PAPER[name]['e0']:.3f}")