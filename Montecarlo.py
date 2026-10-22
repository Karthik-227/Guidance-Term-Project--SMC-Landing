"""
montecarlo.py - robustness of SMC-IO (and ZEM/ZEV-G) to real-world errors.

The paper assumes perfect knowledge. Here the simulation is split into
    TRUTH     : the real asteroid and thruster  (where the errors live)
    GUIDANCE  : still uses the nominal model, and only sees NOISY measurements

Experiments (100 runs each by default):
    E1  navigation noise only        r, v measured with noise at every step
    E2  gravity-model error only     true GM = (1 + d) * nominal GM,  d ~ U(-20%, +20%)
    E3  thrust errors only           magnitude bias + per-step pointing jitter
    E4  initial-state dispersion     r0 +- 5 m, v0 +- 0.05 m/s (per axis)
    E5  ALL errors together          the realistic headline case

Usage:
    python montecarlo.py                       # E1..E5, 100 runs each, SMC-IO only
    python montecarlo.py --exp E5 --law both   # E5 with SMC-IO and ZEM/ZEV-G (same random draws)
    python montecarlo.py --runs 10             # quick test
    python montecarlo.py --workers 4           # use 4 CPU cores (default: all cores)
    python montecarlo.py --exp E1 --filter     # same, but guidance sees a Kalman-filtered state
    python montecarlo.py --report              # only print tables + make figures from saved results

Results: results/mc_<exp>_<law>.npz      Figures: figs/mc_*.png
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

import params as P
import smcio
import zemzev
from dynamics import g_total, clip_thrust
from gravity import grad_U

os.makedirs("results", exist_ok=True)
os.makedirs("figs", exist_ok=True)

CASE = "LS1"
TF_ZEM = 624.20            # ZEM/ZEV-G final time = our nominal SMC-IO landing time for LS1

# ---------------------------------------------------------------- error sizes
SIG_R = 1.0                # navigation noise, position (m, 1-sigma, per axis, every step)
SIG_V = 0.01               # navigation noise, velocity (m/s, 1-sigma, per axis, every step)
GM_ERR = 0.20              # true GM within +-20 % of the model
SIG_T_BIAS = 0.02          # thrust magnitude bias (1-sigma, per run)
SIG_T_NOISE = 0.01         # thrust magnitude noise (1-sigma, per step)
SIG_POINT = np.radians(1)  # thrust pointing jitter (1-sigma, per step)
DR0 = 5.0                  # initial position dispersion (m, uniform +-, per axis)
DV0 = 0.05                 # initial velocity dispersion (m/s, uniform +-, per axis)

EXPERIMENTS = {
    "E1": dict(nav=True),
    "E2": dict(grav=True),
    "E3": dict(thrust=True),
    "E4": dict(init=True),
    "E5": dict(nav=True, grav=True, thrust=True, init=True),
}
NAMES = {"E1": "Navigation noise", "E2": "Gravity-model error", "E3": "Thrust errors",
         "E4": "Initial dispersion", "E5": "All errors"}

# ---------------------------------------------------------------- stopping rule
R_LAND = 0.5               # landed when TRUE distance to site < 0.5 m
T_LIMIT = 1.5 * 624.2      # give up after 1.5x the nominal landing time
DIVERGE = 2.0              # stop if true distance grows 2 m beyond its minimum


class NavFilter:
    """Simple Kalman filter giving the guidance a smoothed estimate of r and v.

    Model (per axis, axes treated independently):
        predict : r <- r + v h,   v <- v + (g_model(r, v) + a_cmd) h
        update  : measurements of r (sigma SIG_R) and v (sigma SIG_V)
    Process noise Q covers what the model does not know (gravity error, thrust errors).
    """
    def __init__(self, r0, v0, q_acc=1e-3):
        self.r, self.v = r0.copy(), v0.copy()
        h = P.H
        self.F = np.array([[1.0, h], [0.0, 1.0]])
        self.Q = q_acc**2 * np.array([[h**4 / 4, h**3 / 2], [h**3 / 2, h**2]])
        self.Rm = np.diag([SIG_R**2, SIG_V**2])
        self.P = np.array([np.diag([SIG_R**2, SIG_V**2]) for _ in range(3)])   # one 2x2 per axis

    def predict(self, a_cmd):
        g = g_total(self.r, self.v)
        self.r = self.r + self.v * P.H
        self.v = self.v + (g + a_cmd) * P.H
        for i in range(3):
            self.P[i] = self.F @ self.P[i] @ self.F.T + self.Q

    def update(self, r_m, v_m):
        for i in range(3):
            S = self.P[i] + self.Rm
            K = self.P[i] @ np.linalg.inv(S)
            innov = np.array([r_m[i] - self.r[i], v_m[i] - self.v[i]])
            x = np.array([self.r[i], self.v[i]]) + K @ innov
            self.r[i], self.v[i] = x
            self.P[i] = (np.eye(2) - K) @ self.P[i]
        return self.r.copy(), self.v.copy()


def tilt(a, ang, rng):
    """Rotate vector a by a small angle `ang` about a random axis perpendicular to it."""
    n = np.linalg.norm(a)
    if n < 1e-15 or ang == 0.0:
        return a
    u = a / n
    p = np.cross(u, rng.standard_normal(3))
    p /= np.linalg.norm(p)
    return n * (np.cos(ang) * u + np.sin(ang) * p)


def one_run(args):
    """One landing with random errors. args = (exp, law, seed). Returns a dict of results."""
    exp, law, seed = args[:3]
    use_filter = len(args) > 3 and args[3]
    on = EXPERIMENTS[exp]
    rng = np.random.default_rng(seed)

    # ---- per-run random draws (drawn in the same order for both laws -> identical errors)
    d_gm = rng.uniform(-GM_ERR, GM_ERR)
    t_bias = rng.normal(0.0, SIG_T_BIAS)
    dr0 = rng.uniform(-DR0, DR0, 3)
    dv0 = rng.uniform(-DV0, DV0, 3)
    step_rng = np.random.default_rng(seed + 10_000)   # per-step noise stream

    cs = P.CASES[CASE]
    r = cs["r0"] + (dr0 if on.get("init") else 0.0)    # TRUE initial state
    v = cs["v0"] + (dv0 if on.get("init") else 0.0)
    m, t, h = P.M0, 0.0, P.H
    gm_scale = 1.0 + (d_gm if on.get("grav") else 0.0)

    # guidance context: the spacecraft builds its envelope from the state it believes it starts in
    ctx = dict(cs)
    ctx.pop("_last", None)
    ctx["r0"], ctx["v0"] = r.copy(), v.copy()

    Rmin, effort1, effort2, fallbacks, n = np.inf, 0.0, 0.0, 0, 0
    nav = NavFilter(r, v) if (use_filter and on.get("nav")) else None
    a_cmd = np.zeros(3)
    a_hist = []
    while True:
        # -- what the spacecraft measures
        if on.get("nav"):
            r_m = r + step_rng.normal(0.0, SIG_R, 3)
            v_m = v + step_rng.normal(0.0, SIG_V, 3)
        else:
            r_m, v_m = r, v
        if nav is not None:                       # guidance sees the filtered estimate
            if n > 0:
                nav.predict(a_cmd)
            r_m, v_m = nav.update(r_m, v_m)

        # -- stopping rule, on the TRUE state
        Rtrue = np.linalg.norm(cs["rfd"] - r)
        Rmin = min(Rmin, Rtrue)
        if law == "smcio":
            done = Rtrue < R_LAND or t > T_LIMIT or Rtrue > Rmin + DIVERGE
        else:
            done = t >= TF_ZEM - 1e-9 or t > T_LIMIT
        if done:
            break

        # -- guidance command from the measured state (nominal model inside)
        if law == "smcio":
            a_cmd, _, st = smcio.smcio_command(r_m, v_m, m, ctx)
            fallbacks += int(st == 1)
        else:
            a_cmd = zemzev.zemzev_command(t, r_m, v_m, m, ctx, TF_ZEM)

        # -- what the engine actually delivers
        a = a_cmd
        if on.get("thrust"):
            a = a * (1.0 + t_bias + step_rng.normal(0.0, SIG_T_NOISE))
            a = tilt(a, abs(step_rng.normal(0.0, SIG_POINT)), step_rng)
        a = clip_thrust(a, m)

        # -- TRUE dynamics (gravity scaled if the model is wrong)
        g_true = g_total(r, v) + (gm_scale - 1.0) * grad_U(r)
        effort1 += np.linalg.norm(a) * h
        effort2 += float(a @ a) * h
        r = r + v * h
        v = v + (g_true + a) * h
        m = m - m * np.linalg.norm(a) / P.c * h
        t += h
        n += 1
        if n % 20 == 0:
            a_hist.append(np.linalg.norm(a))

    err = float(np.linalg.norm(r - cs["rfd"]))
    vf = float(np.linalg.norm(v))
    # chattering indicator: mean |change| of |a| between samples (higher = more jumpy thrust)
    ah = np.array(a_hist) if a_hist else np.zeros(2)
    chatter = float(np.mean(np.abs(np.diff(ah)))) if len(ah) > 1 else 0.0
    return dict(err=err, vf=vf, dm=P.M0 - m, tf=t, effort=effort2, fallbacks=fallbacks,
                chatter=chatter, d_gm=d_gm, t_bias=t_bias)


def run_batch(exp, law, runs, workers, use_filter=False, seed0=1000):
    seeds = [seed0 + i for i in range(runs)]
    t0 = time.time()
    with Pool(workers) as pool:
        out = []
        for i, res in enumerate(pool.imap(one_run, [(exp, law, s, use_filter) for s in seeds]), 1):
            out.append(res)
            if i % max(1, runs // 10) == 0 or i == runs:
                print(f"   {exp} {law:6s} {i:4d}/{runs}   ({time.time() - t0:6.0f} s)", flush=True)
    keys = out[0].keys()
    data = {k: np.array([o[k] for o in out]) for k in keys}
    tag = law + ("_kf" if use_filter else "")
    np.savez(f"results/mc_{exp}_{tag}.npz", seeds=np.array(seeds), **data)
    return data


# ---------------------------------------------------------------- reporting
def load(exp, law):
    f = f"results/mc_{exp}_{law}.npz"
    return dict(np.load(f)) if os.path.exists(f) else None


def stats_line(exp, law, d):
    ok_paper = np.mean((d["err"] < 3.0) & (d["vf"] < 3.0)) * 100
    ok_strict = np.mean((d["err"] < 1.0) & (d["vf"] < 0.1)) * 100
    return (f"{exp:3s} {NAMES[exp]:20s} {law:9s} {len(d['err']):4d}  "
            f"{np.mean(d['err']):6.2f} ± {np.std(d['err']):5.2f}  {np.max(d['err']):6.2f}  "
            f"{np.mean(d['vf']):6.3f} ± {np.std(d['vf']):5.3f}  "
            f"{np.mean(d['dm']):5.2f} ± {np.std(d['dm']):4.2f}  "
            f"{ok_paper:5.1f}%  {ok_strict:5.1f}%  {np.mean(d['fallbacks']):6.1f}")


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\nexp name                 law    runs  pos err (m)     max     |vf| (m/s)       "
          "fuel (kg)     ok<3m   ok<1m   fallbacks")
    for exp in EXPERIMENTS:
        for law in ("smcio", "smcio_kf", "zemzev", "zemzev_kf"):
            d = load(exp, law)
            if d is not None:
                print(stats_line(exp, law, d))
    print("ok<3m : error < 3 m and |vf| < 3 m/s (paper's requirement)")
    print("ok<1m : error < 1 m and |vf| < 0.1 m/s (stricter)")

    # Figure 1: SMC-IO sensitivity, E1..E5 side by side
    have = [e for e in EXPERIMENTS if load(e, "smcio") is not None]
    if have:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for k, (key, lab) in enumerate((("err", "Landing error (m)"), ("vf", "Touchdown speed (m/s)"),
                                        ("dm", "Fuel used (kg)"))):
            ax[k].boxplot([load(e, "smcio")[key] for e in have], showfliers=True)
            ax[k].set_xticks(range(1, len(have) + 1))
            ax[k].set_xticklabels([f"{e}\n{NAMES[e]}" for e in have], fontsize=7)
            ax[k].set_ylabel(lab)
            ax[k].grid(alpha=.3)
        ax[2].axhline(2.18, color="gray", ls=":", label="nominal (no errors)")
        ax[2].legend(fontsize=7)
        fig.suptitle("SMC-IO robustness, LS1: which error matters most?")
        fig.tight_layout()
        fig.savefig("figs/mc_sensitivity.png", dpi=150)
        plt.close(fig)

    # Figure 2: E5, SMC-IO vs ZEM/ZEV-G histograms
    s, z = load("E5", "smcio"), load("E5", "zemzev")
    if s is not None:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for k, (key, lab) in enumerate((("err", "Landing error (m)"), ("vf", "Touchdown speed (m/s)"),
                                        ("dm", "Fuel used (kg)"))):
            data = [s[key]] + ([z[key]] if z is not None else [])
            bins = np.linspace(min(map(np.min, data)), max(map(np.max, data)), 25)
            ax[k].hist(s[key], bins=bins, alpha=.6, label="SMC-IO")
            if z is not None:
                ax[k].hist(z[key], bins=bins, alpha=.6, label="ZEM/ZEV-G")
            ax[k].set_xlabel(lab)
            ax[k].set_ylabel("runs")
            ax[k].grid(alpha=.3)
            ax[k].legend()
        fig.suptitle("LS1, all errors together (E5)")
        fig.tight_layout()
        fig.savefig("figs/mc_E5_compare.png", dpi=150)
        plt.close(fig)
    print("figures: figs/mc_sensitivity.png, figs/mc_E5_compare.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", help="E1..E5 or all")
    ap.add_argument("--law", default="smcio", choices=["smcio", "zemzev", "both"])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--report", action="store_true", help="only report saved results")
    ap.add_argument("--filter", action="store_true", help="guidance uses a Kalman-filtered state")
    a = ap.parse_args()

    if not a.report:
        exps = list(EXPERIMENTS) if a.exp == "all" else [a.exp]
        laws = ["smcio", "zemzev"] if a.law == "both" else [a.law]
        for exp in exps:
            for law in laws:
                kf = " + Kalman filter" if a.filter else ""
                print(f"{exp} ({NAMES[exp]}), {law}{kf}, {a.runs} runs on {a.workers} cores")
                run_batch(exp, law, a.runs, a.workers, a.filter)
    report()