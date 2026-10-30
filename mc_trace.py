"""mc_trace.py - one Monte Carlo landing with everything logged (same logic as montecarlo.one_run).

    python3 mc_trace.py E5 1000 smcio 0   -> results/trace_E5_1000_smcio.npz
    python3 mc_trace.py E5 1000 smcio 1   -> results/trace_E5_1000_smcio_kf.npz
"""
import sys
import numpy as np
import params as P
import smcio
from dynamics import g_total, clip_thrust
from gravity import grad_U
import Montecarlo as MC

exp, seed, law, use_filter = sys.argv[1], int(sys.argv[2]), sys.argv[3], bool(int(sys.argv[4]))
on = MC.EXPERIMENTS[exp]
rng = np.random.default_rng(seed)
d_gm = rng.uniform(-MC.GM_ERR, MC.GM_ERR)
t_bias = rng.normal(0.0, MC.SIG_T_BIAS)
dr0 = rng.uniform(-MC.DR0, MC.DR0, 3)
dv0 = rng.uniform(-MC.DV0, MC.DV0, 3)
step_rng = np.random.default_rng(seed + 10_000)

cs = P.CASES[MC.CASE]
r = cs["r0"] + (dr0 if on.get("init") else 0.0)
v = cs["v0"] + (dv0 if on.get("init") else 0.0)
m, t, h = P.M0, 0.0, P.H
gm_scale = 1.0 + (d_gm if on.get("grav") else 0.0)
ctx = dict(cs); ctx.pop("_last", None)
ctx["r0"], ctx["v0"] = r.copy(), v.copy()

nav = MC.NavFilter(r, v) if (use_filter and on.get("nav")) else None
a_cmd = np.zeros(3)
L = {k: [] for k in ("t", "r", "v", "r_raw", "v_raw", "r_seen", "v_seen", "a_cmd", "a",
                     "e_true", "e_seen", "st", "R", "lo_seen", "hi_seen", "lo_true", "hi_true")}
Rmin, n = np.inf, 0


def box(rr, vv):
    g = g_total(rr, vv)
    VM = smcio.envelope(rr + vv * h, ctx["r0"], ctx["v0"], ctx["rfd"])
    Vl, Vu = smcio.envelope_bounds(VM, ctx["r0"], ctx["v0"], ctx["rfd"])
    return (Vl - vv) / h - g, (Vu - vv) / h - g


while True:
    if on.get("nav"):
        r_raw = r + step_rng.normal(0.0, MC.SIG_R, 3)
        v_raw = v + step_rng.normal(0.0, MC.SIG_V, 3)
    else:
        r_raw, v_raw = r.copy(), v.copy()
    r_m, v_m = r_raw, v_raw
    if nav is not None:
        if n > 0:
            nav.predict(a_cmd)
        r_m, v_m = nav.update(r_raw, v_raw)

    Rtrue = np.linalg.norm(cs["rfd"] - r)
    Rmin = min(Rmin, Rtrue)
    if Rtrue < MC.R_LAND or t > MC.T_LIMIT or Rtrue > Rmin + MC.DIVERGE:
        break

    a_cmd, e_seen, st = smcio.smcio_command(r_m, v_m, m, ctx)
    a = a_cmd
    if on.get("thrust"):
        a = a * (1.0 + t_bias + step_rng.normal(0.0, MC.SIG_T_NOISE))
        a = MC.tilt(a, abs(step_rng.normal(0.0, MC.SIG_POINT)), step_rng)
    a = clip_thrust(a, m)

    if n % 10 == 0:   # log every 0.5 s
        lo_s, hi_s = box(r_m, v_m); lo_t, hi_t = box(r, v)
        for k, val in (("t", t), ("r", r), ("v", v), ("r_raw", r_raw), ("v_raw", v_raw),
                       ("r_seen", r_m), ("v_seen", v_m), ("a_cmd", a_cmd), ("a", a),
                       ("e_true", smcio.heading_error(r, v, cs["rfd"])), ("e_seen", e_seen),
                       ("st", st), ("R", Rtrue), ("lo_seen", lo_s), ("hi_seen", hi_s),
                       ("lo_true", lo_t), ("hi_true", hi_t)):
            L[k].append(np.array(val, dtype=float).copy())

    g_true = g_total(r, v) + (gm_scale - 1.0) * grad_U(r)
    r = r + v * h
    v = v + (g_true + a) * h
    m = m - m * np.linalg.norm(a) / P.c * h
    t += h
    n += 1

out = {k: np.array(vv) for k, vv in L.items()}
tag = law + ("_kf" if use_filter else "")
np.savez(f"results/trace_{exp}_{seed}_{tag}.npz", d_gm=d_gm, t_bias=t_bias, dr0=dr0, dv0=dv0,
         rfd=cs["rfd"], tf=t, dm=P.M0 - m, err=np.linalg.norm(r - cs["rfd"]), vf=np.linalg.norm(v), **out)
print(f"{exp} seed {seed} {tag}: tf={t:.2f}s  err={np.linalg.norm(r-cs['rfd']):.3f} m  "
      f"vf={np.linalg.norm(v):.4f}  dm={P.M0-m:.3f} kg  d_gm={d_gm:+.4f} t_bias={t_bias:+.4f} "
      f"dr0={np.round(dr0,3)} dv0={np.round(dv0,4)}")