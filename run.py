"""Simulate one landing case with either guidance law.

    python run.py LS1            # SMC-IO, then ZEM/ZEV-G with the same final time
    python run.py LS1 smcio      # SMC-IO only
    python run.py all            # LS1..LS7, both laws

Results are saved to results/<case>_<law>.npz and read by plots.py.
"""
import sys, time, os
import numpy as np
import params as P
from dynamics import euler_step, g_total
from smcio import smcio_command
from zemzev import zemzev_command

os.makedirs("results", exist_ok=True)


def simulate(case, law, tf=None, h=P.H):
    c = dict(P.CASES[case])                      # copy: zemzev stores its held command in here
    c.pop("_last", None)
    r, v, m, t = c["r0"].copy(), c["v0"].copy(), P.M0, 0.0
    T, Rr, V, A, M, E, ST = [], [], [], [], [], [], []
    effort1 = effort2 = 0.0                      # integral |a| dt  and  integral |a|^2 dt
    t0 = time.perf_counter()
    n = 0
    while True:
        if law == "smcio":
            a, e, st = smcio_command(r, v, m, c)
            done = np.linalg.norm(c["rfd"] - r) <= P.LAND_TOL or t > P.T_SIM_MAX
        else:
            a = zemzev_command(t, r, v, m, c, tf)
            e = _heading(r, v, c["rfd"]); st = 0
            done = t >= tf - 1e-9
        T.append(t); Rr.append(r); V.append(v); A.append(a); M.append(m); E.append(e); ST.append(st)
        if done:
            break
        effort1 += np.linalg.norm(a) * h
        effort2 += float(a @ a) * h
        r, v, m = euler_step(r, v, m, a, h)
        t += h
        n += 1
    cpu = (time.perf_counter() - t0) / max(n, 1)
    out = dict(t=np.array(T), r=np.array(Rr), v=np.array(V), a=np.array(A), m=np.array(M),
               e=np.array(E), status=np.array(ST), effort_a=effort1, effort_a2=effort2,
               tf=t, cpu_per_step=cpu)
    return out


def _heading(r, v, rfd):
    R = rfd - r
    return float(np.arccos(np.clip(R @ v / (np.linalg.norm(R) * max(np.linalg.norm(v), 1e-12)), -1, 1)))


def summary(case, law, o):
    c = P.CASES[case]
    err = np.linalg.norm(o["r"][-1] - c["rfd"])
    print(f"{case} {law:6s} tf={o['tf']:8.2f} s  dm={P.M0 - o['m'][-1]:5.2f} kg  "
          f"pos err={err:6.3f} m  |vf|={np.linalg.norm(o['v'][-1]):.4f} m/s  "
          f"int|a|dt={o['effort_a']:.3f}  int|a|^2dt={o['effort_a2']:.4f}  "
          f"fallbacks={int((o['status'] > 0).sum())}  cpu/step={o['cpu_per_step']*1e3:.2f} ms")


def run_case(case, laws=("smcio", "zemzev")):
    s = simulate(case, "smcio")
    np.savez(f"results/{case}_smcio.npz", **s); summary(case, "smcio", s)
    if "zemzev" in laws:
        z = simulate(case, "zemzev", tf=s["tf"])          # same final time, as in the paper
        np.savez(f"results/{case}_zemzev.npz", **z); summary(case, "zemzev", z)


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "LS1"
    laws = ("smcio",) if (len(sys.argv) > 2 and sys.argv[2] == "smcio") else ("smcio", "zemzev")
    for cs in (P.CASES if case == "all" else [case]):
        run_case(cs, laws)