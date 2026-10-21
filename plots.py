"""Figures and tables from results/*.npz  (run run.py first).

    python plots.py            # LS1: Fig.3-style panel, SMC-IO vs ZEM/ZEV-G comparison, tables
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from smcio import envelope, envelope_bounds

os.makedirs("figs", exist_ok=True)
PAPER = {  # Table 3 (LS1..LS7): dm SMC-IO, dm ZEM/ZEV-G, int|a|^2 SMC-IO, int|a|^2 ZEM/ZEV-G
    "LS1": (2.19, 3.59, 0.0371, 0.0677), "LS2": (1.85, 4.94, 0.0222, 0.1549),
    "LS3": (2.54, 5.36, 0.0557, 0.1505), "LS4": (2.08, 5.68, 0.0319, 0.2020),
    "LS5": (2.12, 5.29, 0.0347, 0.1808), "LS6": (3.28, 5.24, 0.1048, 0.1651),
    "LS7": (4.12, 3.84, 0.1114, 0.1095)}
LBL = ["x", "y", "z"]; COL = ["tab:green", "tab:blue", "tab:red"]; LS = [":", "--", "-"]


def load(case, law):
    f = f"results/{case}_{law}.npz"
    return dict(np.load(f)) if os.path.exists(f) else None


def thrust(d):  return d["m"] * np.linalg.norm(d["a"], axis=1)
def effort_curve(d):  return np.cumsum(np.sum(d["a"] ** 2, axis=1)) * P.H


def lines(ax, t, Y, ylabel):
    for i in range(3):
        ax.plot(t, Y[:, i], LS[i], color=COL[i], label=LBL[i])
    ax.set_ylabel(ylabel); ax.set_xlabel("Time (s)"); ax.grid(alpha=.3); ax.legend()


def fig_smcio(case="LS1"):
    d = load(case, "smcio"); c = P.CASES[case]; t = d["t"]
    fig, ax = plt.subplots(4, 2, figsize=(11, 13))
    lines(ax[0, 0], t, d["r"], "Position (m)"); ax[0, 0].set_title("a) Position")
    lines(ax[0, 1], t, d["v"], "Velocity (m/s)"); ax[0, 1].set_title("b) Velocity")
    lines(ax[1, 0], t, d["a"], "Acceleration (m/s$^2$)"); ax[1, 0].set_title("c) Acceleration applied")
    ax[1, 1].plot(t, thrust(d)); ax[1, 1].set_title("d) Thrust"); ax[1, 1].set_ylabel("T (N)")
    ax[2, 0].plot(t, d["m"], label=f"$\\Delta m$ = {P.M0 - d['m'][-1]:.2f} kg"); ax[2, 0].legend(); ax[2, 0].set_title("e) Mass"); ax[2, 0].set_ylabel("m (kg)")
    ax[2, 1].plot(t, effort_curve(d), label=f"{d['effort_a2']:.4f}"); ax[2, 1].legend(); ax[2, 1].set_title("f) Control effort  $\\int|a|^2dt$")
    ax[3, 0].plot(t, d["e"]); ax[3, 0].set_title("g) Sliding variable $s_e = e$"); ax[3, 0].set_ylabel("rad")
    VM = np.array([envelope(r + v * P.H, c["r0"], c["v0"], c["rfd"]) for r, v in zip(d["r"], d["v"])])
    VL = np.zeros_like(VM); VU = np.zeros_like(VM)
    for k in range(len(VM)):
        VL[k], VU[k] = envelope_bounds(VM[k], c["r0"], c["v0"], c["rfd"])
    for i in range(3):
        ax[3, 1].plot(t, VL[:, i], LS[i], color=COL[i], label=f"$V_{LBL[i]}$"); ax[3, 1].plot(t, VU[:, i], LS[i], color=COL[i])
    ax[3, 1].set_title("h) Velocity envelope"); ax[3, 1].legend()
    for a in ax.flat[2:]: a.set_xlabel("Time (s)"); a.grid(alpha=.3)
    fig.suptitle(f"SMC-IO guidance, {case} (our simulation)"); fig.tight_layout()
    fig.savefig(f"figs/{case}_smcio_panel.png", dpi=150); plt.close(fig)


def fig_compare(case="LS1"):
    s, z = load(case, "smcio"), load(case, "zemzev")
    if z is None: return
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for d, name, st in ((s, "SMC-IO", "-"), (z, "ZEM/ZEV-G", "--")):
        t = d["t"]
        ax[0, 0].plot(t, np.linalg.norm(d["a"], axis=1), st, label=name)
        ax[0, 1].plot(t, thrust(d), st, label=name)
        ax[1, 0].plot(t, d["m"], st, label=f"{name}: $\\Delta m$={P.M0 - d['m'][-1]:.2f} kg")
        ax[1, 1].plot(t, effort_curve(d), st, label=f"{name}: {d['effort_a2']:.4f}")
    for a, yl in zip(ax.flat, ["|a| (m/s$^2$)", "Thrust (N)", "Mass (kg)", "$\\int|a|^2dt$"]):
        a.set_ylabel(yl); a.set_xlabel("Time (s)"); a.grid(alpha=.3); a.legend()
    fig.suptitle(f"{case}: SMC-IO vs ZEM/ZEV-G (same final time)"); fig.tight_layout()
    fig.savefig(f"figs/{case}_compare.png", dpi=150); plt.close(fig)


def tables():
    print("\nLS   law        tf(s)   dm(kg)  paper dm   int|a|^2dt  paper    pos err(m)  |vf|(m/s)")
    for cs in P.CASES:
        for law, k in (("smcio", 0), ("zemzev", 1)):
            d = load(cs, law)
            if d is None: continue
            err = np.linalg.norm(d["r"][-1] - P.CASES[cs]["rfd"])
            print(f"{cs}  {law:7s} {d['tf']:8.2f}  {P.M0 - d['m'][-1]:6.2f}  {PAPER[cs][k]:8.2f}  "
                  f"{d['effort_a2']:10.4f}  {PAPER[cs][2 + k]:7.4f}  {err:9.3f}  {np.linalg.norm(d['v'][-1]):9.4f}")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "LS1"
    fig_smcio(case); fig_compare(case); tables()
    print("figures saved in figs/")