# SMC-IO Guidance for Precision Soft Landing on an Asteroid

**AS5570 Term Project · Group 8 · IIT Madras**
Korlam Venkata Sai Karthik (AE24B036) · H MuraliKrishnan (AE24B041)
Instructor: Prof. Satadal Ghosh

A Python reimplementation of the paper

> V. S. Shincy and S. Ghosh, *"Sliding-Mode-Control–Based Instantaneously Optimal Guidance for Precision Soft Landing on Asteroid,"* Journal of Spacecraft and Rockets, 60(1):146–159, 2023.

A spacecraft a few hundred metres from a small, rotating, irregular asteroid must land **within 3 m** of a chosen site at **under 3 m/s**, using as little fuel as possible. SMC-IO does this by:

1. **Precision:** driving the heading error *e* (the angle between velocity and line of sight) to zero with sliding mode control.
2. **Softness:** keeping each velocity component inside an envelope that shrinks to zero at the site.
3. **Optimality:** choosing, every 0.05 s, the **smallest acceleration** that satisfies both rules and the thrust limit.

It needs no linearised dynamics and no time-to-go estimate. We compare it with the waypoint-based **ZEM/ZEV-G** guidance used in the paper.

---

## Results so far (Report 2)

SMC-IO, our simulation vs the paper (Table 3 of the paper):

| Case | Site latitude | Fuel Δm, paper (kg) | Fuel Δm, ours (kg) | ∫\|a\|²dt, paper | ∫\|a\|²dt, ours |
|---|---|---|---|---|---|
| LS1 | 0°  | 2.19 | 2.18 | 0.0371 | 0.0371 |
| LS2 | 15° | 1.85 | 1.84 | 0.0222 | 0.0222 |
| LS3 | 30° | 2.54 | 2.53 | 0.0557 | 0.0557 |
| LS4 | 45° | 2.08 | 2.08 | 0.0319 | 0.0319 |
| LS5 | 60° | 2.12 | 2.11 | 0.0347 | 0.0348 |
| LS6 | 75° | 3.28 | 3.17 | 0.1048 | 0.1066 |
| LS7 | 90° | 4.12 | 2.13 | 0.1114 | 0.0383 |

- **LS1–LS6** reproduce the paper within 3.4% in fuel, with near-identical control effort. Every landing is within 0.05 m of the site at about 1 mm/s.
- **LS7 (pole)** differs. There the second-order gravity model is used far inside its range of validity; this is under investigation.
- **LS1, SMC-IO vs ZEM/ZEV-G:** 2.18 kg vs 3.46 kg of fuel, so ZEM/ZEV-G uses 1.59× more (paper: 1.64×).
- **Speed:** about 0.2 ms per guidance step in Python, against the 50 ms guidance cycle.

---

## Repository layout

```
params.py     all constants from the paper (asteroid, spacecraft, gains, LS1–LS7)
gravity.py    2nd-order spherical-harmonic gravity, analytic gradient
dynamics.py   rotating-frame equations of motion + Euler step
smcio.py      SMC-IO guidance: heading error, sliding constraint, velocity envelope, min-norm solve
zemzev.py     ZEM/ZEV-G guidance with waypoint (comparison baseline)
run.py        simulation loop; saves results/<case>_<law>.npz
plots.py      figures (figs/) and paper-vs-ours tables
```

Every module has a self-test: run it directly (e.g. `python3 gravity.py`) to check it.

| File | Self-test checks |
|---|---|
| `params.py` | GM, C₂₀, C₂₂ match the paper; all sites lie on the surface; initial heading errors match Table 2 |
| `gravity.py` | analytic gradient = finite differences; point-mass limit far away |
| `dynamics.py` | Jacobi constant conserved while coasting; mass loss = rocket equation; thrust limiter |
| `smcio.py` | ė = α·a + α₄ matches a numerical derivative; first command at LS1 |

---

## Quick start

```bash
git clone https://github.com/Karthik-227/Guidance-Term-Project--SMC-Landing.git
cd Guidance-Term-Project--SMC-Landing
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib

python3 run.py LS1          # SMC-IO + ZEM/ZEV-G for landing scenario 1 (~30 s)
python3 run.py all          # all seven scenarios
python3 plots.py LS1        # figures -> figs/, comparison table -> terminal
```

Example output:

```
LS1 smcio  tf=624.20 s  dm=2.18 kg  pos err=0.050 m  |vf|=0.0010 m/s  int|a|^2dt=0.0371
LS1 zemzev tf=624.20 s  dm=3.46 kg  pos err=0.000 m  |vf|=0.0014 m/s  int|a|^2dt=0.0594
```

---

## Simulation setup

| Item | Value |
|---|---|
| Asteroid | triaxial ellipsoid, semi-axes 1000 × 500 × 250 m, 3 g/cm³, 8 h rotation |
| Gravity | 2nd-order spherical harmonics, C₂₀ = −0.1125, C₂₂ = 0.0375 |
| Spacecraft | m₀ = 1400 kg, T_max = 80 N, c = 2206 m/s |
| SMC-IO gains | k_e = 0.006, ε_e = 10⁻⁴, b = (9, 8, 5), c = (0, 1, 4) |
| Integration | explicit Euler, h = 0.05 s (as in the paper) |
| Landing | stop when within 0.05 m of the site |

---

## Project timeline

- [x] **Report 1** (12 Oct): understanding the paper
- [x] **Report 2** (22 Oct): simulating the paper's results
- [ ] **Final report** (31 Oct): extensions and their validation
- [ ] **Presentation** (31 Oct)

## Planned extensions

- Behaviour near the pole (LS7) and a gravity model valid close to the surface
- Robustness to navigation noise, gravity-model errors and thrust errors (Monte Carlo)
- Guaranteeing feasibility of the per-step optimisation (adaptive *k_e* or slack variables)

---

*We used large language models (LLMs) to help understand the paper and write parts of the code and reports.*
