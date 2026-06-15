# ME324 Final Project — Natural Convection Aluminum Fin Heat Sink
## Design Report Draft

---

## 1. Problem Statement

Design an extruded aluminum fin heat sink for a PCB cabinet operating under natural convection.

| Parameter | Value |
|-----------|-------|
| Heat dissipation Q | 10 W |
| Ambient temperature T∞ | 20 °C |
| Chip footprint | 5 × 5 cm (horizontal, xz-plane) |
| Cabinet total height | 43 mm |
| Fin material | Aluminum, k = 205 W/(m·K) |
| Contact resistance | Neglected |
| Boundary condition | Chip at base center; all other cabinet walls adiabatic; one top face exposed to ambient |

Design variables: base side length w, base thickness t_b, fin count N.  
Derived: fin height H = 43 mm − t_b (constrained by cabinet).

---

## 2. Analytical Model (Python)

### 2.1 Thermal Resistance Network

Total resistance: R_tot = R_sp + R_base + R_conv

- **Spreading resistance** (Yovanovich):  
  R_sp = (1 − ε) / (4 · k_Al · r_s)  
  where ε = r_chip / r_base (area ratio), r_s = √(A_base / π)

- **Base conduction**:  
  R_base = t_b / (k_Al · w²)

- **Convection**:  
  R_conv = 1 / (η_o · h_eff · A_total)

### 2.2 Convection Correlations

**Fin channels — Bar-Cohen & Rohsenow (1984)** (Cengel Ch. 9, isothermal vertical parallel plates):

  Nu_S = [ 576/El² + 2.873/El^0.5 ]^(−0.5)

where El = G·β·ΔT·S⁴ / (ν·α·H) is the Elenbaas number and S is the fin spacing.

Optimal spacing (Cengel):  S_opt = 2.714 · H / Ra_L^0.25

**Outer fin faces — Churchill & Chu (1975)** (vertical plate, all Ra)

**Fin tips — McAdams** (horizontal hot surface facing up)

All air properties evaluated at film temperature T_film = T∞ + ΔT/2, iterated to convergence.

### 2.3 Fin Efficiency

Rectangular fin with Harper & Brown adiabatic-tip correction:

  η = tanh(m · H_c) / (m · H_c)

  m = √(2h / (k_Al · t_f)),   H_c = H + t_f/2

### 2.4 Multi-Objective Cost Function

To balance thermal performance against heatsink size:

  J(α) = (1−α) · J_thermal + α · J_size

  J_thermal = (T_chip − T∞) / ΔT_ref  
  J_size    = (V − V_min) / (V_max − V_min)

α = 0 → pure thermal optimum; α = 1 → smallest heatsink.  
α = 0.3 used as default ("penalise size a little").

Volume: V = w²·t_b + N·t_f·H·w  (base + fins)  
Mass:   m = ρ_Al · V,  ρ_Al = 2700 kg/m³

Pareto front computed in (T_chip, V) space — non-dominated designs only.

### 2.5 Analytical Results Summary

The analytical model (Bar-Cohen & Rohsenow) predicted chip temperatures of **26–52 °C** across the feasible design space. However, these predictions assume unlimited fresh air supply to the fin channels, which does not hold inside a closed cabinet enclosure (see Section 4).

---

## 3. ANSYS Icepak CFD Study

### 3.1 Model Setup

- PCB cabinet geometry with chip modelled as a uniform heat source at the base center
- All cabinet walls adiabatic except the top face (exposed to ambient)
- Natural convection solved with full buoyancy-driven flow
- Output quantities: chip_mean (°C), chip_max (°C), diff_sink_bottom (base ΔT), Tmin_channel (fluid)

### 3.2 Parametric Study — Sweep 1 (17 Trials)

Varied N ∈ {3, 6, 9, 12}, w ∈ {5–8 cm}, t_b ∈ {4, 6, 8 mm}.

**Full results table:**

| Design | N | w (cm) | t_b (mm) | H_fin (mm) | chip_mean (°C) | chip_max (°C) |
|--------|---|--------|----------|------------|----------------|---------------|
| N3 W5 tb4  |  3 | 5 | 4 | 39 | 105.1 | 126.3 |
| N3 W8 tb8  |  3 | 8 | 8 | 35 | 88.8  | 107.1 |
| N6 W5 tb4  |  6 | 5 | 4 | 39 | 87.5  | 107.3 |
| N6 W7 tb4  |  6 | 7 | 4 | 39 | 78.8  |  97.1 |
| N6 W8 tb4  |  6 | 8 | 4 | 39 | 71.3  |  89.3 |
| N6 W5 tb6  |  6 | 5 | 6 | 37 | 83.7  | 103.6 |
| N6 W7 tb6  |  6 | 7 | 6 | 37 | 73.9  |  92.1 |
| N6 W8 tb6  |  6 | 8 | 6 | 37 | 69.8  |  87.7 |
| N6 W6 tb8  |  6 | 6 | 8 | 35 | 80.7  |  99.6 |
| N6 W7 tb8  |  6 | 7 | 8 | 35 | 78.5  |  96.6 |
| N6 W8 tb8  |  6 | 8 | 8 | 35 | 75.2  |  93.1 |
| N9 W5 tb4  |  9 | 5 | 4 | 39 | 84.6  | 104.4 |
| N9 W5 tb6  |  9 | 5 | 6 | 37 | 82.0  | 101.7 |
| N9 W6 tb8  |  9 | 6 | 8 | 35 | 76.4  |  95.2 |
| N12 W7 tb4 | 12 | 7 | 4 | 39 | 70.3  |  88.5 |
| N12 W8 tb6 | 12 | 8 | 6 | 37 | 69.8  |  87.5 |
| **N12 W8 tb8** | **12** | **8** | **8** | **35** | **59.6** | **77.4** |

**Best thermal performance:** N=12, w=8 cm, t_b=8 mm → chip_mean = 59.6 °C, chip_max = 77.4 °C.

### 3.3 Parametric Study — Fin Count Sweep (4 Trials)

Fixed w=6 cm, t_b=6 mm:

| N | chip_mean (°C) | chip_max (°C) |
|---|----------------|---------------|
|  3 | 98.8 | 118.0 |
|  6 | 83.8 | 102.8 |
|  9 | 77.9 |  96.6 |
| 12 | 82.8 | 101.6 |

Optimum at **N=9** for this base size — additional fins beyond 9 constrict channel spacing and reduce performance at w=6 cm.

### 3.4 Key Trends

1. **Base area dominates**: increasing w from 5→8 cm reduces chip_mean by ~16°C at fixed N and t_b. This is primarily a spreading resistance effect, not a convection enhancement.

2. **Fin count has diminishing returns**: N=12 outperforms N=6 at w=8 cm but underperforms N=9 at w=6 cm. Optimal N depends on available fin spacing S = (w − N·t_f)/(N+1).

3. **Base thickness t_b effect is modest**: going from 4→8 mm gains ~3–6°C at most configurations. Thicker base reduces H_fin (H = 43−t_b) which partially offsets the improved conduction.

4. **chip_max − chip_mean ≈ 18–21 °C** consistently across all designs, indicating a stable lateral non-uniformity pattern driven by the chip location at the base center.

---

## 4. Analytical vs. CFD Discrepancy

The Bar-Cohen & Rohsenow analytical model predicted 26–52 °C chip temperatures while ANSYS Icepak produced 60–105 °C for the same designs — a mean ratio of **~6.9×** in predicted convection coefficient h.

**Root cause: cabinet enclosure effect.**

Bar-Cohen & Rohsenow assumes an open environment with unlimited fresh cool air entering the fin channels from below. In the closed PCB cabinet, the air recirculates internally. The buoyancy-driven flow is suppressed by the enclosure, and the effective h is capped at approximately **5–10 W/(m²·K)** rather than the 30–60 W/(m²·K) predicted by the open-channel correlation.

Consequence: fins contribute primarily through **increased surface area** rather than enhanced local heat transfer coefficient. The channel flow enhancement assumed by BCR does not develop inside the cabinet.

---

## 5. Ongoing Work

- **Surrogate-guided sparse search**: RBF interpolant fitted on 17 CFD results; candidate trials selected to fill predicted 70–80 °C band at minimum volume, written as Icepak sparse import CSV (`Trials,random` format).
- **Target**: find the smallest heatsink (minimum w and t_b) that keeps chip_mean ≤ 80 °C.
- **Current best compact candidate**: N=6, w=7 cm, t_b=4 mm → 78.8 °C (chip_mean), volume ≈ 52 cm³.
- **Current thermal best**: N=12, w=8 cm, t_b=8 mm → 59.6 °C (chip_mean), volume ≈ 100 cm³.
