# ME324 Final Project — Natural Convection Heat Sink Optimization

Surrogate-guided optimization of an extruded aluminum fin heat sink for a PCB cabinet under natural convection, combining ANSYS Icepak CFD simulations with an RBF surrogate model.

## Problem

Design a heat sink for a PCB cabinet that dissipates 10 W at ambient 20 °C with minimum chip temperature.

| Parameter | Value |
|-----------|-------|
| Heat dissipation | 10 W |
| Ambient temperature | 20 °C |
| Chip footprint | 5 × 5 cm |
| Cabinet height | 43 mm |
| Fin material | Aluminum, k = 205 W/(m·K) |

**Design variables:** fin count N, base width w, base thickness t_b  
**Derived:** fin height H = 43 mm − t_b (cabinet-constrained)

## Approach

1. **CFD simulations** via ANSYS Icepak — buoyancy-driven natural convection in a closed cabinet
2. **RBF surrogate** (thin-plate spline) fitted on completed runs to interpolate T_max over the 4D design space
3. **Surrogate-guided search** — the Streamlit agent proposes new trial points in unexplored or high-interest regions
4. **Gradient-based optimization** (L-BFGS-B) over the surrogate for each discrete N value

### Key finding

The Bar-Cohen & Rohsenow open-channel correlation overpredicts convection by ~7×. Inside a closed cabinet, air recirculates rather than flowing freely, capping effective h at 5–10 W/(m²·K). Fins help primarily through increased surface area, not channel flow enhancement.

### Surrogate optimum (148 evaluated points)

| N | w (cm) | t_b (mm) | H (mm) | T_max (°C) |
|---|--------|----------|--------|------------|
| 8 | 8.95   | 3.0      | 34.9   | 81.5       |

## Files

| File | Description |
|------|-------------|
| `icepak_app.py` | Streamlit UI — configure, generate trial CSVs, upload results, explore surrogate |
| `icepak_agent.py` | Agent logic — RBF fitting, candidate selection, acquisition |
| `icepak_db.py` | SQLite database layer (all simulation results and predictions) |
| `pyproject.toml` | Project dependencies (uv) |

## Running

```bash
# Install dependencies
uv sync

# Launch the optimization dashboard
streamlit run icepak_app.py

# Re-fit surrogate and update manifold predictions
uv run update_manifold_preds.py

# Run gradient-descent optimization over surrogate
uv run optimize_gd.py
```

## Dependencies

- Python 3.13+
- numpy, scipy (RBF surrogate, L-BFGS-B)
- streamlit, plotly (dashboard)
- scikit-learn, matplotlib