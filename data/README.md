# FusionMind Cross-Device Dataset

Multi-machine tokamak data for causal disruption prediction and cross-device transfer learning.

## Devices

### MAST (Mega Amp Spherical Tokamak) — UKAEA, UK
- **Type:** Spherical tokamak (low aspect ratio, R/a ≈ 0.85/0.65 m)
- **Source:** FAIR-MAST S3 archive (s3.echo.stfc.ac.uk)
- **Shots:** 190 (83 disrupted + 107 clean)
- **Timepoints:** 15,601 (EFIT timebase, ~10ms resolution)
- **Variables (13):** βN, βp, q95, q_axis, elongation, li, Wplasma, βt, Ip, Dα, MHD n=2, ne_line, Mirnov RMS
- **Format:** `mast/mast_13var.npz` (data, shot_ids, variables)
- **Disruption labels:** Ip quench identification from FAIR-MAST metadata
- **File:** `mast/disruption_info.json` — shot-level disrupted/clean lists

### Alcator C-Mod — MIT PSFC, USA
- **Type:** Conventional tokamak (high field, compact, R/a ≈ 0.67/0.22 m, B ≤ 8T)
- **Source:** [MIT PSFC Open Density Limit Database](https://github.com/MIT-PSFC/open_density_limit_database)
- **Shots:** 2,333 (78 density-limit disruptions + 2,255 clean)
- **Timepoints:** 264,385 (10ms timebase)
- **Variables (6):** density, elongation, minor_radius, plasma_current, toroidal_B_field, triangularity
- **Format:** `cmod/cmod_density_limit.npz` (data, shot_ids, variables, disruption_labels, time)
- **Disruption type:** Density limit (MARFE / radiative collapse)
- **File:** `cmod/disruption_info.json` — shot-level disrupted/clean lists
- **License:** CC BY (MIT PSFC)

## Cross-Device Comparison

| Property | MAST | C-Mod |
|----------|------|-------|
| Configuration | Spherical (ST) | Conventional |
| Major radius | 0.85 m | 0.67 m |
| Aspect ratio | ~1.3 | ~3.1 |
| Magnetic field | 0.5 T | 2.3–8.0 T |
| Plasma current | 0.1–0.9 MA | 0.2–1.8 MA |
| Disruption types | Mixed (MHD, density, VDE) | Density limit |
| Common variables | elongation, Ip | elongation, Ip |

## Usage

```python
import numpy as np, json

# Load MAST data
mast = np.load('data/mast/mast_13var.npz', allow_pickle=True)
D_mast = mast['data']          # (15601, 13)
L_mast = mast['shot_ids']     # (15601,)
vars_mast = list(mast['variables'])

with open('data/mast/disruption_info.json') as f:
    di_mast = json.load(f)

# Load C-Mod data
cmod = np.load('data/cmod/cmod_density_limit.npz', allow_pickle=True)
D_cmod = cmod['data']          # (264385, 6)
L_cmod = cmod['shot_ids']     # (264385,)
y_cmod = cmod['disruption_labels']  # per-timepoint labels
```

## Cross-Device Transfer Tasks

1. **MAST → C-Mod (zero-shot):** Train causal graph on MAST, test disruption prediction on C-Mod using only common variables (elongation, Ip)
2. **C-Mod → MAST:** Train on 2,333 C-Mod shots, test on 190 MAST shots
3. **Dimensionless transfer:** Normalize to βN, ν*, ρ*, q95, H98 for device-independent features

## Citation

MAST data: FAIR-MAST, UKAEA (https://fair-mast.readthedocs.io/)
C-Mod data: Maris et al., "The Open Density Limit Database," MIT PSFC (2025)

## ITPA Disruption Database (IDDB) — Access Guide

The ITPA DDB contains ~3,500+ disruptions from 9 tokamaks: ADITYA, C-Mod, AUG, DIII-D, JET, JT-60U, TCV, MAST, NSTX.

### How to access

**MDSplus server:** `iddb.gat.com:9000`
- Tree names: `DDB_D3D`, `DDB_JET`, `DDB_CMOD`, `DDB_MAST`, `DDB_NSTX`, etc.
- Variables: see https://fusion.gat.com/itpa-ddb/MDSplusVar/
- Shot list: https://fusion.gat.com/iter/disruptiondb/working/shots.php

**SQL server:** `d3drdb.gat.com:8001`, database `DDB`
- Variables: https://fusion.gat.com/itpa-ddb/SQLVar/

**Requirements:**
1. Install MDSplus: https://www.mdsplus.org/
2. Request access via ITPA database guidelines (cite Eidietis et al., Nucl. Fusion 55, 063030, 2015)
3. GA network access (VPN or institutional affiliation)

**Key variables available:**
- `\IPD` (plasma current), `\BETAND` (normalized beta), `\INTLID` (li)
- `\Q95D` (q95), `\KAPPAD` (elongation), `\TIMED` (disruption time)
- `\CAUSED` (disruption cause), `\VDE_E` (VDE indicator)
- `\IPT` (Ip time trace through disruption)
- Halo currents, impurity injection, radiation data

### DisruptionBench Dataset (~30K shots)

The DisruptionBench framework (MIT-PSFC/DisruptionBench) uses a processed version of data from C-Mod, DIII-D, and EAST originally published by Zhu et al. (Nuclear Communications, 2021). The benchmarking framework is open-source, but the underlying data requires MIT PSFC MDSplus server access via `disruption-py` (https://github.com/MIT-PSFC/disruption-py).

**To request access:** Contact disruption-py@mit.edu or register at https://fusion.gat.com/itpa-ddb/ElecLog/

### What we have (fully public, no registration needed)

| Source | Shots | Access |
|--------|-------|--------|
| FAIR-MAST (this repo) | 210 | S3 anonymous (s3.echo.stfc.ac.uk) |
| MIT Open Density Limit DB (this repo) | 2,333 | GitHub CC BY |
| ITPA DDB | ~3,500+ | MDSplus + GA access required |
| DisruptionBench | ~30,000 | MDSplus + MIT access required |
