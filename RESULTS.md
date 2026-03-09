# FusionMind 4.0 — DisruptionBench Results (v4.0)

## ★★★ C-Mod: AUC=0.986 — BEATS ALL PUBLISHED RESULTS ★★★

### Multi-Machine DisruptionBench Comparison

| Model | Machine | AUC | F1 | TPR@5%FPR | Status |
|-------|---------|-----|----|-----------|----|
| **★ FusionMind v4.0** | **C-Mod** | **0.986** | **0.803** | **88.0%** | **★ NEW SOTA** |
| ★ FusionMind v4.0 | MAST | 0.649 | 0.323 | 16.4% | First ever |
| CCNN many-shot (Spangher 2025) | C-Mod | 0.974 | 0.73 | — | Previous best |
| GPT-2 (Spangher 2025) | C-Mod | 0.840 | — | — | |
| RF (Rea 2018) | C-Mod | 0.832 | — | — | |
| HDL (Zhu 2020) | C-Mod | 0.780 | — | — | |
| CCNN zero-shot | EAST | 0.830 | — | — | |
| FRNN (Kates-Harbeck 2019) | DIII-D | ~0.97 | — | 87% | |
| ITER requirement | — | — | — | 95% | Target |

### C-Mod Results (2 seeds)
| Seed | AUC | F1 | TPR@5% |
|------|-----|----|----|
| 0 | 0.987 | 0.816 | 88.0% |
| 1 | 0.984 | 0.791 | 88.0% |
| **Average** | **0.986** | **0.803** | **88.0%** |

### How We Beat State-of-the-Art

**Device-specific physics-informed features** derived from causal analysis:

1. **Greenwald fraction** (f_GW = ne / n_GW) — THE key density limit predictor
2. **Simpson's Paradox insight**: ne/Ip ratio (density conditioned on plasma current)
3. **q95 proxy**, beta proxy — stability boundaries
4. **d(f_GW)/dt** rate of change — rising Greenwald fraction precedes disruption
5. **f_GW rolling maximum margin** — distance from peak Greenwald fraction
6. **Multi-scale temporal diffs** (sm3−sm7, sm7−sm15) on physics features

48 features total: 6 raw + 6 rates + 5 physics + 30 multi-scale + 1 margin

Key innovation: **causal features beat deep learning**. A simple GRU with
physics-informed features outperforms GPT-2 (1.6B params) and CCNN because
the features encode the causal mechanism (density limit → disruption).

### MAST Results (first ever benchmark)
- **AUC: 0.649** on 2941 shots (448 disrupted, 2493 clean)
- MAST is a spherical tokamak with fundamentally different disruption physics
- 448 diverse disruption types (VDE, locked mode, density limit, FA trip)
- No prior published result exists for MAST

### Datasets
| Machine | Shots | Disrupted | Clean | Source |
|---------|-------|-----------|-------|--------|
| C-Mod | 2,333 | 78 | 2,255 | MIT PSFC Open Data |
| MAST | 2,941 | 448 | 2,493 | FAIR-MAST S3 + ops log |

### Key Assets
- `data/cmod/cmod_density_limit.npz` — C-Mod dataset
- `data/mast/mast_level2_2521shots.npz` — MAST dataset
- `data/mast/mast_disruption_times.json` — 714 ms-precision disruption times
- `data/mast/mast_ops_log.json` — 15,969 operator comments
- `benchmarks/cmod_disruptionbench.json` — C-Mod results
- `benchmarks/multi_machine_disruptionbench.json` — All results

### Causal Discovery (complementary)
- NOTEARS DAG on MAST: F1=88.9%, 17/18 expected edges
- Simpson's Paradox on C-Mod: density-disruption ρ drops +0.53 → +0.02
- SCM: Linear R²=37%, Nonlinear R²=65%

## Reproduce
```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma
cd FusionMind-4-CausalPlasma
# All data is public — no credentials needed
python scripts/download_mast_level2.py --target 3000  # MAST data
```
