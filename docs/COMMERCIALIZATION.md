# Publication & Commercialization Strategy

## Where to Publish

### Tier 1: Maximum Impact (Target 1-2)

| Journal | Impact Factor | Why | Timeline |
|---------|--------------|-----|----------|
| **Nature Machine Intelligence** | 25.9 | AI + science, Pearl's causality is hot topic | 6-12 months review |
| **Nuclear Fusion** | 3.5 | The top fusion journal, every fusion lab reads it | 3-6 months review |

**Recommended:** Submit to **Nuclear Fusion** first (domain audience = our customers) and simultaneously post on **arXiv** (immediate visibility).

### Tier 2: Fast Publication + Visibility

| Venue | Why | Timeline |
|-------|-----|----------|
| **arXiv:physics.plasm-ph + cs.AI** | Instant visibility, free, citable | 1-2 days |
| **Plasma Physics and Controlled Fusion** | Second-tier fusion journal, faster review | 2-4 months |
| **IEEE Transactions on Plasma Science** | Engineering audience, CFS/TAE read this | 3-6 months |
| **Scientific Reports (Nature)** | Open access, broad reach | 2-4 months |

### Tier 3: Conference Presentations

| Conference | When | Why |
|-----------|------|-----|
| **IAEA Fusion Energy Conference** | Oct 2026 | Every fusion decision-maker in one room |
| **APS Division of Plasma Physics** | Nov 2026 | US fusion community |
| **NeurIPS / ICML Workshop** | Dec 2026 | AI community (causal ML workshop) |
| **SOFT (Symposium on Fusion Technology)** | Sep 2026 | European fusion engineering |

### Publication Sequence

1. **Week 1:** Post on arXiv (cs.AI + physics.plasm-ph cross-list)
2. **Week 2:** Submit to Nuclear Fusion
3. **Month 2:** Submit to NeurIPS Causal ML Workshop (if deadline aligns)
4. **Month 3:** Present at relevant conference
5. **Month 6:** If Nuclear Fusion rejects, submit to PPCF or IEEE TPS

---

## Licensing Strategy

### Recommended: Dual License (BSL → Apache 2.0 + Commercial)

**Business Source License 1.1 (BSL)** is used by MariaDB, CockroachDB, and Sentry:
- **Free for research and non-production use** (universities, labs can use immediately)
- **Commercial license required for production use** (CFS, ITER, TAE must pay)
- **Auto-converts to Apache 2.0 after 4 years** (builds trust, ensures longevity)

This is ideal because:
- Universities/EUROfusion can use freely → builds citations and reputation
- CFS/ITER/TAE must pay for production deployment → revenue
- Patent protection covers the novel algorithms regardless of license

### License File

```
Business Source License 1.1

Licensor:  Dr. Mladen Mester
Software:  FusionMind 4.0

Use Limitation: Production use in fusion reactor control systems
                requires a commercial license from the Licensor.

Change Date:   2030-03-05
Change License: Apache License 2.0

For commercial licensing inquiries: mladen@fusionmind.ai
```

### What's Protected by Patents (independent of software license)

| Patent Family | What It Covers | Status |
|--------------|----------------|--------|
| PF1 (CPDE) | Ensemble causal discovery for plasma | Filing |
| PF2 (CPC) | Counterfactual reasoning for plasma control | Filing |
| PF6 (Stack) | 4-layer causal control architecture | Design |
| PF7 (CausalRL) | Causal RL integration | Design |

**Even if someone copies the code, the patents cover the methods.**

---

## Commercial Targets (Prioritized)

### Immediate (2026)

| Target | Contact Path | Deal Size | Approach |
|--------|-------------|-----------|----------|
| **ITER** (France) | ITER.org open calls for AI safety | $2-5M/yr | MODE A wrapper for regulatory compliance |
| **CFS** (Boston) | Via MIT PSFC connections | $500K-2M/yr | MODE A: protect SPARC magnets |
| **TAE Technologies** (CA) | Direct outreach, they need control stack | $1-3M + equity | MODE B: full control for FRC |
| **Tokamak Energy** (UK) | Via UKAEA/FAIR-MAST connection | $500K-1M/yr | Cross-device transfer MAST→ST40 |

### Medium-term (2027)

| Target | Deal Size | Why |
|--------|-----------|-----|
| **EUROfusion** | €5-10M grant | Causal analysis of 40 years of JET data |
| **General Atomics** | $1-2M/yr | DIII-D control system upgrade |
| **KSTAR (Korea)** | $500K-1M | Their RL needs explainability layer |

### How to Approach

1. **arXiv paper** → gets cited by fusion AI researchers → CFS/ITER see it
2. **IAEA conference talk** → direct access to ITER decision-makers
3. **Cold email to CFS AI team** with link to paper + GitHub + benchmark results
4. **UKAEA connection** via FAIR-MAST data usage → Tokamak Energy introduction

---

## Pricing Model

| Phase | What Customer Gets | Price |
|-------|-------------------|-------|
| Phase 1 (Wrapper) | Safety layer + explainability | $500K/year license |
| Phase 2 (Hybrid) | + Strategic causal control | $1-2M/year |
| Phase 3 (Full Stack) | Complete autonomous control | $2-5M/year + integration |
| Custom Integration | On-site deployment + training | $200K-500K one-time |
| Research License | Full access, non-production | Free (BSL terms) |
