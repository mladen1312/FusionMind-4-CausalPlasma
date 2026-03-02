import { useState } from "react";

// ═══════════════════════════════════════════════════════════
// REAL VALIDATION DATA FROM FAIR-MAST PIPELINE
// ═══════════════════════════════════════════════════════════
const MAST_RESULTS = {
  source: "UKAEA FAIR-MAST (s3.echo.stfc.ac.uk)",
  shots_loaded: 8,
  shots_attempted: 10,
  timepoints: 625,
  variables: 9,
  edges_discovered: 11,
  physics_checks: { passed: 2, total: 5 },
  key_discoveries: [
    { edge: "βN → βp", corr: 0.945, note: "Pressure coupling confirmed" },
    { edge: "q_axis → li", corr: null, note: "MHD stability chain found" },
    { edge: "κ → q95", corr: null, note: "Geometry-safety link discovered" },
    { edge: "li → βN", corr: null, note: "Internal inductance feedback" },
  ],
  missing: ["P_NBI → βN (actuator not resolved)", "ne_core → βp (scale mismatch)"],
};

const D3R_RESULTS = {
  grid: "48×48",
  compression: "156:1",
  rmse: "4.50 ± 0.03 keV",
  physics_checks: { passed: 3, total: 5 },
  strengths: ["Positivity enforced", "Boundary constraint works", "High compression"],
  weaknesses: ["Relative error >100% (PoC score network)", "Smoothness not yet achieved"],
};

const PATENT_FAMILIES = [
  {
    id: "PF1", name: "Causal Discovery Engine", abbr: "CPDE",
    novelty: 9, status: "Validated", trl: 3,
    metric: "F1=88.9% (synth) / 11 edges (MAST)",
    detail: "NOTEARS + Granger + PC ensemble on 14-var plasma model. Real MAST: 625 timepoints from 8 shots.",
    honest: "Linear SCM only. No GPU optimizer. NOTEARS is toy-scale vs full DAG-GNN approaches.",
    value: "$50–100M",
  },
  {
    id: "PF2", name: "Counterfactual Controller", abbr: "CPC",
    novelty: 10, status: "Validated", trl: 2,
    metric: "All do(X) queries physically correct",
    detail: "SCM-based counterfactual reasoning: P(Y|do(X)). Answers 'what would happen if we changed NBI power?'",
    honest: "Linear SCM — real plasma is nonlinear. No real-time loop. Cannot match 4kHz RL control speed.",
    value: "$80–150M",
  },
  {
    id: "PF3", name: "Universal Plasma Foundation", abbr: "UPFM",
    novelty: 8, status: "PoC", trl: 2,
    metric: "CV=0.267 across 6 devices",
    detail: "Dimensionless tokenization (βn, ν*, ρ*, q95, H98) for cross-device transfer learning.",
    honest: "Single-layer PoC. TokaMind already has full transformer on MAST. No pretraining at scale.",
    value: "$100–200M",
  },
  {
    id: "PF4", name: "Diffusion 3D Reconstruction", abbr: "D3R",
    novelty: 7, status: "PoC", trl: 2,
    metric: "156:1 compression, MAST geometry",
    detail: "Conditional denoising diffusion for 2D/3D plasma state from sparse diagnostics.",
    honest: "Score network is Gaussian prior, not neural. 111% relative error. Needs U-Net backbone.",
    value: "$30–60M",
  },
  {
    id: "PF5", name: "Active Experiment Design", abbr: "AEDE",
    novelty: 7, status: "PoC", trl: 2,
    metric: "Bootstrap uncertainty + ranking",
    detail: "Suggests most informative experiments to resolve causal ambiguity.",
    honest: "No integration with any experiment planning system. Purely algorithmic.",
    value: "$20–40M",
  },
];

const COMPETITORS = [
  {
    name: "DeepMind / CFS",
    approach: "RL coil control (TCV 2022, Nature)",
    strength: "Live plasma control demonstrated. Working on SPARC.",
    weakness: "No causal reasoning. Black-box RL.",
    threat: "HIGH",
    color: "#EF4444",
  },
  {
    name: "KSTAR / SNU",
    approach: "RL tearing mode avoidance",
    strength: "Real-time 4kHz control on operating tokamak.",
    weakness: "Single instability focus. No cross-device transfer.",
    threat: "HIGH",
    color: "#F59E0B",
  },
  {
    name: "TokaMind (UKAEA)",
    approach: "Foundation model on MAST data",
    strength: "Full transformer, real data, Feb 2026 release.",
    weakness: "Correlational — no causal/counterfactual capability.",
    threat: "MEDIUM",
    color: "#F59E0B",
  },
  {
    name: "Princeton PPPL",
    approach: "Diag2Diag, Kalman filters",
    strength: "Deep physics expertise. NSTX-U access.",
    weakness: "Traditional ML, not causal. Slower innovation cycle.",
    threat: "LOW",
    color: "#22C55E",
  },
];

const TRL_LABELS = {
  1: "Basic principles", 2: "Technology concept", 3: "Analytical PoC",
  4: "Lab validation", 5: "Relevant environment", 6: "Demonstrated",
  7: "Operational prototype", 8: "Qualified", 9: "Flight proven",
};

// ═══════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════

function MetricCard({ label, value, sub, color = "#60A5FA" }) {
  return (
    <div style={{
      background: "rgba(15,23,42,0.8)", borderRadius: 12, padding: "16px 20px",
      border: `1px solid ${color}22`, minWidth: 140,
    }}>
      <div style={{ fontSize: 11, color: "#64748B", letterSpacing: 1, textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 800, color, marginTop: 4, fontFamily: "'DM Mono', monospace" }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: "#94A3B8", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function TRLBar({ level }) {
  return (
    <div style={{ display: "flex", gap: 2, alignItems: "center" }}>
      {[1,2,3,4,5,6,7,8,9].map(i => (
        <div key={i} style={{
          width: 14, height: 8, borderRadius: 2,
          background: i <= level ? (i <= 3 ? "#3B82F6" : i <= 6 ? "#22C55E" : "#F59E0B") : "#1E293B",
        }} />
      ))}
      <span style={{ fontSize: 9, color: "#64748B", marginLeft: 4 }}>TRL {level}</span>
    </div>
  );
}

function HonestyTag({ text }) {
  return (
    <div style={{
      fontSize: 10, color: "#F59E0B", background: "#422006", borderRadius: 6,
      padding: "4px 8px", border: "1px solid #713F12", lineHeight: 1.4, marginTop: 6,
    }}>
      ⚠ {text}
    </div>
  );
}

function PatentCard({ pf, expanded, onToggle }) {
  return (
    <div
      onClick={onToggle}
      style={{
        background: expanded ? "#0F172A" : "rgba(15,23,42,0.6)",
        borderRadius: 12, padding: 16, cursor: "pointer",
        border: `1px solid ${expanded ? "#3B82F6" : "#1E293B"}`,
        transition: "all 0.2s",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{
              fontSize: 10, fontWeight: 800, color: "#0F172A", background: "#3B82F6",
              padding: "2px 6px", borderRadius: 4,
            }}>{pf.id}</span>
            <span style={{ fontSize: 14, fontWeight: 700, color: "#F1F5F9" }}>{pf.name}</span>
            <span style={{
              fontSize: 9, padding: "2px 6px", borderRadius: 10,
              background: pf.status === "Validated" ? "#052E16" : "#1E1B4B",
              color: pf.status === "Validated" ? "#4ADE80" : "#A78BFA",
              border: `1px solid ${pf.status === "Validated" ? "#166534" : "#4C1D95"}`,
            }}>{pf.status}</span>
          </div>
          <div style={{ fontSize: 11, color: "#94A3B8", marginTop: 4, fontFamily: "'DM Mono', monospace" }}>
            {pf.metric}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 11, color: "#22C55E", fontWeight: 700 }}>{pf.value}</div>
          <div style={{ marginTop: 4 }}><TRLBar level={pf.trl} /></div>
        </div>
      </div>

      {expanded && (
        <div style={{ marginTop: 12, borderTop: "1px solid #1E293B", paddingTop: 10 }}>
          <div style={{ fontSize: 11, color: "#CBD5E1", lineHeight: 1.5, marginBottom: 6 }}>{pf.detail}</div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 10, color: "#64748B" }}>Novelty:</span>
            <div style={{ display: "flex", gap: 1 }}>
              {[1,2,3,4,5,6,7,8,9,10].map(i => (
                <div key={i} style={{
                  width: 10, height: 10, borderRadius: 2,
                  background: i <= pf.novelty ? "#3B82F6" : "#1E293B",
                }} />
              ))}
            </div>
            <span style={{ fontSize: 10, color: "#3B82F6" }}>{pf.novelty}/10</span>
          </div>
          <HonestyTag text={pf.honest} />
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// MAIN DASHBOARD
// ═══════════════════════════════════════════════════════════

export default function InvestorDashboard() {
  const [tab, setTab] = useState("overview");
  const [expandedPF, setExpandedPF] = useState(null);

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "validation", label: "Real Data" },
    { id: "landscape", label: "Competitive" },
    { id: "roadmap", label: "Roadmap" },
  ];

  return (
    <div style={{
      background: "#020617", minHeight: "100vh", color: "#E2E8F0",
      fontFamily: "'Inter', 'DM Sans', system-ui, sans-serif",
      padding: "20px 16px",
      maxWidth: 800, margin: "0 auto",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 24 }}>
        <div style={{ fontSize: 10, letterSpacing: 3, color: "#475569", marginBottom: 6 }}>
          CONFIDENTIAL — INVESTOR BRIEFING
        </div>
        <h1 style={{
          fontSize: 28, fontWeight: 900, margin: 0, letterSpacing: -0.5,
          background: "linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #F59E0B 100%)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
        }}>
          FusionMind 4.0
        </h1>
        <div style={{ fontSize: 12, color: "#94A3B8", marginTop: 4 }}>
          Causal AI for Fusion Plasma Control — Proof of Concept
        </div>
        <div style={{
          fontSize: 10, color: "#F59E0B", marginTop: 8, padding: "4px 12px",
          background: "#422006", borderRadius: 20, display: "inline-block",
          border: "1px solid #713F12",
        }}>
          TRL 2–3 · Pre-Patent · March 2026
        </div>
      </div>

      {/* Tabs */}
      <div style={{
        display: "flex", gap: 4, marginBottom: 20, justifyContent: "center",
        background: "#0F172A", borderRadius: 12, padding: 4,
      }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: "8px 16px", borderRadius: 8, border: "none", fontSize: 12,
            fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
            background: tab === t.id ? "#1E293B" : "transparent",
            color: tab === t.id ? "#F1F5F9" : "#64748B",
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ═══ OVERVIEW ═══ */}
      {tab === "overview" && (
        <div>
          {/* Core thesis */}
          <div style={{
            background: "#0F172A", borderRadius: 12, padding: 20,
            border: "1px solid #1E293B", marginBottom: 16,
          }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#F1F5F9", marginBottom: 8 }}>
              Core Thesis
            </div>
            <div style={{ fontSize: 12, color: "#CBD5E1", lineHeight: 1.6 }}>
              Every existing fusion AI system — DeepMind, KSTAR, Princeton, TokaMind — is{" "}
              <span style={{ color: "#F59E0B", fontWeight: 700 }}>purely correlational</span>.
              They learn "when X happens, Y follows" but cannot answer{" "}
              <span style={{ color: "#3B82F6", fontWeight: 700 }}>"would Y change if we intervened on X?"</span>{" "}
              FusionMind 4.0 introduces Pearl's do-calculus to tokamak physics, enabling counterfactual
              reasoning, causal safety guarantees, and explainable control decisions.
            </div>
            <div style={{
              marginTop: 12, fontSize: 11, color: "#94A3B8", lineHeight: 1.6,
              borderTop: "1px solid #1E293B", paddingTop: 10,
            }}>
              <strong style={{ color: "#F59E0B" }}>Honest positioning:</strong> FusionMind is not a replacement for
              DeepMind's RL controller. It is a <strong style={{ color: "#10B981" }}>causal safety layer</strong> —
              the next capability fusion needs after basic RL stability is solved. Think of it as the "why"
              behind the RL's "what".
            </div>
          </div>

          {/* Top metrics */}
          <div style={{
            display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 16,
          }}>
            <MetricCard label="Patent Families" value="5" sub="6 US Provisionals planned" />
            <MetricCard label="Real Data" value="625" sub="MAST timepoints (8 shots)" color="#10B981" />
            <MetricCard label="Causal Edges" value="11" sub="Discovered on real MAST" color="#F59E0B" />
            <MetricCard label="Portfolio" value="$280M+" sub="If validated at TRL 5+" color="#A78BFA" />
          </div>

          {/* Patent families */}
          <div style={{ fontSize: 11, fontWeight: 700, color: "#64748B", letterSpacing: 1, marginBottom: 8 }}>
            PATENT FAMILIES
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {PATENT_FAMILIES.map(pf => (
              <PatentCard
                key={pf.id}
                pf={pf}
                expanded={expandedPF === pf.id}
                onToggle={() => setExpandedPF(expandedPF === pf.id ? null : pf.id)}
              />
            ))}
          </div>

          {/* Current limitations box */}
          <div style={{
            marginTop: 16, background: "#1C0A0A", borderRadius: 12, padding: 16,
            border: "1px solid #450A0A",
          }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#FCA5A5", marginBottom: 8 }}>
              Current Limitations (Transparent)
            </div>
            {[
              "PoC stage — 1 GitHub repo, NumPy-only, no GPU acceleration",
              "Linear SCM — real plasma is nonlinear (RL/transformers handle this better today)",
              "No real-time control loop — RL competitors run at 4kHz on live tokamaks",
              "No integration with TORAX, OMAS, or PlasmaPy ecosystems",
              "Only Alcator C-Mod + MAST validation — no live experiment on DIII-D/TCV/KSTAR",
              "Competitors have Nature papers + deployed systems; we have a prototype",
            ].map((item, i) => (
              <div key={i} style={{
                fontSize: 10, color: "#FDA4AF", padding: "3px 0",
                display: "flex", gap: 6, lineHeight: 1.4,
              }}>
                <span style={{ color: "#EF4444", flexShrink: 0 }}>✗</span>
                <span>{item}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ═══ REAL DATA ═══ */}
      {tab === "validation" && (
        <div>
          <div style={{
            background: "#0F172A", borderRadius: 12, padding: 16,
            border: "1px solid #1E293B", marginBottom: 12,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#F1F5F9" }}>FAIR-MAST Real Data Validation</div>
                <div style={{ fontSize: 10, color: "#64748B", marginTop: 2 }}>
                  Source: {MAST_RESULTS.source}
                </div>
              </div>
              <div style={{
                fontSize: 9, padding: "3px 8px", borderRadius: 10,
                background: "#052E16", color: "#4ADE80", border: "1px solid #166534",
              }}>LIVE S3 DATA</div>
            </div>
          </div>

          <div style={{
            display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginBottom: 12,
          }}>
            <MetricCard label="Shots Loaded" value={`${MAST_RESULTS.shots_loaded}/${MAST_RESULTS.shots_attempted}`} sub="M5–M9 campaigns" />
            <MetricCard label="Timepoints" value={MAST_RESULTS.timepoints} sub="EFIT-aligned" color="#10B981" />
            <MetricCard label="Variables" value={MAST_RESULTS.variables} sub="βN, βp, q95, ne, P_NBI..." color="#F59E0B" />
          </div>

          {/* Discovered edges */}
          <div style={{
            background: "#0F172A", borderRadius: 12, padding: 16,
            border: "1px solid #1E293B", marginBottom: 12,
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#22C55E", marginBottom: 8 }}>
              CAUSAL EDGES DISCOVERED ON REAL MAST DATA
            </div>
            {MAST_RESULTS.key_discoveries.map((d, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", gap: 8, padding: "5px 0",
                borderBottom: i < MAST_RESULTS.key_discoveries.length - 1 ? "1px solid #1E293B" : "none",
              }}>
                <span style={{ color: "#22C55E", fontSize: 12 }}>→</span>
                <span style={{ fontSize: 11, fontWeight: 700, color: "#3B82F6", fontFamily: "'DM Mono', monospace", minWidth: 100 }}>{d.edge}</span>
                {d.corr && <span style={{ fontSize: 9, color: "#64748B" }}>r={d.corr}</span>}
                <span style={{ fontSize: 10, color: "#94A3B8", marginLeft: "auto" }}>{d.note}</span>
              </div>
            ))}
            <div style={{ marginTop: 10, fontSize: 10, color: "#F59E0B" }}>
              ⚠ Not resolved: {MAST_RESULTS.missing.join(" · ")}
            </div>
          </div>

          {/* Key finding */}
          <div style={{
            background: "linear-gradient(135deg, #0C1222, #0F172A)", borderRadius: 12, padding: 16,
            border: "1px solid #3B82F6", marginBottom: 12,
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#60A5FA", marginBottom: 6 }}>
              KEY FINDING: βN ↔ βp CAUSAL COUPLING
            </div>
            <div style={{ fontSize: 11, color: "#CBD5E1", lineHeight: 1.5 }}>
              Pearson correlation r=0.945 across 625 real MAST timepoints. CPDE correctly discovers
              βN → βp as direct causal edge (normalized beta drives poloidal beta via pressure).
              Also discovers q_axis → li chain — the MHD stability pathway.
            </div>
            <div style={{ fontSize: 10, color: "#94A3B8", marginTop: 6 }}>
              Previously validated on Alcator C-Mod: AUC=0.974 for density limit prediction,
              Simpson's Paradox detected in density–disruption correlations.
            </div>
          </div>

          {/* D3R results */}
          <div style={{
            background: "#0F172A", borderRadius: 12, padding: 16,
            border: "1px solid #1E293B",
          }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#F1F5F9", marginBottom: 8 }}>
              D3R — Diffusion Reconstruction (PF4)
            </div>
            <div style={{
              display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginBottom: 10,
            }}>
              <MetricCard label="Compression" value={D3R_RESULTS.compression} color="#A78BFA" />
              <MetricCard label="Grid" value={D3R_RESULTS.grid} sub="MAST poloidal" color="#60A5FA" />
              <MetricCard label="Physics" value={`${D3R_RESULTS.physics_checks.passed}/${D3R_RESULTS.physics_checks.total}`} color="#F59E0B" />
            </div>
            <HonestyTag text="RMSE 4.50 keV (>100% relative error). Score network is Gaussian prior, not neural U-Net. This is a mathematical PoC, not a production reconstructor." />
          </div>
        </div>
      )}

      {/* ═══ COMPETITIVE ═══ */}
      {tab === "landscape" && (
        <div>
          <div style={{
            background: "#0F172A", borderRadius: 12, padding: 16,
            border: "1px solid #1E293B", marginBottom: 16,
          }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#F1F5F9", marginBottom: 4 }}>
              Where FusionMind Sits
            </div>
            <div style={{ fontSize: 11, color: "#94A3B8", lineHeight: 1.5 }}>
              Fusion AI is a 2×2 matrix: <strong style={{ color: "#CBD5E1" }}>correlational vs causal</strong> and{" "}
              <strong style={{ color: "#CBD5E1" }}>deployed vs prototype</strong>.
              Everyone else is in the "correlational + deployed" quadrant.
              FusionMind is alone in "causal + prototype". The question is whether we can
              reach "causal + deployed" before others add causal reasoning to their deployed systems.
            </div>
          </div>

          {/* Matrix visual */}
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, marginBottom: 16,
            background: "#1E293B", borderRadius: 12, padding: 2, overflow: "hidden",
          }}>
            <div style={{ background: "#0A1628", borderRadius: "10px 0 0 0", padding: 14 }}>
              <div style={{ fontSize: 9, color: "#64748B", textTransform: "uppercase", letterSpacing: 1 }}>
                Causal + Deployed
              </div>
              <div style={{ fontSize: 18, color: "#22C55E", fontWeight: 800, marginTop: 8 }}>TARGET</div>
              <div style={{ fontSize: 10, color: "#94A3B8", marginTop: 4 }}>2027–2028 with TORAX integration + DIII-D test</div>
            </div>
            <div style={{ background: "#0F172A", borderRadius: "0 10px 0 0", padding: 14 }}>
              <div style={{ fontSize: 9, color: "#64748B", textTransform: "uppercase", letterSpacing: 1 }}>
                Correlational + Deployed
              </div>
              <div style={{ fontSize: 10, marginTop: 8 }}>
                {["DeepMind/CFS (TCV)", "KSTAR RL (4kHz)", "PPPL Kalman"].map((c, i) => (
                  <div key={i} style={{ color: "#EF4444", padding: "2px 0", fontWeight: 600 }}>{c}</div>
                ))}
              </div>
            </div>
            <div style={{ background: "#0F172A", borderRadius: "0 0 0 10px", padding: 14 }}>
              <div style={{ fontSize: 9, color: "#64748B", textTransform: "uppercase", letterSpacing: 1 }}>
                Causal + Prototype
              </div>
              <div style={{
                fontSize: 14, fontWeight: 800, marginTop: 8,
                background: "linear-gradient(90deg, #3B82F6, #10B981)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>FusionMind 4.0</div>
              <div style={{ fontSize: 10, color: "#94A3B8", marginTop: 4 }}>Only player here. Novel IP position.</div>
            </div>
            <div style={{ background: "#0A0E18", borderRadius: "0 0 10px 0", padding: 14 }}>
              <div style={{ fontSize: 9, color: "#64748B", textTransform: "uppercase", letterSpacing: 1 }}>
                Correlational + Prototype
              </div>
              <div style={{ fontSize: 10, marginTop: 8 }}>
                <div style={{ color: "#F59E0B", fontWeight: 600 }}>TokaMind (MAST foundation)</div>
                <div style={{ color: "#94A3B8", marginTop: 2 }}>Many academic groups</div>
              </div>
            </div>
          </div>

          {/* Competitor cards */}
          {COMPETITORS.map((c, i) => (
            <div key={i} style={{
              background: "#0F172A", borderRadius: 12, padding: 14,
              border: "1px solid #1E293B", marginBottom: 8,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: "#F1F5F9" }}>{c.name}</span>
                <span style={{
                  fontSize: 9, padding: "2px 8px", borderRadius: 10,
                  background: c.threat === "HIGH" ? "#450A0A" : c.threat === "MEDIUM" ? "#422006" : "#052E16",
                  color: c.color, border: `1px solid ${c.color}44`,
                }}>{c.threat}</span>
              </div>
              <div style={{ fontSize: 10, color: "#94A3B8", marginTop: 4 }}>{c.approach}</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
                <div>
                  <div style={{ fontSize: 9, color: "#22C55E" }}>Their strength</div>
                  <div style={{ fontSize: 10, color: "#CBD5E1" }}>{c.strength}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: "#EF4444" }}>Their gap</div>
                  <div style={{ fontSize: 10, color: "#CBD5E1" }}>{c.weakness}</div>
                </div>
              </div>
            </div>
          ))}

          {/* Why FusionMind matters */}
          <div style={{
            marginTop: 8, background: "#0C1222", borderRadius: 12, padding: 16,
            border: "1px solid #1E3A5F",
          }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#60A5FA", marginBottom: 6 }}>
              Why Causal Matters (Even If RL Works)
            </div>
            <div style={{ fontSize: 11, color: "#CBD5E1", lineHeight: 1.6 }}>
              ITER and DEMO will require <strong>explainability</strong> for regulatory approval.
              "The RL said to increase coil current by 12%" is not acceptable for a nuclear facility.
              "The causal model shows that increasing coil current causes q95 to drop below safety
              threshold with 94% confidence" is what regulators need. FusionMind provides the causal
              reasoning layer that makes RL decisions <strong>interpretable, auditable, and safe</strong>.
            </div>
          </div>
        </div>
      )}

      {/* ═══ ROADMAP ═══ */}
      {tab === "roadmap" && (
        <div>
          {[
            {
              phase: "NOW", period: "Q1 2026", status: "complete",
              title: "Proof of Concept", color: "#3B82F6",
              items: [
                { done: true, text: "CPDE v3.2: F1=88.9% on synthetic, 11 edges on real MAST" },
                { done: true, text: "CPC v2.0: Counterfactual queries physically validated" },
                { done: true, text: "UPFM PoC: Dimensionless tokenization across 6 devices" },
                { done: true, text: "FAIR-MAST pipeline: 8 shots, 625 timepoints loaded via S3" },
                { done: true, text: "D3R PoC: 156:1 compression (score network needs U-Net)" },
                { done: false, text: "File PF1 + PF2 US Provisionals (URGENT — DeepMind threat)" },
              ],
            },
            {
              phase: "NEXT", period: "Q2–Q3 2026", status: "planned",
              title: "Technical De-risking", color: "#F59E0B",
              items: [
                { done: false, text: "Nonlinear SCM (neural SEM or additive noise models)" },
                { done: false, text: "TORAX integration for sim-to-real causal validation" },
                { done: false, text: "Scale MAST validation: 100+ shots, multi-campaign" },
                { done: false, text: "D3R with real U-Net score network (target <20% error)" },
                { done: false, text: "OMAS/PlasmaPy data format compliance" },
                { done: false, text: "File PF3–PF5 provisionals, PCT applications" },
              ],
            },
            {
              phase: "2027", period: "Q1–Q4 2027", status: "planned",
              title: "Real-World Validation", color: "#10B981",
              items: [
                { done: false, text: "Partnership with CFS/TAE/GA for DIII-D or TCV test" },
                { done: false, text: "Causal safety layer running alongside RL controller" },
                { done: false, text: "Real-time inference pipeline (<10ms for causal queries)" },
                { done: false, text: "First paper: 'Causal Discovery in Tokamak Plasmas'" },
                { done: false, text: "ITER-relevant demonstration on SPARC-like scenarios" },
              ],
            },
            {
              phase: "2028+", period: "2028 onwards", status: "vision",
              title: "Deployment & Licensing", color: "#A78BFA",
              items: [
                { done: false, text: "Licensing to CFS, TAE, ITER for causal safety compliance" },
                { done: false, text: "Regulatory framework contribution (causal explainability)" },
                { done: false, text: "Foundation model with causal pretraining across 10+ devices" },
                { done: false, text: "Target: standard causal safety layer for all fusion facilities" },
              ],
            },
          ].map((phase, pi) => (
            <div key={pi} style={{
              background: "#0F172A", borderRadius: 12, padding: 16,
              border: `1px solid ${phase.status === "complete" ? phase.color + "44" : "#1E293B"}`,
              marginBottom: 10, position: "relative",
            }}>
              {pi < 3 && (
                <div style={{
                  position: "absolute", bottom: -10, left: "50%", width: 2, height: 10,
                  background: "#1E293B",
                }} />
              )}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{
                    fontSize: 10, fontWeight: 800, color: "#020617", background: phase.color,
                    padding: "2px 8px", borderRadius: 4,
                  }}>{phase.phase}</span>
                  <span style={{ fontSize: 13, fontWeight: 700, color: "#F1F5F9" }}>{phase.title}</span>
                </div>
                <span style={{ fontSize: 10, color: "#64748B" }}>{phase.period}</span>
              </div>
              {phase.items.map((item, ii) => (
                <div key={ii} style={{
                  display: "flex", gap: 6, padding: "3px 0", fontSize: 11,
                  color: item.done ? "#CBD5E1" : "#64748B",
                }}>
                  <span style={{ color: item.done ? "#22C55E" : "#334155", flexShrink: 0 }}>
                    {item.done ? "✓" : "○"}
                  </span>
                  <span>{item.text}</span>
                </div>
              ))}
            </div>
          ))}

          {/* Investment thesis */}
          <div style={{
            marginTop: 12, background: "linear-gradient(135deg, #0C1222, #0A1628)",
            borderRadius: 12, padding: 16, border: "1px solid #1E3A5F",
          }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#60A5FA", marginBottom: 8 }}>
              Investment Thesis (Honest Version)
            </div>
            <div style={{ fontSize: 11, color: "#CBD5E1", lineHeight: 1.6 }}>
              FusionMind is not "better than DeepMind" today. DeepMind controlled real plasma in 2022.
              We have a prototype in 2026. But we have something they don't:{" "}
              <strong style={{ color: "#F1F5F9" }}>causal understanding of plasma physics</strong>.
            </div>
            <div style={{ fontSize: 11, color: "#CBD5E1", lineHeight: 1.6, marginTop: 8 }}>
              The path to value: (1) nonlinear SCM + TORAX integration in 2026, (2) test on DIII-D/TCV
              in 2027 as causal safety layer on top of existing RL, (3) if validated, becomes{" "}
              <strong style={{ color: "#10B981" }}>required infrastructure</strong> for ITER/DEMO
              regulatory compliance. That's the $280M+ scenario.
            </div>
            <div style={{
              fontSize: 10, color: "#F59E0B", marginTop: 10,
              borderTop: "1px solid #1E293B", paddingTop: 8,
            }}>
              Risk: If DeepMind adds causal reasoning to TORAX before we reach TRL 5,
              our patent position weakens significantly. Filing PF1+PF2 provisionals is
              time-critical.
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{
        textAlign: "center", marginTop: 24, paddingTop: 12,
        borderTop: "1px solid #1E293B",
      }}>
        <div style={{ fontSize: 9, color: "#334155" }}>
          FusionMind 4.0 · Dr. Mladen Mester · March 2026
        </div>
        <div style={{ fontSize: 8, color: "#1E293B", marginTop: 2 }}>
          Validated on FAIR-MAST (UKAEA) + Alcator C-Mod (MIT PSFC)
        </div>
      </div>
    </div>
  );
}
