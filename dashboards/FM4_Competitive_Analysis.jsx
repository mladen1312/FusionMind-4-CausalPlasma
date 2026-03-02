import { useState } from "react";

const COMPETITORS = [
  {
    id: "fm4",
    name: "FusionMind 4.0",
    org: "Mester (Ours)",
    year: "2026",
    color: "#22D3EE",
    accent: "#0E7490",
    type: "Causal AI + Foundation Model",
    approach: "Pearl's do-calculus, SCM counterfactuals, physics-constrained DAGs, dimensionless tokenization",
    devices: "6 (sim) + Alcator C-Mod (real)",
    causal: true,
    crossDevice: true,
    foundation: true,
    counterfactual: true,
    openSource: false,
    realTime: true,
    key_innovation: "First-ever causal inference for tokamak plasma — discovers WHY plasma behaves, not just correlations",
    limitations: "PoC stage, no on-device deployment yet",
    papers: "In preparation (6 patent families)",
    status: "PoC validated",
    threat: "none",
    scores: { novelty: 10, causal: 10, cross: 8, foundation: 8, realtime: 7, maturity: 4 },
  },
  {
    id: "tokamind",
    name: "TokaMind",
    org: "Boschi et al. (KDD 2026)",
    year: "Feb 2026",
    color: "#A78BFA",
    accent: "#6D28D9",
    type: "Multi-Modal Transformer FM",
    approach: "DCT3D tokenization, MMT architecture, warm-start fine-tuning, TokaMark benchmark",
    devices: "MAST only (single device)",
    causal: false,
    crossDevice: false,
    foundation: true,
    counterfactual: false,
    openSource: true,
    realTime: false,
    key_innovation: "First open-source FM for tokamak data with DCT3D compression and multi-modal support",
    limitations: "Single device (MAST), no causal reasoning, no control, DCT tokens lose physics meaning",
    papers: "arXiv:2602.15084",
    status: "Published, code coming",
    threat: "medium",
    scores: { novelty: 6, causal: 0, cross: 2, foundation: 8, realtime: 3, maturity: 6 },
  },
  {
    id: "deepmind",
    name: "DeepMind + CFS",
    org: "Google DeepMind",
    year: "2022–2026",
    color: "#FB923C",
    accent: "#C2410C",
    type: "RL Magnet Control + TORAX Sim",
    approach: "Deep RL for coil control, TORAX differentiable simulator (JAX), evolutionary search",
    devices: "TCV (EPFL), SPARC (CFS planned)",
    causal: false,
    crossDevice: false,
    foundation: false,
    counterfactual: false,
    openSource: true,
    realTime: true,
    key_innovation: "First RL controller deployed on real tokamak (TCV), TORAX differentiable simulator for CFS/SPARC",
    limitations: "Pure RL — no causal understanding, no cross-device transfer, single-task controllers",
    papers: "Nature 2022, TORAX 2024, CFS partnership Oct 2025",
    status: "Deployed (TCV), Active (SPARC)",
    threat: "high",
    scores: { novelty: 7, causal: 0, cross: 2, foundation: 3, realtime: 9, maturity: 9 },
  },
  {
    id: "kstar",
    name: "KSTAR/DIII-D RL",
    org: "Princeton / SNU / GA",
    year: "2024",
    color: "#F472B6",
    accent: "#BE185D",
    type: "RL Tearing Avoidance",
    approach: "Multimodal tearing prediction + RL obstacle avoidance, PACMAN framework",
    devices: "KSTAR, DIII-D",
    causal: false,
    crossDevice: false,
    foundation: false,
    counterfactual: false,
    openSource: false,
    realTime: true,
    key_innovation: "First AI to actively avoid tearing instabilities in real-time on DIII-D tokamak",
    limitations: "Correlational prediction, device-specific, no causal model of WHY tears occur",
    papers: "Nature 2024 (Seo et al.)",
    status: "Deployed (DIII-D)",
    threat: "medium",
    scores: { novelty: 7, causal: 0, cross: 1, foundation: 0, realtime: 9, maturity: 8 },
  },
  {
    id: "princeton",
    name: "Princeton PPPL",
    org: "PPPL / Kolemen group",
    year: "2023–2025",
    color: "#34D399",
    accent: "#047857",
    type: "Diagnostic ML + Reconstruction",
    approach: "Diag2Diag supervised learning, Kalman filtering, kinetic reconstruction, multimodal super-resolution",
    devices: "DIII-D, NSTX-U, KSTAR",
    causal: false,
    crossDevice: false,
    foundation: false,
    counterfactual: false,
    openSource: false,
    realTime: true,
    key_innovation: "Real-time kinetic profile reconstruction, multimodal super-resolution (Nature Comms 2025)",
    limitations: "Task-specific models, correlational, no unified framework, no causal inference",
    papers: "Nature Comms 2025, NF 2023–2025",
    status: "Deployed (DIII-D)",
    threat: "low",
    scores: { novelty: 5, causal: 0, cross: 3, foundation: 2, realtime: 8, maturity: 8 },
  },
  {
    id: "hl3",
    name: "HL-3 RL Control",
    org: "SWIP (China)",
    year: "2025",
    color: "#FBBF24",
    accent: "#A16207",
    type: "Data-driven RL Simulator",
    approach: "High-fidelity dynamics model as RL environment, Ip + shape control",
    devices: "HL-3",
    causal: false,
    crossDevice: false,
    foundation: false,
    counterfactual: false,
    openSource: false,
    realTime: true,
    key_innovation: "Data-driven dynamics model enabling 400ms stable RL control on HL-3",
    limitations: "Single device, correlational, no transfer capability",
    papers: "Comms Physics 2025",
    status: "Deployed (HL-3)",
    threat: "low",
    scores: { novelty: 4, causal: 0, cross: 1, foundation: 0, realtime: 7, maturity: 7 },
  },
];

const DIMENSIONS = [
  { key: "novelty", label: "Novelty", max: 10 },
  { key: "causal", label: "Causal", max: 10 },
  { key: "cross", label: "Cross-Device", max: 10 },
  { key: "foundation", label: "Foundation Model", max: 10 },
  { key: "realtime", label: "Real-Time Ready", max: 10 },
  { key: "maturity", label: "Maturity", max: 10 },
];

const CAP_MAP = {
  causal: { yes: "Causal Inference", no: "Correlational Only" },
  crossDevice: { yes: "Cross-Device", no: "Single Device" },
  foundation: { yes: "Foundation Model", no: "Task-Specific" },
  counterfactual: { yes: "Counterfactual", no: "No Counterfactual" },
  openSource: { yes: "Open Source", no: "Proprietary" },
  realTime: { yes: "Real-Time", no: "Offline" },
};

function RadarChart({ competitors, selected }) {
  const cx = 150, cy = 150, R = 120;
  const n = DIMENSIONS.length;
  const angleStep = (2 * Math.PI) / n;

  const gridLevels = [0.25, 0.5, 0.75, 1.0];
  const axisPoints = DIMENSIONS.map((_, i) => {
    const angle = -Math.PI / 2 + i * angleStep;
    return { x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle), angle };
  });

  return (
    <svg viewBox="0 0 300 300" style={{ width: "100%", maxWidth: 320 }}>
      {gridLevels.map((lev) => (
        <polygon
          key={lev}
          points={axisPoints.map((p) => {
            const dx = (p.x - cx) * lev;
            const dy = (p.y - cy) * lev;
            return `${cx + dx},${cy + dy}`;
          }).join(" ")}
          fill="none"
          stroke="#1E293B"
          strokeWidth="0.5"
        />
      ))}
      {axisPoints.map((p, i) => (
        <g key={i}>
          <line x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="#1E293B" strokeWidth="0.5" />
          <text
            x={cx + (p.x - cx) * 1.18}
            y={cy + (p.y - cy) * 1.18}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#6B7280"
            fontSize="8"
            fontFamily="'JetBrains Mono', monospace"
          >
            {DIMENSIONS[i].label}
          </text>
        </g>
      ))}
      {competitors
        .filter((c) => selected.includes(c.id))
        .map((comp) => {
          const pts = DIMENSIONS.map((dim, i) => {
            const val = comp.scores[dim.key] / dim.max;
            const angle = -Math.PI / 2 + i * angleStep;
            return `${cx + R * val * Math.cos(angle)},${cy + R * val * Math.sin(angle)}`;
          }).join(" ");
          return (
            <polygon
              key={comp.id}
              points={pts}
              fill={comp.color + "20"}
              stroke={comp.color}
              strokeWidth={comp.id === "fm4" ? 2.5 : 1.5}
              opacity={comp.id === "fm4" ? 1 : 0.7}
            />
          );
        })}
    </svg>
  );
}

function CapabilityMatrix({ competitors }) {
  const caps = Object.keys(CAP_MAP);
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: "6px 8px", borderBottom: "1px solid #1F2937", color: "#6B7280" }}>System</th>
            {caps.map((c) => (
              <th key={c} style={{ padding: "6px 4px", borderBottom: "1px solid #1F2937", color: "#6B7280", textAlign: "center", fontSize: 9 }}>
                {CAP_MAP[c].yes}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {competitors.map((comp) => (
            <tr key={comp.id} style={{ background: comp.id === "fm4" ? "#0E293B" : "transparent" }}>
              <td style={{ padding: "6px 8px", borderBottom: "1px solid #111827", color: comp.color, fontWeight: 700 }}>
                {comp.name}
              </td>
              {caps.map((c) => (
                <td key={c} style={{ textAlign: "center", padding: "6px 4px", borderBottom: "1px solid #111827" }}>
                  {comp[c] ? (
                    <span style={{ color: "#22C55E", fontWeight: 800 }}>✓</span>
                  ) : (
                    <span style={{ color: "#4B5563" }}>✗</span>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function GapBar({ label, fm4, best_competitor, best_name }) {
  const barW = 200;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 3 }}>
        <span style={{ color: "#9CA3AF" }}>{label}</span>
        <span style={{ color: fm4 > best_competitor ? "#22C55E" : "#EF4444", fontWeight: 700 }}>
          {fm4 > best_competitor ? `+${fm4 - best_competitor}` : `${fm4 - best_competitor}`} vs {best_name}
        </span>
      </div>
      <div style={{ position: "relative", height: 14, background: "#111827", borderRadius: 4 }}>
        <div style={{ position: "absolute", height: "100%", width: `${(best_competitor / 10) * 100}%`, background: "#374151", borderRadius: 4 }} />
        <div style={{ position: "absolute", height: "100%", width: `${(fm4 / 10) * 100}%`, background: fm4 > best_competitor ? "#22D3EE33" : "#EF444433", borderRadius: 4, border: `1px solid ${fm4 > best_competitor ? "#22D3EE" : "#EF4444"}` }} />
      </div>
    </div>
  );
}

export default function CompetitiveAnalysis() {
  const [view, setView] = useState("overview");
  const [selected, setSelected] = useState(["fm4", "tokamind", "deepmind"]);
  const [expandedCard, setExpandedCard] = useState(null);

  const toggle = (id) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const fm4 = COMPETITORS[0];

  return (
    <div style={{
      background: "#070B14",
      minHeight: "100vh",
      color: "#D1D5DB",
      fontFamily: "'JetBrains Mono', 'Fira Code', ui-monospace, monospace",
      padding: 16,
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <div style={{ fontSize: 9, letterSpacing: 5, color: "#EF4444", marginBottom: 4 }}>
          COMPETITIVE INTELLIGENCE — MARCH 2026
        </div>
        <h1 style={{
          fontSize: 20,
          fontWeight: 900,
          background: "linear-gradient(135deg, #22D3EE, #A78BFA, #FB923C)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          margin: 0,
        }}>
          FUSION AI LANDSCAPE
        </h1>
        <div style={{ color: "#4B5563", fontSize: 10, marginTop: 4 }}>
          FusionMind 4.0 vs. Global Competition · Updated with TokaMind (Feb 2026)
        </div>
      </div>

      {/* Key Finding Banner */}
      <div style={{
        background: "linear-gradient(135deg, #0E293B, #1A1040)",
        border: "1px solid #22D3EE44",
        borderRadius: 10,
        padding: 14,
        marginBottom: 16,
        textAlign: "center",
      }}>
        <div style={{ fontSize: 11, fontWeight: 800, color: "#22D3EE", marginBottom: 4 }}>
          KEY FINDING: NO COMPETITOR USES CAUSAL INFERENCE
        </div>
        <div style={{ fontSize: 10, color: "#9CA3AF", lineHeight: 1.6 }}>
          All 5 major competitors remain <span style={{ color: "#FB923C", fontWeight: 700 }}>purely correlational</span>.
          TokaMind (Feb 2026) confirms the gap — uses DCT tokens, not dimensionless physics.
          DeepMind/CFS partnership focuses on <span style={{ color: "#FB923C", fontWeight: 700 }}>RL + simulation</span>, no causal graphs.
          FusionMind 4.0 remains the <span style={{ color: "#22C55E", fontWeight: 700 }}>only causal approach</span> globally.
        </div>
      </div>

      {/* Navigation */}
      <div style={{ display: "flex", justifyContent: "center", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
        {[
          ["overview", "◉ Overview"],
          ["radar", "⬡ Radar"],
          ["matrix", "⊞ Capability Matrix"],
          ["threats", "⚠ Threat Assessment"],
          ["detail", "⊛ Deep Dive"],
        ].map(([k, label]) => (
          <button
            key={k}
            onClick={() => setView(k)}
            style={{
              padding: "5px 14px",
              borderRadius: 16,
              border: "none",
              fontSize: 10,
              fontWeight: 600,
              cursor: "pointer",
              fontFamily: "inherit",
              background: view === k ? "#22D3EE" : "#1F2937",
              color: view === k ? "#000" : "#9CA3AF",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* OVERVIEW */}
      {view === "overview" && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 10 }}>
            {COMPETITORS.map((c) => (
              <div
                key={c.id}
                onClick={() => setExpandedCard(expandedCard === c.id ? null : c.id)}
                style={{
                  background: c.id === "fm4" ? "#0A1628" : "#0F1629",
                  border: `1px solid ${c.id === "fm4" ? c.color + "66" : "#1F2937"}`,
                  borderRadius: 10,
                  padding: 14,
                  cursor: "pointer",
                  transition: "border-color 0.2s",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 800, color: c.color }}>{c.name}</div>
                    <div style={{ fontSize: 9, color: "#6B7280" }}>{c.org} · {c.year}</div>
                  </div>
                  <div style={{
                    padding: "2px 8px",
                    borderRadius: 8,
                    fontSize: 8,
                    fontWeight: 700,
                    background: c.threat === "none" ? "#22C55E22" : c.threat === "high" ? "#EF444422" : c.threat === "medium" ? "#F59E0B22" : "#6B728022",
                    color: c.threat === "none" ? "#22C55E" : c.threat === "high" ? "#EF4444" : c.threat === "medium" ? "#F59E0B" : "#6B7280",
                    border: `1px solid ${c.threat === "none" ? "#22C55E44" : c.threat === "high" ? "#EF444444" : c.threat === "medium" ? "#F59E0B44" : "#6B728044"}`,
                  }}>
                    {c.threat === "none" ? "US" : c.threat.toUpperCase()} THREAT
                  </div>
                </div>

                <div style={{ fontSize: 9, color: "#9CA3AF", lineHeight: 1.5, marginBottom: 8 }}>
                  <span style={{ color: c.color, fontWeight: 600 }}>{c.type}</span>
                </div>

                <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 8 }}>
                  {Object.entries(CAP_MAP).map(([k, v]) => (
                    <span key={k} style={{
                      padding: "1px 6px",
                      borderRadius: 6,
                      fontSize: 8,
                      background: c[k] ? "#22C55E15" : "#1F2937",
                      color: c[k] ? "#22C55E" : "#374151",
                      border: `1px solid ${c[k] ? "#22C55E33" : "#1F293700"}`,
                    }}>
                      {c[k] ? "✓" : "✗"} {v.yes.split(" ")[0]}
                    </span>
                  ))}
                </div>

                <div style={{ fontSize: 9, color: "#6B7280", lineHeight: 1.5 }}>
                  {c.key_innovation}
                </div>

                {expandedCard === c.id && (
                  <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid #1F2937" }}>
                    <div style={{ fontSize: 9, lineHeight: 1.6 }}>
                      <div><span style={{ color: "#6B7280" }}>Approach:</span> <span style={{ color: "#D1D5DB" }}>{c.approach}</span></div>
                      <div style={{ marginTop: 4 }}><span style={{ color: "#6B7280" }}>Devices:</span> <span style={{ color: "#D1D5DB" }}>{c.devices}</span></div>
                      <div style={{ marginTop: 4 }}><span style={{ color: "#6B7280" }}>Papers:</span> <span style={{ color: "#D1D5DB" }}>{c.papers}</span></div>
                      <div style={{ marginTop: 4 }}><span style={{ color: "#EF4444" }}>Limitations:</span> <span style={{ color: "#D1D5DB" }}>{c.limitations}</span></div>
                      <div style={{ marginTop: 4 }}><span style={{ color: "#6B7280" }}>Status:</span> <span style={{ color: c.color }}>{c.status}</span></div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* RADAR */}
      {view === "radar" && (
        <div style={{ maxWidth: 500, margin: "0 auto" }}>
          <div style={{ display: "flex", gap: 4, justifyContent: "center", flexWrap: "wrap", marginBottom: 12 }}>
            {COMPETITORS.map((c) => (
              <button
                key={c.id}
                onClick={() => toggle(c.id)}
                style={{
                  padding: "3px 10px",
                  borderRadius: 10,
                  border: `1px solid ${selected.includes(c.id) ? c.color : "#374151"}`,
                  background: selected.includes(c.id) ? c.color + "20" : "transparent",
                  color: selected.includes(c.id) ? c.color : "#6B7280",
                  fontSize: 9,
                  cursor: "pointer",
                  fontFamily: "inherit",
                  fontWeight: selected.includes(c.id) ? 700 : 400,
                }}
              >
                {c.name}
              </button>
            ))}
          </div>
          <RadarChart competitors={COMPETITORS} selected={selected} />
          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap", marginTop: 8 }}>
            {COMPETITORS.filter((c) => selected.includes(c.id)).map((c) => (
              <div key={c.id} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 9 }}>
                <div style={{ width: 8, height: 8, borderRadius: 4, background: c.color }} />
                <span style={{ color: c.color }}>{c.name}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* CAPABILITY MATRIX */}
      {view === "matrix" && (
        <div style={{ maxWidth: 600, margin: "0 auto" }}>
          <CapabilityMatrix competitors={COMPETITORS} />
          <div style={{
            marginTop: 16,
            background: "#0A1628",
            border: "1px solid #22D3EE33",
            borderRadius: 10,
            padding: 14,
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#22D3EE", marginBottom: 8 }}>
              UNIQUE CAPABILITIES — FM4 ONLY
            </div>
            {[
              "Causal discovery via physics-constrained DAGs (Pearl's do-calculus)",
              "Counterfactual reasoning: P(Y | do(X)) for what-if scenarios",
              "Dimensionless tokenization (βN, ν*, ρ*, q95, H98) preserving physics",
              "Cross-device transfer via dimensionless physics — not data-driven embeddings",
              "Active Experiment Design — causality-driven experimental optimization",
              "Simpson's Paradox detection in real tokamak data (validated C-Mod)",
            ].map((item, i) => (
              <div key={i} style={{ display: "flex", gap: 6, alignItems: "flex-start", marginBottom: 4, fontSize: 10 }}>
                <span style={{ color: "#22C55E", flexShrink: 0 }}>◆</span>
                <span style={{ color: "#D1D5DB" }}>{item}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* THREAT ASSESSMENT */}
      {view === "threats" && (
        <div style={{ maxWidth: 540, margin: "0 auto" }}>
          {/* Gap Analysis */}
          <div style={{ background: "#0F1629", borderRadius: 10, padding: 14, marginBottom: 12, border: "1px solid #1F2937" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#9CA3AF", marginBottom: 10 }}>
              FM4 GAP ANALYSIS vs BEST COMPETITOR
            </div>
            {DIMENSIONS.map((dim) => {
              const others = COMPETITORS.filter((c) => c.id !== "fm4");
              const best = others.reduce((a, b) => (b.scores[dim.key] > a.scores[dim.key] ? b : a));
              return (
                <GapBar
                  key={dim.key}
                  label={dim.label}
                  fm4={fm4.scores[dim.key]}
                  best_competitor={best.scores[dim.key]}
                  best_name={best.name}
                />
              );
            })}
          </div>

          {/* TokaMind Deep Analysis */}
          <div style={{
            background: "#0F1629",
            borderRadius: 10,
            padding: 14,
            marginBottom: 12,
            border: "1px solid #A78BFA33",
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#A78BFA", marginBottom: 8 }}>
              ⚡ NEW THREAT: TokaMind (Feb 16, 2026)
            </div>
            <div style={{ fontSize: 10, lineHeight: 1.7 }}>
              <div style={{ marginBottom: 8 }}>
                <span style={{ color: "#EF4444", fontWeight: 700 }}>What it does:</span>{" "}
                <span style={{ color: "#D1D5DB" }}>
                  Multi-Modal Transformer FM for MAST tokamak. Uses DCT3D training-free tokenization, supports time-series + 2D profiles + videos. Beats CNN baseline on 13/14 TokaMark tasks.
                </span>
              </div>
              <div style={{ marginBottom: 8 }}>
                <span style={{ color: "#22C55E", fontWeight: 700 }}>Why FM4 is safe:</span>
                {[
                  "DCT tokens = mathematical compression, NOT physics-meaningful → our dimensionless tokens preserve scaling laws",
                  "Single device (MAST) — no cross-device demonstrated",
                  "Zero causal reasoning — purely correlational reconstruction/prediction",
                  "No control capability — reconstruction only",
                  "No counterfactual queries possible",
                ].map((item, i) => (
                  <div key={i} style={{ display: "flex", gap: 6, marginTop: 3, fontSize: 9 }}>
                    <span style={{ color: "#22C55E", flexShrink: 0 }}>✓</span>
                    <span style={{ color: "#9CA3AF" }}>{item}</span>
                  </div>
                ))}
              </div>
              <div>
                <span style={{ color: "#F59E0B", fontWeight: 700 }}>Action items:</span>
                {[
                  "Cite TokaMind in UPFM patent as prior art — differentiate dimensionless vs DCT tokens",
                  "Add TokaMark benchmark comparison to UPFM evaluation when code released",
                  "Emphasize cross-device transfer (our 6 devices vs their 1) in patent claims",
                ].map((item, i) => (
                  <div key={i} style={{ display: "flex", gap: 6, marginTop: 3, fontSize: 9 }}>
                    <span style={{ color: "#F59E0B", flexShrink: 0 }}>→</span>
                    <span style={{ color: "#D1D5DB" }}>{item}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* DeepMind/CFS Threat */}
          <div style={{
            background: "#0F1629",
            borderRadius: 10,
            padding: 14,
            marginBottom: 12,
            border: "1px solid #FB923C33",
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#FB923C", marginBottom: 8 }}>
              🔴 HIGH THREAT: DeepMind + CFS (Oct 2025)
            </div>
            <div style={{ fontSize: 10, lineHeight: 1.7 }}>
              <div style={{ marginBottom: 8 }}>
                <span style={{ color: "#EF4444", fontWeight: 700 }}>Danger:</span>{" "}
                <span style={{ color: "#D1D5DB" }}>
                  TORAX differentiable simulator integrated with SPARC. RL controllers for coil control + evolutionary search. Google-scale compute. CFS aims for net energy late 2026/early 2027.
                </span>
              </div>
              <div style={{ marginBottom: 8 }}>
                <span style={{ color: "#22C55E", fontWeight: 700 }}>FM4 differentiation:</span>
                {[
                  "DeepMind = RL blackbox (learns what works, not why) → FM4 = causal understanding",
                  "TORAX = forward simulation → FM4 = counterfactual (what would happen if...)",
                  "No cross-device transfer in DeepMind approach — SPARC-specific",
                  "If DeepMind adds causal layer later, our patents block them (file ASAP!)",
                ].map((item, i) => (
                  <div key={i} style={{ display: "flex", gap: 6, marginTop: 3, fontSize: 9 }}>
                    <span style={{ color: "#22C55E", flexShrink: 0 }}>✓</span>
                    <span style={{ color: "#9CA3AF" }}>{item}</span>
                  </div>
                ))}
              </div>
              <div style={{
                background: "#7F1D1D22",
                border: "1px solid #EF444444",
                borderRadius: 6,
                padding: 8,
                fontSize: 9,
                color: "#FCA5A5",
                fontWeight: 600,
              }}>
                ⚠ URGENCY: File PF1+PF2 provisionals before DeepMind publishes causal work. Their CFS partnership gives them access to real data that could enable similar discoveries.
              </div>
            </div>
          </div>

          {/* Timeline */}
          <div style={{
            background: "#0F1629",
            borderRadius: 10,
            padding: 14,
            border: "1px solid #1F2937",
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#9CA3AF", marginBottom: 8 }}>
              PATENT FILING URGENCY TIMELINE
            </div>
            {[
              { date: "Now", event: "File PF1 (CPDE) + PF2 (CPC) provisionals", color: "#EF4444", urgent: true },
              { date: "Q2 2026", event: "File PF3 (UPFM) — TokaMind published, differentiate", color: "#F59E0B", urgent: true },
              { date: "Q3 2026", event: "DeepMind likely publishes SPARC-specific results", color: "#FB923C", urgent: false },
              { date: "Q4 2026", event: "File PF4-PF6, PCT applications", color: "#22D3EE", urgent: false },
              { date: "2027", event: "SPARC achieves net energy — field explodes", color: "#A78BFA", urgent: false },
            ].map((item, i) => (
              <div key={i} style={{
                display: "flex",
                gap: 10,
                alignItems: "center",
                padding: "6px 0",
                borderBottom: i < 4 ? "1px solid #111827" : "none",
              }}>
                <div style={{
                  width: 8, height: 8, borderRadius: 4,
                  background: item.color,
                  boxShadow: item.urgent ? `0 0 8px ${item.color}` : "none",
                  flexShrink: 0,
                }} />
                <div style={{ fontSize: 9, color: item.color, fontWeight: 700, minWidth: 60 }}>
                  {item.date}
                </div>
                <div style={{ fontSize: 10, color: "#D1D5DB" }}>
                  {item.event}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* DEEP DIVE */}
      {view === "detail" && (
        <div style={{ maxWidth: 600, margin: "0 auto" }}>
          <div style={{
            background: "#0F1629",
            borderRadius: 10,
            padding: 14,
            marginBottom: 12,
            border: "1px solid #1F2937",
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#9CA3AF", marginBottom: 10 }}>
              TOKENIZATION COMPARISON: FM4 vs TokaMind
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div style={{ background: "#0A1628", borderRadius: 8, padding: 10, border: "1px solid #22D3EE33" }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#22D3EE", marginBottom: 6 }}>FM4: Dimensionless Tokens</div>
                <div style={{ fontSize: 9, color: "#9CA3AF", lineHeight: 1.6 }}>
                  <div>βN = nkT/(B²/2μ₀) → pressure/magnetic</div>
                  <div>ν* = collision frequency ratio</div>
                  <div>ρ* = Larmor radius / minor radius</div>
                  <div>q95 = edge safety factor</div>
                  <div>H98 = confinement quality</div>
                  <div style={{ color: "#22C55E", marginTop: 4, fontWeight: 600 }}>✓ Physics-preserving</div>
                  <div style={{ color: "#22C55E", fontWeight: 600 }}>✓ Cross-device by construction</div>
                  <div style={{ color: "#22C55E", fontWeight: 600 }}>✓ Validated CV=0.267 across 6 devices</div>
                </div>
              </div>
              <div style={{ background: "#1A0A30", borderRadius: 8, padding: 10, border: "1px solid #A78BFA33" }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#A78BFA", marginBottom: 6 }}>TokaMind: DCT3D Tokens</div>
                <div style={{ fontSize: 9, color: "#9CA3AF", lineHeight: 1.6 }}>
                  <div>DCT(signal) → frequency coefficients</div>
                  <div>Training-free compression</div>
                  <div>Configurable retention (H̃,W̃,T̃)</div>
                  <div>Explained variance ~0.95+</div>
                  <div>&lt;10M parameters</div>
                  <div style={{ color: "#EF4444", marginTop: 4, fontWeight: 600 }}>✗ No physics meaning</div>
                  <div style={{ color: "#EF4444", fontWeight: 600 }}>✗ Device-specific (MAST only)</div>
                  <div style={{ color: "#F59E0B", fontWeight: 600 }}>◐ Good compression but lossy</div>
                </div>
              </div>
            </div>
          </div>

          <div style={{
            background: "#0F1629",
            borderRadius: 10,
            padding: 14,
            marginBottom: 12,
            border: "1px solid #1F2937",
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#9CA3AF", marginBottom: 10 }}>
              PARADIGM MAP: CORRELATION → CAUSATION
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {[
                { level: "Correlational Prediction", desc: "What happens next?", systems: ["DeepMind RL", "KSTAR", "Princeton", "HL-3", "TokaMind"], color: "#EF4444" },
                { level: "Differentiable Simulation", desc: "How does the system evolve?", systems: ["TORAX (DeepMind)"], color: "#F59E0B" },
                { level: "Causal Discovery", desc: "What causes what?", systems: ["FusionMind 4.0 (CPDE)"], color: "#22C55E" },
                { level: "Counterfactual Reasoning", desc: "What would happen if...?", systems: ["FusionMind 4.0 (CPC)"], color: "#22D3EE" },
              ].map((row, i) => (
                <div key={i} style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "8px 10px",
                  background: i >= 2 ? "#0A162866" : "#11182766",
                  borderRadius: 6,
                  border: `1px solid ${row.color}22`,
                }}>
                  <div style={{
                    width: 30,
                    height: 30,
                    borderRadius: 6,
                    background: row.color + "22",
                    border: `1px solid ${row.color}44`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 12,
                    fontWeight: 900,
                    color: row.color,
                    flexShrink: 0,
                  }}>
                    L{i + 1}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: row.color }}>{row.level}</div>
                    <div style={{ fontSize: 9, color: "#6B7280" }}>{row.desc}</div>
                  </div>
                  <div style={{ display: "flex", gap: 3, flexWrap: "wrap", maxWidth: 180 }}>
                    {row.systems.map((s) => (
                      <span key={s} style={{
                        padding: "1px 6px",
                        borderRadius: 4,
                        fontSize: 8,
                        background: s.includes("FusionMind") ? "#22D3EE15" : "#1F2937",
                        color: s.includes("FusionMind") ? "#22D3EE" : "#6B7280",
                        border: s.includes("FusionMind") ? "1px solid #22D3EE44" : "1px solid transparent",
                        fontWeight: s.includes("FusionMind") ? 700 : 400,
                      }}>
                        {s}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Summary Box */}
          <div style={{
            background: "linear-gradient(135deg, #0E293B, #1A0A30)",
            border: "1px solid #22D3EE44",
            borderRadius: 10,
            padding: 14,
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#22D3EE", marginBottom: 8 }}>
              BOTTOM LINE
            </div>
            <div style={{ fontSize: 10, color: "#D1D5DB", lineHeight: 1.7 }}>
              FusionMind 4.0 operates at <span style={{ color: "#22C55E", fontWeight: 700 }}>Pearl's Ladder Level 2-3</span> (intervention + counterfactual) while all competitors remain at <span style={{ color: "#EF4444", fontWeight: 700 }}>Level 1</span> (association/prediction). TokaMind's Feb 2026 publication validates our thesis — even the newest FM uses mathematical tokenization, not physics-meaningful tokens. The DeepMind/CFS partnership is the highest threat due to resources, but their approach is fundamentally different (RL optimization vs. causal understanding). <span style={{ color: "#F59E0B", fontWeight: 700 }}>Priority: file PF1+PF2 provisionals immediately.</span>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{ textAlign: "center", marginTop: 20, padding: 10, borderTop: "1px solid #1F2937" }}>
        <div style={{ color: "#374151", fontSize: 8 }}>
          FusionMind 4.0 Competitive Intelligence · Dr. Mladen Mester · March 2026 · CONFIDENTIAL
        </div>
      </div>
    </div>
  );
}
