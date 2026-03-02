import { useState, useRef, useEffect, useCallback } from "react";

// ── Causal Graph Data (from CPDE v3.2 on real MAST data) ──
const CAUSAL_EDGES = [
  { from: "I_p", to: "q95", weight: -0.85, mechanism: "q95 ∝ B·a²/(R·Ip)" },
  { from: "I_p", to: "W_MHD", weight: 0.45, mechanism: "Better confinement" },
  { from: "I_p", to: "li", weight: 0.40, mechanism: "Current peaking" },
  { from: "P_NBI", to: "T_e", weight: 0.72, mechanism: "Direct heating" },
  { from: "P_NBI", to: "T_i", weight: 0.68, mechanism: "Ion heating via collisions" },
  { from: "P_NBI", to: "W_MHD", weight: 0.81, mechanism: "Power balance" },
  { from: "P_NBI", to: "beta_N", weight: 0.65, mechanism: "βN ∝ W/(I·B·a)" },
  { from: "P_ECRH", to: "T_e", weight: 0.55, mechanism: "Electron cyclotron resonance" },
  { from: "P_ECRH", to: "q95", weight: 0.20, mechanism: "Current drive → q modification" },
  { from: "gas_puff", to: "n_e", weight: 0.90, mechanism: "Direct fueling" },
  { from: "n_e", to: "P_rad", weight: 0.78, mechanism: "Prad ∝ ne²·Lz(Te)" },
  { from: "n_e", to: "T_e", weight: -0.35, mechanism: "Density dilution at fixed power" },
  { from: "T_e", to: "beta_N", weight: 0.42, mechanism: "β ∝ nT/B²" },
  { from: "kappa", to: "W_MHD", weight: 0.30, mechanism: "Shaping improves confinement" },
  { from: "li", to: "q95", weight: -0.25, mechanism: "Peaked current → lower edge q" },
];

const VARIABLES = {
  I_p: { label: "Plasma Current", unit: "MA", value: 0.8, type: "actuator", color: "#f59e0b" },
  P_NBI: { label: "NBI Power", unit: "MW", value: 5.2, type: "actuator", color: "#f59e0b" },
  P_ECRH: { label: "ECRH Power", unit: "MW", value: 2.0, type: "actuator", color: "#f59e0b" },
  gas_puff: { label: "Gas Puff", unit: "10²⁰/s", value: 3.1, type: "actuator", color: "#f59e0b" },
  n_e: { label: "Density", unit: "10¹⁹ m⁻³", value: 4.5, type: "state", color: "#3b82f6" },
  T_e: { label: "Electron Temp", unit: "keV", value: 5.2, type: "state", color: "#3b82f6" },
  T_i: { label: "Ion Temp", unit: "keV", value: 4.8, type: "state", color: "#3b82f6" },
  beta_N: { label: "βN", unit: "", value: 2.1, type: "state", color: "#3b82f6" },
  q95: { label: "Safety Factor", unit: "", value: 3.5, type: "state", color: "#3b82f6" },
  W_MHD: { label: "Stored Energy", unit: "MJ", value: 1.2, type: "state", color: "#3b82f6" },
  P_rad: { label: "Radiated Power", unit: "MW", value: 0.8, type: "diagnostic", color: "#8b5cf6" },
  kappa: { label: "Elongation", unit: "", value: 1.7, type: "shape", color: "#06b6d4" },
  li: { label: "Internal Inductance", unit: "", value: 0.95, type: "diagnostic", color: "#8b5cf6" },
};

const SCM_EQUATIONS = {
  T_e: "T_e = 0.72·P_NBI + 0.55·P_ECRH − 0.35·n_e + ε",
  T_i: "T_i = 0.68·P_NBI + ε",
  q95: "q95 = −0.85·I_p + 0.20·P_ECRH − 0.25·li + ε",
  W_MHD: "W_MHD = 0.45·I_p + 0.81·P_NBI + 0.30·κ + ε",
  beta_N: "βN = 0.65·P_NBI + 0.42·T_e + ε",
  n_e: "n_e = 0.90·gas_puff + ε",
  P_rad: "P_rad = 0.78·n_e + ε",
};

const EXAMPLE_QUERIES = [
  { q: "What happens to Te if we increase P_NBI to 8 MW?", level: 2, cat: "Intervention" },
  { q: "Why did stored energy drop?", level: "E", cat: "Explanation" },
  { q: "Would Te have been higher if we used ECRH instead of NBI?", level: 3, cat: "Counterfactual" },
  { q: "Is there a disruption risk at current beta_N?", level: "S", cat: "Safety" },
  { q: "What new causal relationships should we test?", level: "H", cat: "Hypothesis" },
  { q: "What are the confounders between density and temperature?", level: 1, cat: "Observation" },
  { q: "If we reduce gas puff, what is the causal effect on radiation?", level: 2, cat: "Intervention" },
  { q: "What would have happened if we had 1.2 MA instead of 0.8 MA?", level: 3, cat: "Counterfactual" },
];

// ── Build system prompt ──
function buildSystemPrompt() {
  const edgeLines = CAUSAL_EDGES.map(e => {
    const sign = e.weight > 0 ? "+" : "−";
    return `  ${e.from} →(${sign}${Math.abs(e.weight).toFixed(2)}) ${e.to}  [${e.mechanism}]`;
  }).join("\n");

  const eqLines = Object.entries(SCM_EQUATIONS).map(([k,v]) => `  ${v}`).join("\n");

  const stateLines = Object.entries(VARIABLES).map(([k,v]) =>
    `  ${k} = ${v.value} ${v.unit} (${v.type})`
  ).join("\n");

  return `You are FusionMind Causal Copilot — an AI assistant for tokamak plasma control that reasons using Pearl's causal inference framework.

You have access to a discovered CAUSAL GRAPH (DAG) from real MAST tokamak data (F1=91.9%), structural causal model (SCM) equations, and the current plasma state.

CAUSAL GRAPH (Discovered by CPDE v3.2 — 15 edges from real MAST + FM3Lite data):
${edgeLines}

STRUCTURAL CAUSAL MODEL (SCM equations):
${eqLines}

CURRENT PLASMA STATE:
${stateLines}

SAFETY LIMITS:
  βN: max 4.0, warning 3.5
  q95: min 1.5, warning 2.0
  n_e: max Greenwald limit
  P_rad fraction: max 0.8

KEY RESULTS:
  CPDE F1 on real MAST data: 91.9% (8 shots, 625 timepoints)
  Simpson's Paradox detected on Alcator C-Mod: density-disruption ρ drops +0.53→+0.02 when conditioning on Ip
  Cross-device transfer validated on 6 tokamaks (CV=0.267)

REASONING PROTOCOL:
1. IDENTIFY query type: Observation (Level 1), Intervention (Level 2), or Counterfactual (Level 3).
2. TRACE causal paths in the DAG.
3. CHECK for confounders — warn if correlation ≠ causation (Simpson's Paradox risk).
4. USE SCM equations for quantitative estimates.
5. VERIFY against safety limits.
6. Provide confidence level and reasoning.

For interventional queries: Use do-calculus notation P(Y|do(X=x)).
For counterfactuals: Use abduction→action→prediction.
Keep answers concise (2-4 paragraphs). Use the causal graph edges and SCM equations for specific numbers.`;
}

// ── Pearl Level Badge ──
function LevelBadge({ level }) {
  const config = {
    1: { label: "L1 · Observation", bg: "#1e3a5f", border: "#3b82f6" },
    2: { label: "L2 · Intervention", bg: "#3b1f0b", border: "#f59e0b" },
    3: { label: "L3 · Counterfactual", bg: "#3b0764", border: "#a855f7" },
    E: { label: "Explanation", bg: "#1a2e1a", border: "#22c55e" },
    H: { label: "Hypothesis", bg: "#2d1b2e", border: "#ec4899" },
    S: { label: "Safety", bg: "#3b0e0e", border: "#ef4444" },
  };
  const c = config[level] || config[1];
  return (
    <span style={{
      display: "inline-block", padding: "2px 10px", borderRadius: 4,
      fontSize: 11, fontWeight: 600, letterSpacing: 0.5,
      background: c.bg, border: `1px solid ${c.border}`, color: c.border,
    }}>
      {c.label}
    </span>
  );
}

// ── Causal Graph Visualization (SVG) ──
function CausalGraphSVG({ highlightEdge }) {
  const positions = {
    I_p: [80, 50], P_NBI: [220, 50], P_ECRH: [360, 50], gas_puff: [500, 50],
    n_e: [500, 160], T_e: [290, 160], T_i: [150, 160],
    q95: [80, 270], W_MHD: [220, 270], beta_N: [360, 270],
    P_rad: [500, 270], kappa: [80, 160], li: [150, 270],
  };

  return (
    <svg viewBox="0 0 600 330" style={{ width: "100%", height: "auto" }}>
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#475569" />
        </marker>
        <marker id="arrow-hl" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#f59e0b" />
        </marker>
      </defs>
      {CAUSAL_EDGES.map((e, i) => {
        const [x1, y1] = positions[e.from] || [0,0];
        const [x2, y2] = positions[e.to] || [0,0];
        const hl = highlightEdge && (highlightEdge.from === e.from && highlightEdge.to === e.to);
        const dx = x2-x1, dy = y2-y1;
        const len = Math.sqrt(dx*dx+dy*dy);
        const nx = dx/len, ny = dy/len;
        const sx = x1+nx*28, sy = y1+ny*28;
        const ex = x2-nx*28, ey = y2-ny*28;
        const mx = (sx+ex)/2 + ny*15, my = (sy+ey)/2 - nx*15;
        return (
          <g key={i}>
            <path
              d={`M ${sx} ${sy} Q ${mx} ${my} ${ex} ${ey}`}
              fill="none"
              stroke={hl ? "#f59e0b" : (e.weight > 0 ? "#334155" : "#7f1d1d")}
              strokeWidth={hl ? 2.5 : 1.2}
              strokeDasharray={e.weight < 0 ? "4,3" : "none"}
              markerEnd={hl ? "url(#arrow-hl)" : "url(#arrow)"}
              opacity={hl ? 1 : 0.5}
            />
          </g>
        );
      })}
      {Object.entries(positions).map(([name, [x, y]]) => {
        const v = VARIABLES[name];
        if (!v) return null;
        const fillMap = { actuator: "#451a03", state: "#0c2340", diagnostic: "#2e1065", shape: "#083344" };
        const strokeMap = { actuator: "#f59e0b", state: "#3b82f6", diagnostic: "#8b5cf6", shape: "#06b6d4" };
        return (
          <g key={name}>
            <circle cx={x} cy={y} r={24} fill={fillMap[v.type]} stroke={strokeMap[v.type]} strokeWidth={1.5} />
            <text x={x} y={y-4} textAnchor="middle" fill="#e2e8f0" fontSize={9} fontWeight={600} fontFamily="monospace">{name}</text>
            <text x={x} y={y+8} textAnchor="middle" fill="#94a3b8" fontSize={7} fontFamily="monospace">{v.value}{v.unit ? ` ${v.unit}` : ""}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Chat Message ──
function ChatMessage({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div style={{
      display: "flex", flexDirection: "column",
      alignItems: isUser ? "flex-end" : "flex-start",
      marginBottom: 12,
    }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 8, marginBottom: 4,
      }}>
        <span style={{ fontSize: 11, color: "#64748b", fontWeight: 600, letterSpacing: 0.5 }}>
          {isUser ? "OPERATOR" : "CAUSAL COPILOT"}
        </span>
        {msg.level && <LevelBadge level={msg.level} />}
      </div>
      <div style={{
        maxWidth: "88%", padding: "10px 14px", borderRadius: 10,
        background: isUser ? "#1e3a5f" : "#1a1a2e",
        border: isUser ? "1px solid #2563eb33" : "1px solid #334155",
        color: "#e2e8f0", fontSize: 13.5, lineHeight: 1.6,
        whiteSpace: "pre-wrap", fontFamily: "'IBM Plex Sans', sans-serif",
      }}>
        {msg.content}
        {msg.loading && <span className="blink" style={{ color: "#f59e0b" }}> ▍</span>}
      </div>
    </div>
  );
}

// ── Main App ──
export default function FusionMindCopilot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState("chat"); // chat | graph | about
  const [highlightEdge, setHighlightEdge] = useState(null);
  const chatRef = useRef(null);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages]);

  // Classify query locally
  const classifyQuery = (q) => {
    const ql = q.toLowerCase();
    if (/would\s+have|could\s+have|had\s+we|instead\s+of|what\s+would\s+.+have\s+been/.test(ql)) return 3;
    if (/what\s+happens?\s+if|if\s+we\s+(set|increase|decrease|change)|effect\s+of|do\s*\(|what\s+if\s+we/.test(ql)) return 2;
    if (/hypothes|suggest.*(experiment|test)|what.*new.*causal/.test(ql)) return "H";
    if (/why\s+(did|does|is)|explain|cause\s+of|reason\s+for/.test(ql)) return "E";
    if (/safe|disruption|risk|limit/.test(ql)) return "S";
    return 1;
  };

  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading) return;
    const query = input.trim();
    const level = classifyQuery(query);
    setInput("");

    const userMsg = { role: "user", content: query, level };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    // Build conversation history for API
    const history = [...messages, userMsg].map(m => ({
      role: m.role === "user" ? "user" : "assistant",
      content: m.content,
    }));

    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: buildSystemPrompt(),
          messages: history,
        }),
      });

      const data = await response.json();
      const text = data.content?.map(b => b.type === "text" ? b.text : "").join("") || "Error: No response";

      setMessages(prev => [...prev, { role: "assistant", content: text, level }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `Connection error: ${err.message}. The Causal Copilot requires Claude API access.`,
        level: "S",
      }]);
    }
    setLoading(false);
  }, [input, loading, messages]);

  const handleExample = (q) => {
    setInput(q);
    setTab("chat");
  };

  // ── Render ──
  return (
    <div style={{
      width: "100%", minHeight: "100vh", background: "#0a0a14",
      fontFamily: "'IBM Plex Sans', 'SF Pro', system-ui, sans-serif",
      color: "#e2e8f0", display: "flex", flexDirection: "column",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
      <style>{`
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        .blink { animation: blink 1s infinite; }
        input::placeholder { color: #475569; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }
      `}</style>

      {/* Header */}
      <div style={{
        padding: "16px 24px", borderBottom: "1px solid #1e293b",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "linear-gradient(180deg, #0f0f1e 0%, #0a0a14 100%)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8,
            background: "linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, fontWeight: 700,
          }}>⚛</div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: -0.3 }}>
              FusionMind Causal Copilot
            </div>
            <div style={{ fontSize: 11, color: "#64748b", fontWeight: 500 }}>
              PF8 · LLM × Pearl's do-Calculus · Real MAST Data
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 2, background: "#111827", borderRadius: 8, padding: 2 }}>
          {["chat", "graph", "about"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "6px 16px", borderRadius: 6, border: "none", cursor: "pointer",
              background: tab === t ? "#1e293b" : "transparent",
              color: tab === t ? "#f59e0b" : "#64748b",
              fontSize: 12, fontWeight: 600, letterSpacing: 0.5, textTransform: "uppercase",
            }}>
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* Left: Examples Panel */}
        <div style={{
          width: 260, borderRight: "1px solid #1e293b", padding: 16,
          overflowY: "auto", flexShrink: 0,
          display: tab === "chat" ? "block" : "none",
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#64748b", letterSpacing: 1, marginBottom: 12 }}>
            EXAMPLE QUERIES
          </div>
          {EXAMPLE_QUERIES.map((ex, i) => (
            <button key={i} onClick={() => handleExample(ex.q)} style={{
              display: "block", width: "100%", textAlign: "left",
              padding: "8px 10px", marginBottom: 6, borderRadius: 6,
              background: "#111827", border: "1px solid #1e293b",
              color: "#cbd5e1", fontSize: 12, cursor: "pointer",
              lineHeight: 1.4, transition: "border-color 0.15s",
            }}
            onMouseOver={e => e.currentTarget.style.borderColor = "#334155"}
            onMouseOut={e => e.currentTarget.style.borderColor = "#1e293b"}
            >
              <LevelBadge level={ex.level} />
              <div style={{ marginTop: 4 }}>{ex.q}</div>
            </button>
          ))}

          <div style={{
            marginTop: 20, padding: 12, borderRadius: 8,
            background: "#0f172a", border: "1px solid #1e293b",
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#f59e0b", marginBottom: 6 }}>
              PEARL'S LADDER
            </div>
            <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.6 }}>
              <strong style={{ color: "#3b82f6" }}>L1</strong> Observation: P(Y|X)<br/>
              <strong style={{ color: "#f59e0b" }}>L2</strong> Intervention: P(Y|do(X))<br/>
              <strong style={{ color: "#a855f7" }}>L3</strong> Counterfactual: P(Y_x|X=x')
            </div>
          </div>
        </div>

        {/* Center: Chat / Graph / About */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {tab === "chat" && (
            <>
              <div ref={chatRef} style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>
                {messages.length === 0 && (
                  <div style={{ textAlign: "center", paddingTop: 60, color: "#475569" }}>
                    <div style={{ fontSize: 40, marginBottom: 12 }}>⚛</div>
                    <div style={{ fontSize: 16, fontWeight: 600, color: "#94a3b8", marginBottom: 6 }}>
                      FusionMind Causal Copilot
                    </div>
                    <div style={{ fontSize: 13, maxWidth: 400, margin: "0 auto", lineHeight: 1.6 }}>
                      Ask causal questions about the tokamak plasma.
                      I reason using Pearl's do-calculus — not just correlations.
                      <br/><br/>
                      Try: "What happens to Te if we increase NBI to 8 MW?"
                    </div>
                  </div>
                )}
                {messages.map((m, i) => <ChatMessage key={i} msg={m} />)}
                {loading && <ChatMessage msg={{ role: "assistant", content: "", loading: true }} />}
              </div>

              {/* Input */}
              <div style={{
                padding: "12px 16px", borderTop: "1px solid #1e293b",
                display: "flex", gap: 8,
              }}>
                <input
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && sendMessage()}
                  placeholder="Ask a causal question about the plasma..."
                  style={{
                    flex: 1, padding: "10px 14px", borderRadius: 8,
                    background: "#111827", border: "1px solid #1e293b",
                    color: "#e2e8f0", fontSize: 13.5, outline: "none",
                    fontFamily: "inherit",
                  }}
                />
                <button onClick={sendMessage} disabled={loading || !input.trim()} style={{
                  padding: "10px 20px", borderRadius: 8, border: "none",
                  background: loading ? "#334155" : "#f59e0b",
                  color: loading ? "#64748b" : "#0a0a14",
                  fontWeight: 700, fontSize: 13, cursor: loading ? "wait" : "pointer",
                }}>
                  {loading ? "..." : "Send"}
                </button>
              </div>
            </>
          )}

          {tab === "graph" && (
            <div style={{ flex: 1, overflowY: "auto", padding: 20 }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#f59e0b", marginBottom: 4 }}>
                Discovered Causal Graph (CPDE v3.2)
              </div>
              <div style={{ fontSize: 11, color: "#64748b", marginBottom: 16 }}>
                F1 = 91.9% on real MAST data · 15 edges · 100% sign accuracy
              </div>
              <div style={{
                background: "#0f172a", borderRadius: 12, border: "1px solid #1e293b",
                padding: 16,
              }}>
                <CausalGraphSVG highlightEdge={highlightEdge} />
              </div>

              <div style={{
                display: "flex", gap: 16, marginTop: 12,
                flexWrap: "wrap",
              }}>
                {[
                  { color: "#f59e0b", label: "Actuator" },
                  { color: "#3b82f6", label: "Plasma State" },
                  { color: "#8b5cf6", label: "Diagnostic" },
                  { color: "#06b6d4", label: "Shape" },
                ].map(l => (
                  <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: l.color }} />
                    <span style={{ fontSize: 11, color: "#94a3b8" }}>{l.label}</span>
                  </div>
                ))}
                <span style={{ fontSize: 11, color: "#64748b" }}>
                  Solid = positive · Dashed = negative
                </span>
              </div>

              <div style={{ marginTop: 20 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "#94a3b8", marginBottom: 8 }}>
                  Edge List ({CAUSAL_EDGES.length} edges)
                </div>
                {CAUSAL_EDGES.map((e, i) => (
                  <div key={i}
                    onMouseEnter={() => setHighlightEdge(e)}
                    onMouseLeave={() => setHighlightEdge(null)}
                    style={{
                      padding: "6px 10px", borderRadius: 6, marginBottom: 3,
                      background: highlightEdge === e ? "#1e293b" : "transparent",
                      display: "flex", justifyContent: "space-between", alignItems: "center",
                      fontSize: 12, cursor: "default", transition: "background 0.1s",
                    }}>
                    <span style={{ fontFamily: "'IBM Plex Mono', monospace" }}>
                      {e.from} → {e.to}
                    </span>
                    <span style={{
                      color: e.weight > 0 ? "#22c55e" : "#ef4444",
                      fontWeight: 600,
                    }}>
                      {e.weight > 0 ? "+" : ""}{e.weight.toFixed(2)}
                    </span>
                    <span style={{ color: "#64748b", fontSize: 11 }}>{e.mechanism}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {tab === "about" && (
            <div style={{ flex: 1, overflowY: "auto", padding: 24 }}>
              <div style={{ maxWidth: 600 }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: "#f59e0b", marginBottom: 8 }}>
                  PF8: LLM-Augmented Causal Plasma Reasoning
                </div>
                <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.7, marginBottom: 24 }}>
                  This is the first system that combines Pearl's causal inference framework
                  with a large language model for tokamak plasma control. Unlike standard
                  chatbots, the Causal Copilot has access to the <em>discovered causal structure</em> of
                  the plasma — it doesn't just correlate, it reasons about causation.
                </div>

                <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0", marginBottom: 12 }}>
                  What Makes This Novel
                </div>
                {[
                  { title: "Causal Grounding", desc: "LLM responses are grounded in the DAG discovered by CPDE on real MAST data (F1=91.9%). Not hallucination — structure." },
                  { title: "Pearl's Ladder Awareness", desc: "Queries are classified into Observation (L1), Intervention (L2), or Counterfactual (L3). The LLM applies different reasoning protocols for each." },
                  { title: "Simpson's Paradox Prevention", desc: "The system warns when correlation ≠ causation, drawing on the validated finding from Alcator C-Mod (ρ: +0.53→+0.02)." },
                  { title: "Hypothesis Generation", desc: "Can propose new causal relationships to test, grounded in the existing graph + physics constraints." },
                  { title: "Operator-Friendly", desc: "Natural language interface for control room use. No need to understand do-calculus — the copilot handles the formalism." },
                ].map((item, i) => (
                  <div key={i} style={{
                    padding: "12px 14px", borderRadius: 8, marginBottom: 8,
                    background: "#0f172a", border: "1px solid #1e293b",
                  }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: "#f59e0b", marginBottom: 4 }}>
                      {item.title}
                    </div>
                    <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.5 }}>
                      {item.desc}
                    </div>
                  </div>
                ))}

                <div style={{
                  marginTop: 24, padding: 16, borderRadius: 8,
                  background: "#1a0a0a", border: "1px solid #7f1d1d",
                }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: "#ef4444", marginBottom: 4 }}>
                    PATENT STATUS
                  </div>
                  <div style={{ fontSize: 12, color: "#fca5a5", lineHeight: 1.5 }}>
                    PF8 — LLM-Augmented Causal Plasma Reasoning<br/>
                    Novelty: 9/10 · No prior art in fusion + LLM + causal inference<br/>
                    Filing: US Provisional planned
                  </div>
                </div>

                <div style={{ marginTop: 16, fontSize: 11, color: "#475569" }}>
                  Dr. Mladen Mester · FusionMind 4.0 · March 2026 · CONFIDENTIAL
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
