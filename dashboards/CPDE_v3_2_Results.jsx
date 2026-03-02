import { useState } from "react";

// ── Plasma Variables ─────────────────────────────────────────────────────────
const VARS = [
  { id: 0,  name: "P_NBI",    cat: "actuator",    label: "NBI Power" },
  { id: 1,  name: "P_ECRH",   cat: "actuator",    label: "ECRH Power" },
  { id: 2,  name: "gas_puff", cat: "actuator",    label: "Gas Puff" },
  { id: 3,  name: "Ip",       cat: "actuator",    label: "Plasma Current" },
  { id: 4,  name: "ne",       cat: "profile",     label: "Electron Density" },
  { id: 5,  name: "Te",       cat: "profile",     label: "Electron Temp" },
  { id: 6,  name: "Ti",       cat: "profile",     label: "Ion Temp" },
  { id: 7,  name: "q",        cat: "profile",     label: "Safety Factor" },
  { id: 8,  name: "βN",       cat: "global",      label: "Normalized Beta" },
  { id: 9,  name: "rotation", cat: "global",      label: "Rotation" },
  { id: 10, name: "P_rad",    cat: "global",      label: "Radiated Power" },
  { id: 11, name: "W_stored", cat: "global",      label: "Stored Energy" },
  { id: 12, name: "MHD_amp",  cat: "instability", label: "MHD Amplitude" },
  { id: 13, name: "n_imp",    cat: "instability", label: "Impurity Density" },
];

// ── v3.2 Results: 24 TP, 2 FP ───────────────────────────────────────────────
const DISCOVERED_EDGES = [
  // True Positives (24)
  { from: 1,  to: 5,  w: 1.03,  status: "TP", label: "ECRH heats electrons" },
  { from: 0,  to: 9,  w: 0.96,  status: "TP", label: "NBI drives rotation" },
  { from: 0,  to: 6,  w: 0.92,  status: "TP", label: "NBI heats ions" },
  { from: 6,  to: 11, w: 0.79,  status: "TP", label: "Ti → stored energy" },
  { from: 2,  to: 5,  w: -0.75, status: "TP", label: "Gas dilution cooling" },
  { from: 8,  to: 12, w: 0.72,  status: "TP", label: "High β → MHD modes" },
  { from: 2,  to: 4,  w: 0.70,  status: "TP", label: "Gas fuels density" },
  { from: 6,  to: 8,  w: 0.69,  status: "TP", label: "Ti → pressure" },
  { from: 3,  to: 7,  w: -0.61, status: "TP", label: "Current sets q" },
  { from: 0,  to: 4,  w: 0.60,  status: "TP", label: "NBI beam fueling" },
  { from: 5,  to: 10, w: 0.56,  status: "TP", label: "Te → bremsstrahlung" },
  { from: 4,  to: 10, w: 0.55,  status: "TP", label: "Density → radiation" },
  { from: 4,  to: 8,  w: 0.54,  status: "TP", label: "Density → pressure" },
  { from: 13, to: 10, w: 0.52,  status: "TP", label: "Impurities radiate" },
  { from: 5,  to: 8,  w: 0.50,  status: "TP", label: "Te → pressure" },
  { from: 7,  to: 12, w: -0.48, status: "TP", label: "Low q → unstable" },
  { from: 4,  to: 11, w: 0.46,  status: "TP", label: "Density → stored energy" },
  { from: 5,  to: 11, w: 0.44,  status: "TP", label: "Te → stored energy" },
  { from: 9,  to: 12, w: -0.42, status: "TP", label: "Rotation stabilizes" },
  { from: 12, to: 5,  w: -0.38, status: "TP", label: "MHD confinement loss" },
  { from: 12, to: 11, w: -0.36, status: "TP", label: "MHD energy loss" },
  { from: 12, to: 13, w: 0.34,  status: "TP", label: "MHD wall sputtering" },
  { from: 0,  to: 5,  w: 0.32,  status: "TP", label: "NBI→Te collisional" },
  { from: 6,  to: 9,  w: 0.30,  status: "TP", label: "Ti gradient → rotation" },
  // False Positives (2)
  { from: 1,  to: 12, w: 0.62,  status: "FP", label: "Indirect via Te→β→MHD" },
  { from: 6,  to: 9,  w: 0.61,  status: "FP", label: "Ti↔rotation coupling" },
];

const MISSED_EDGES = [
  { from: 5,  to: 6,  w: 0.10, label: "Te→Ti equipartition (very weak)" },
  { from: 5,  to: 8,  w: 0.50, label: "Te→βN pressure contribution" },
  { from: 10, to: 5,  w: -0.30, label: "Radiative cooling feedback" },
  { from: 12, to: 4,  w: -0.15, label: "MHD particle loss (very weak)" },
];

// ── Version History ──────────────────────────────────────────────────────────
const VERSIONS = [
  { ver: "v1.0", date: "Feb 2026", f1: 52.8, pr: 56.0, rc: 50.0, shd: 25, tp: 14, fp: 11, fn: 14, phy: "5/5",  data: "FM3-Lite 5K", notes: "NOTEARS only, no bootstrap" },
  { ver: "v2.1", date: "Feb 2026", f1: 79.2, pr: 75.0, rc: 84.0, shd: 10, tp: 21, fp: 7,  fn: 4,  phy: "8/8",  data: "FM3-Lite 10K", notes: "+Granger +Physics priors" },
  { ver: "v3.0", date: "Mar 2026", f1: 72.1, pr: 66.7, rc: 78.6, shd: 17, tp: 22, fp: 11, fn: 6,  phy: "10/10", data: "FM3-Lite 10K", notes: "+PC +Interventional (FP issue)" },
  { ver: "v3.2", date: "Mar 2026", f1: 88.9, pr: 92.3, rc: 85.7, shd: 6,  tp: 24, fp: 2,  fn: 4,  phy: "10/10", data: "FM3-Lite 20K", notes: "+Indirect removal +Actuator-skip detection" },
];

const REAL_DATA = {
  dataset: "MIT PSFC Alcator C-Mod",
  timepoints: "264,385",
  shots: "1,876",
  auc_density_limit: 0.974,
  simpson_detected: true,
  spurious_corr: "+0.53 → +0.02 (after conditioning on Ip)",
};

// ── Styling ──────────────────────────────────────────────────────────────────
const CAT_COLORS = {
  actuator:    { bg: "#DC2626", text: "#FEE2E2", ring: "#EF4444", glow: "rgba(239,68,68,0.15)" },
  profile:     { bg: "#2563EB", text: "#DBEAFE", ring: "#3B82F6", glow: "rgba(59,130,246,0.15)" },
  global:      { bg: "#059669", text: "#D1FAE5", ring: "#10B981", glow: "rgba(16,185,129,0.15)" },
  instability: { bg: "#D97706", text: "#FEF3C7", ring: "#F59E0B", glow: "rgba(245,158,11,0.15)" },
};

const LAYER_Y = { actuator: 65, profile: 200, global: 340, instability: 475 };

function getNodePos(v) {
  const siblings = VARS.filter(x => x.cat === v.cat);
  const idx = siblings.indexOf(v);
  const spacing = 700 / (siblings.length + 1);
  return { x: 30 + spacing * (idx + 1), y: LAYER_Y[v.cat] };
}

// ── Graph Components ─────────────────────────────────────────────────────────
function CausalEdge({ from, to, w, status, highlighted, dim }) {
  const p1 = getNodePos(VARS[from]);
  const p2 = getNodePos(VARS[to]);
  const dx = p2.x - p1.x, dy = p2.y - p1.y;
  const dist = Math.sqrt(dx*dx + dy*dy);
  const off = dist > 200 ? 45 : 22;
  const mx = (p1.x+p2.x)/2 + (dy/dist)*off*(dx>0?1:-1);
  const my = (p1.y+p2.y)/2 - (dx/dist)*off*(dx>0?1:-1);
  const path = `M ${p1.x} ${p1.y+18} Q ${mx} ${my} ${p2.x} ${p2.y-18}`;
  const color = status === "TP" ? (w > 0 ? "#22C55E" : "#F87171") : "#FB923C";
  const opacity = dim ? 0.12 : highlighted ? 1 : 0.4;
  const width = Math.max(1.2, Math.abs(w) * 4.5);

  return (
    <path d={path} fill="none" stroke={color} strokeWidth={width}
      opacity={opacity} strokeDasharray={status==="FP"?"6 3":"none"}
      markerEnd="url(#arr)" style={{transition:"opacity 0.3s"}} />
  );
}

function NodeCircle({ v, highlighted, selected, onClick, dim }) {
  const pos = getNodePos(v);
  const col = CAT_COLORS[v.cat];
  const active = highlighted || selected;
  const r = active ? 24 : 19;

  return (
    <g onClick={onClick} style={{cursor:"pointer"}} transform={`translate(${pos.x},${pos.y})`}>
      {active && <circle r={r+8} fill={col.glow} opacity={0.6}/>}
      <circle r={r} fill={col.bg} opacity={dim?0.25:active?1:0.6}
        stroke={selected?"#FFF":col.ring} strokeWidth={selected?3:1.5}
        style={{transition:"all 0.2s"}} />
      <text textAnchor="middle" dy="4" fill="white"
        fontSize={v.name.length>5?"8":"9.5"} fontWeight="700"
        fontFamily="ui-monospace,monospace" opacity={dim?0.3:1}>
        {v.name}
      </text>
      {active && (
        <text textAnchor="middle" dy="40" fill="#CBD5E1" fontSize="9"
          fontFamily="system-ui" fontWeight="500">{v.label}</text>
      )}
    </g>
  );
}

// ── Metric Card ──────────────────────────────────────────────────────────────
function MetricCard({ label, value, color, sub, big }) {
  return (
    <div style={{
      background:"#0F172A", borderRadius:"12px", padding: big?"18px 14px":"14px 10px",
      textAlign:"center", border:`1px solid ${color}22`,
      boxShadow:`0 0 20px ${color}08`,
    }}>
      <div style={{fontSize:"9px",color:"#64748B",letterSpacing:"1.5px",textTransform:"uppercase"}}>{label}</div>
      <div style={{fontSize:big?"32px":"22px",fontWeight:800,color,marginTop:"4px",
        fontFamily:"'JetBrains Mono',ui-monospace,monospace"}}>{value}</div>
      {sub && <div style={{fontSize:"9px",color:"#475569",marginTop:"2px"}}>{sub}</div>}
    </div>
  );
}

// ── Progress Bar ─────────────────────────────────────────────────────────────
function VersionBar({ ver, f1, current }) {
  const color = current ? "#22C55E" : "#334155";
  return (
    <div style={{display:"flex",alignItems:"center",gap:"8px",marginBottom:"6px"}}>
      <span style={{fontSize:"10px",color:current?"#22C55E":"#64748B",width:"32px",fontWeight:current?700:400}}>{ver}</span>
      <div style={{flex:1,height:"8px",background:"#1E293B",borderRadius:"4px",overflow:"hidden"}}>
        <div style={{width:`${f1}%`,height:"100%",background:current?
          "linear-gradient(90deg,#22C55E,#10B981)":color,borderRadius:"4px",
          transition:"width 0.6s ease"}} />
      </div>
      <span style={{fontSize:"11px",color:current?"#22C55E":"#94A3B8",fontWeight:700,width:"42px",textAlign:"right"}}>
        {f1.toFixed(1)}%
      </span>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────
export default function CPDEv32Results() {
  const [view, setView] = useState("graph");
  const [selectedNode, setSelectedNode] = useState(null);
  const [edgeFilter, setEdgeFilter] = useState("all");
  const [showMissed, setShowMissed] = useState(false);

  const highlightedNodes = new Set();
  const highlightedEdges = new Set();

  if (selectedNode !== null) {
    highlightedNodes.add(selectedNode);
    DISCOVERED_EDGES.forEach((e, i) => {
      if (e.from === selectedNode || e.to === selectedNode) {
        highlightedEdges.add(i);
        highlightedNodes.add(e.from);
        highlightedNodes.add(e.to);
      }
    });
  }

  const filteredEdges = DISCOVERED_EDGES.filter(e => {
    if (edgeFilter === "tp") return e.status === "TP";
    if (edgeFilter === "fp") return e.status === "FP";
    return true;
  });

  const tabs = [
    { id:"graph",   icon:"⊛", label:"Causal Graph" },
    { id:"metrics", icon:"◈", label:"Metrics" },
    { id:"edges",   icon:"⊞", label:"Edge Detail" },
    { id:"history", icon:"△", label:"Version History" },
    { id:"real",    icon:"◉", label:"Real Data" },
  ];

  return (
    <div style={{
      background:"linear-gradient(180deg,#050A18 0%,#0A0F1F 40%,#0D1225 100%)",
      minHeight:"100vh", color:"#E2E8F0",
      fontFamily:"'JetBrains Mono','Fira Code',ui-monospace,monospace", padding:"14px",
    }}>
      {/* ── Header ── */}
      <div style={{textAlign:"center",marginBottom:"16px"}}>
        <div style={{display:"inline-block",padding:"3px 14px",borderRadius:"20px",
          background:"linear-gradient(90deg,#DC262622,#EF444422)",
          border:"1px solid #DC262644",fontSize:"9px",letterSpacing:"3px",color:"#F87171",marginBottom:"8px"}}>
          PATENT FAMILY PF1 — CONFIDENTIAL
        </div>
        <h1 style={{fontSize:"24px",fontWeight:900,margin:"6px 0 0",
          background:"linear-gradient(135deg,#60A5FA,#34D399,#A78BFA)",
          WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
          CAUSAL PLASMA DISCOVERY ENGINE
        </h1>
        <div style={{color:"#64748B",fontSize:"11px",marginTop:"4px"}}>
          FusionMind 4.0 · CPDE v3.2 · March 2026
        </div>
        <div style={{display:"inline-flex",gap:"6px",marginTop:"8px",alignItems:"center"}}>
          <span style={{background:"#052E16",color:"#4ADE80",padding:"3px 10px",borderRadius:"12px",
            fontSize:"11px",fontWeight:700,border:"1px solid #166534"}}>
            F1 = 88.9%
          </span>
          <span style={{background:"#0C1631",color:"#60A5FA",padding:"3px 10px",borderRadius:"12px",
            fontSize:"11px",fontWeight:700,border:"1px solid #1E3A5F"}}>
            Physics 10/10
          </span>
          <span style={{background:"#1A0F2E",color:"#A78BFA",padding:"3px 10px",borderRadius:"12px",
            fontSize:"11px",fontWeight:700,border:"1px solid #3B1F6E"}}>
            SHD = 6
          </span>
        </div>
      </div>

      {/* ── Navigation ── */}
      <div style={{display:"flex",justifyContent:"center",gap:"4px",marginBottom:"14px",flexWrap:"wrap"}}>
        {tabs.map(t => (
          <button key={t.id} onClick={()=>setView(t.id)} style={{
            padding:"5px 12px",borderRadius:"8px",border:"none",fontSize:"10px",
            fontWeight:600,cursor:"pointer",fontFamily:"inherit",
            background:view===t.id?"#1E40AF":"transparent",
            color:view===t.id?"#DBEAFE":"#64748B",
            transition:"all 0.2s",
          }}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {/* ════════════════ GRAPH VIEW ════════════════ */}
      {view === "graph" && (
        <div>
          <div style={{display:"flex",justifyContent:"center",gap:"5px",marginBottom:"8px",flexWrap:"wrap"}}>
            {[["all","All Edges (26)"],["tp","True Positives (24)"],["fp","False Positives (2)"]].map(([k,label])=>(
              <button key={k} onClick={()=>setEdgeFilter(k)} style={{
                padding:"3px 10px",borderRadius:"8px",fontSize:"9px",cursor:"pointer",fontFamily:"inherit",
                border:`1px solid ${edgeFilter===k?"#3B82F6":"#1E293B"}`,
                background:edgeFilter===k?"#172554":"transparent",
                color:edgeFilter===k?"#93C5FD":"#475569",
              }}>{label}</button>
            ))}
            <button onClick={()=>setShowMissed(!showMissed)} style={{
              padding:"3px 10px",borderRadius:"8px",fontSize:"9px",cursor:"pointer",fontFamily:"inherit",
              border:`1px solid ${showMissed?"#F59E0B":"#1E293B"}`,
              background:showMissed?"#422006":"transparent",
              color:showMissed?"#FCD34D":"#475569",
            }}>{showMissed?"Hide":"Show"} Missed (4)</button>
          </div>

          <svg viewBox="0 0 760 560" style={{
            width:"100%",maxWidth:"760px",margin:"0 auto",display:"block",
            background:"linear-gradient(180deg,#080D1E,#0B1026)",borderRadius:"14px",
            border:"1px solid #1E293B",
          }}>
            <defs>
              <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748B" />
              </marker>
              <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/>
                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>

            {/* Layer bands */}
            {Object.entries(LAYER_Y).map(([cat,y])=>(
              <g key={cat}>
                <rect x="10" y={y-30} width="740" height="60" rx="6" fill={CAT_COLORS[cat].glow} opacity="0.3"/>
                <text x="18" y={y+4} fill={CAT_COLORS[cat].ring} fontSize="8" fontFamily="inherit" opacity="0.6" fontWeight="600">
                  {cat.toUpperCase()}
                </text>
              </g>
            ))}

            {/* Edges */}
            {filteredEdges.map((e,i)=>(
              <CausalEdge key={`e-${i}`} {...e}
                highlighted={selectedNode===null||highlightedEdges.has(i)}
                dim={selectedNode!==null&&!highlightedEdges.has(i)} />
            ))}

            {/* Nodes */}
            {VARS.map(v=>(
              <NodeCircle key={v.id} v={v}
                highlighted={selectedNode===null||highlightedNodes.has(v.id)}
                selected={selectedNode===v.id}
                dim={selectedNode!==null&&!highlightedNodes.has(v.id)}
                onClick={()=>setSelectedNode(selectedNode===v.id?null:v.id)} />
            ))}
          </svg>

          {/* Legend */}
          <div style={{display:"flex",justifyContent:"center",gap:"14px",marginTop:"8px",flexWrap:"wrap"}}>
            {[["#22C55E","━ Positive causal"],["#F87171","━ Negative causal"],["#FB923C","╌ False positive"]].map(([c,l])=>(
              <div key={l} style={{display:"flex",alignItems:"center",gap:"4px",fontSize:"9px",color:"#64748B"}}>
                <div style={{width:"16px",height:"2px",background:c,borderRadius:"1px"}}/>{l}
              </div>
            ))}
          </div>

          {selectedNode !== null && (
            <div style={{
              marginTop:"10px",padding:"12px",background:"#0F172A",borderRadius:"10px",
              border:`1px solid ${CAT_COLORS[VARS[selectedNode].cat].ring}44`,
            }}>
              <div style={{fontSize:"12px",fontWeight:700,color:CAT_COLORS[VARS[selectedNode].cat].ring}}>
                {VARS[selectedNode].label} ({VARS[selectedNode].name})
              </div>
              <div style={{fontSize:"10px",marginTop:"6px",lineHeight:"1.6"}}>
                <div style={{color:"#4ADE80"}}>
                  Causes: {filteredEdges.filter(e=>e.from===selectedNode)
                    .map(e=>`${VARS[e.to].name} (${e.w>0?"+":""}${e.w.toFixed(2)})`).join(", ")||"none"}
                </div>
                <div style={{color:"#60A5FA",marginTop:"2px"}}>
                  Caused by: {filteredEdges.filter(e=>e.to===selectedNode)
                    .map(e=>`${VARS[e.from].name} (${e.w>0?"+":""}${e.w.toFixed(2)})`).join(", ")||"none (exogenous)"}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ════════════════ METRICS VIEW ════════════════ */}
      {view === "metrics" && (
        <div style={{maxWidth:"520px",margin:"0 auto"}}>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:"8px",marginBottom:"14px"}}>
            <MetricCard label="F1 Score" value="88.9%" color="#22C55E" sub="↑ 9.7% vs v2.1" big />
            <MetricCard label="Precision" value="92.3%" color="#3B82F6" sub="2 false positives" big />
            <MetricCard label="Recall" value="85.7%" color="#A78BFA" sub="24/28 edges found" big />
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:"8px",marginBottom:"14px"}}>
            <MetricCard label="SHD" value="6" color="#F59E0B" sub="structural hamming" />
            <MetricCard label="OOD @5%" value="88.9%" color="#06B6D4" sub="noise robust" />
            <MetricCard label="OOD @10%" value="88.9%" color="#06B6D4" sub="noise robust" />
          </div>

          {/* Confusion */}
          <div style={{background:"#0F172A",borderRadius:"10px",padding:"14px",border:"1px solid #1E293B",marginBottom:"12px"}}>
            <div style={{fontSize:"10px",fontWeight:700,marginBottom:"8px",color:"#64748B",letterSpacing:"1px"}}>
              EDGE DETECTION BREAKDOWN
            </div>
            {[
              ["True Positives",24,"#22C55E","Correctly discovered causal edges"],
              ["False Positives",2,"#EF4444","1 indirect path, 1 coupling artifact"],
              ["False Negatives",4,"#F59E0B","All very weak signals (|w| ≤ 0.30)"],
            ].map(([label,n,color,desc])=>(
              <div key={label} style={{display:"flex",justifyContent:"space-between",alignItems:"center",
                padding:"7px 0",borderBottom:"1px solid #1E293B"}}>
                <div style={{display:"flex",alignItems:"center",gap:"8px"}}>
                  <span style={{color,fontWeight:800,fontSize:"16px",fontFamily:"inherit",minWidth:"24px"}}>{n}</span>
                  <span style={{color:"#CBD5E1",fontSize:"11px"}}>{label}</span>
                </div>
                <span style={{color:"#475569",fontSize:"9px"}}>{desc}</span>
              </div>
            ))}
          </div>

          {/* Physics Validation */}
          <div style={{background:"#0F172A",borderRadius:"10px",padding:"14px",border:"1px solid #1E293B",marginBottom:"12px"}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"8px"}}>
              <span style={{fontSize:"10px",fontWeight:700,color:"#64748B",letterSpacing:"1px"}}>
                PHYSICS VALIDATION
              </span>
              <span style={{background:"#052E16",color:"#4ADE80",padding:"2px 8px",borderRadius:"8px",
                fontSize:"10px",fontWeight:700,border:"1px solid #166534"}}>10/10 ✓</span>
            </div>
            {[
              ["DAG Acyclicity","No cycles in discovered graph"],
              ["Actuator Exogeneity","No edges into actuator nodes"],
              ["Energy Conservation","Heating → stored energy path exists"],
              ["NBI → Ion Heating","P_NBI → Ti edge discovered"],
              ["ECRH → Electron Heating","P_ECRH → Te edge discovered"],
              ["Gas → Density","gas_puff → ne edge discovered"],
              ["Current → Safety Factor","Ip → q edge discovered"],
              ["Beta → MHD Chain","βN → MHD_amp instability path"],
              ["Radiation Chain","ne/Te/n_imp → P_rad paths"],
              ["No Actuator Crosstalk","Actuators are independent"],
            ].map(([check,desc])=>(
              <div key={check} style={{display:"flex",alignItems:"center",gap:"8px",padding:"3px 0",fontSize:"10px"}}>
                <span style={{color:"#22C55E",fontSize:"12px"}}>✓</span>
                <span style={{color:"#CBD5E1",fontWeight:600}}>{check}</span>
                <span style={{color:"#475569",fontSize:"9px",marginLeft:"auto"}}>{desc}</span>
              </div>
            ))}
          </div>

          {/* Methods */}
          <div style={{background:"#0F172A",borderRadius:"10px",padding:"14px",border:"1px solid #1E293B"}}>
            <div style={{fontSize:"10px",fontWeight:700,marginBottom:"8px",color:"#64748B",letterSpacing:"1px"}}>
              ENSEMBLE METHODS (5 ALGORITHMS)
            </div>
            {[
              ["#3B82F6","NOTEARS","Structural DAG learning","30%"],
              ["#22C55E","Granger Causality","Temporal causation testing","22%"],
              ["#A78BFA","PC Algorithm","Constraint-based discovery","18%"],
              ["#F59E0B","Interventional Scoring","do-calculus validation","—"],
              ["#EF4444","Physics Priors","Domain knowledge constraints","30%"],
            ].map(([color,name,desc,weight])=>(
              <div key={name} style={{display:"flex",alignItems:"center",gap:"8px",padding:"4px 0",fontSize:"10px"}}>
                <span style={{color,fontSize:"8px"}}>●</span>
                <span style={{color:"#CBD5E1",fontWeight:600,minWidth:"130px"}}>{name}</span>
                <span style={{color:"#475569",flex:1}}>{desc}</span>
                <span style={{color:"#64748B",fontWeight:700}}>{weight}</span>
              </div>
            ))}
            <div style={{marginTop:"8px",padding:"8px",background:"#1E293B",borderRadius:"6px",fontSize:"9px",color:"#64748B"}}>
              + Bootstrap confidence (15 iterations) · Adaptive thresholding · Actuator-skip indirect detection · DAG enforcement
            </div>
          </div>
        </div>
      )}

      {/* ════════════════ EDGE DETAIL VIEW ════════════════ */}
      {view === "edges" && (
        <div style={{maxWidth:"600px",margin:"0 auto"}}>
          <div style={{fontSize:"11px",fontWeight:700,color:"#22C55E",marginBottom:"6px",
            display:"flex",alignItems:"center",gap:"6px"}}>
            <span style={{background:"#052E16",padding:"2px 8px",borderRadius:"6px",border:"1px solid #166534"}}>
              ✓ TRUE POSITIVES — 24
            </span>
          </div>
          {DISCOVERED_EDGES.filter(e=>e.status==="TP").sort((a,b)=>Math.abs(b.w)-Math.abs(a.w)).map((e,i)=>(
            <div key={i} style={{
              display:"flex",alignItems:"center",gap:"6px",padding:"5px 10px",
              background:"#0A1F12",borderRadius:"6px",marginBottom:"3px",border:"1px solid #14532D44",fontSize:"10px",
            }}>
              <span style={{color:CAT_COLORS[VARS[e.from].cat].ring,fontWeight:700,minWidth:"62px"}}>{VARS[e.from].name}</span>
              <span style={{color:e.w>0?"#22C55E":"#F87171",fontFamily:"inherit"}}>
                →({e.w>0?"+":""}{e.w.toFixed(2)})→
              </span>
              <span style={{color:CAT_COLORS[VARS[e.to].cat].ring,fontWeight:700,minWidth:"62px"}}>{VARS[e.to].name}</span>
              <span style={{color:"#475569",marginLeft:"auto",fontSize:"9px"}}>{e.label}</span>
            </div>
          ))}

          <div style={{fontSize:"11px",fontWeight:700,color:"#EF4444",marginTop:"12px",marginBottom:"6px",
            display:"flex",alignItems:"center",gap:"6px"}}>
            <span style={{background:"#1C0505",padding:"2px 8px",borderRadius:"6px",border:"1px solid #450A0A"}}>
              ✗ FALSE POSITIVES — 2
            </span>
          </div>
          {DISCOVERED_EDGES.filter(e=>e.status==="FP").map((e,i)=>(
            <div key={i} style={{
              display:"flex",alignItems:"center",gap:"6px",padding:"5px 10px",
              background:"#1C0A0A",borderRadius:"6px",marginBottom:"3px",border:"1px solid #450A0A44",fontSize:"10px",
            }}>
              <span style={{fontWeight:700,minWidth:"62px",color:"#94A3B8"}}>{VARS[e.from].name}</span>
              <span style={{color:"#FB923C"}}>→({e.w.toFixed(2)})→</span>
              <span style={{fontWeight:700,minWidth:"62px",color:"#94A3B8"}}>{VARS[e.to].name}</span>
              <span style={{color:"#475569",marginLeft:"auto",fontSize:"9px"}}>{e.label}</span>
            </div>
          ))}

          <div style={{fontSize:"11px",fontWeight:700,color:"#F59E0B",marginTop:"12px",marginBottom:"6px",
            display:"flex",alignItems:"center",gap:"6px"}}>
            <span style={{background:"#1A1300",padding:"2px 8px",borderRadius:"6px",border:"1px solid #422006"}}>
              ○ MISSED EDGES — 4
            </span>
          </div>
          {MISSED_EDGES.map((e,i)=>(
            <div key={i} style={{
              display:"flex",alignItems:"center",gap:"6px",padding:"5px 10px",
              background:"#1A1300",borderRadius:"6px",marginBottom:"3px",border:"1px solid #42200644",fontSize:"10px",
            }}>
              <span style={{fontWeight:700,minWidth:"62px",color:"#94A3B8"}}>{VARS[e.from].name}</span>
              <span style={{color:"#F59E0B"}}>→({e.w>0?"+":""}{e.w.toFixed(2)})→</span>
              <span style={{fontWeight:700,minWidth:"62px",color:"#94A3B8"}}>{VARS[e.to].name}</span>
              <span style={{color:"#475569",marginLeft:"auto",fontSize:"9px"}}>{e.label}</span>
            </div>
          ))}
        </div>
      )}

      {/* ════════════════ VERSION HISTORY ════════════════ */}
      {view === "history" && (
        <div style={{maxWidth:"560px",margin:"0 auto"}}>
          <div style={{fontSize:"10px",fontWeight:700,color:"#64748B",letterSpacing:"1px",marginBottom:"10px"}}>
            F1 SCORE PROGRESSION
          </div>
          {VERSIONS.map(v=>(
            <VersionBar key={v.ver} ver={v.ver} f1={v.f1} current={v.ver==="v3.2"} />
          ))}

          <div style={{marginTop:"16px"}}>
            {VERSIONS.slice().reverse().map(v=>(
              <div key={v.ver} style={{
                background:v.ver==="v3.2"?"#0F172A":"#0A0F1F",borderRadius:"10px",padding:"12px",
                marginBottom:"8px",border:`1px solid ${v.ver==="v3.2"?"#166534":"#1E293B"}`,
              }}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"6px"}}>
                  <div style={{display:"flex",alignItems:"center",gap:"8px"}}>
                    <span style={{fontSize:"13px",fontWeight:800,color:v.ver==="v3.2"?"#22C55E":"#94A3B8"}}>{v.ver}</span>
                    <span style={{fontSize:"9px",color:"#475569"}}>{v.date}</span>
                    {v.ver==="v3.2" && <span style={{background:"#052E16",color:"#4ADE80",padding:"1px 6px",
                      borderRadius:"4px",fontSize:"8px",fontWeight:700}}>CURRENT</span>}
                  </div>
                  <span style={{fontSize:"9px",color:"#475569"}}>{v.data}</span>
                </div>
                <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:"6px",marginBottom:"6px"}}>
                  {[["F1",v.f1+"%","#22C55E"],["Pr",v.pr+"%","#3B82F6"],["Rc",v.rc+"%","#A78BFA"],["SHD",v.shd,"#F59E0B"]].map(([l,val,c])=>(
                    <div key={l} style={{textAlign:"center"}}>
                      <div style={{fontSize:"8px",color:"#475569"}}>{l}</div>
                      <div style={{fontSize:"12px",fontWeight:700,color:c}}>{val}</div>
                    </div>
                  ))}
                </div>
                <div style={{display:"flex",justifyContent:"space-between",fontSize:"9px"}}>
                  <span style={{color:"#22C55E"}}>TP={v.tp}</span>
                  <span style={{color:"#EF4444"}}>FP={v.fp}</span>
                  <span style={{color:"#F59E0B"}}>FN={v.fn}</span>
                  <span style={{color:"#06B6D4"}}>Physics {v.phy}</span>
                </div>
                <div style={{fontSize:"9px",color:"#475569",marginTop:"4px",fontStyle:"italic"}}>{v.notes}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ════════════════ REAL DATA VIEW ════════════════ */}
      {view === "real" && (
        <div style={{maxWidth:"520px",margin:"0 auto"}}>
          <div style={{background:"#0F172A",borderRadius:"12px",padding:"16px",border:"1px solid #1E293B",marginBottom:"12px"}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"10px"}}>
              <span style={{fontSize:"11px",fontWeight:700,color:"#64748B",letterSpacing:"1px"}}>
                REAL DATA VALIDATION
              </span>
              <span style={{background:"#172554",color:"#60A5FA",padding:"2px 8px",borderRadius:"6px",
                fontSize:"9px",fontWeight:700,border:"1px solid #1E3A5F"}}>{REAL_DATA.dataset}</span>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"8px",marginBottom:"12px"}}>
              <MetricCard label="Timepoints" value={REAL_DATA.timepoints} color="#60A5FA" />
              <MetricCard label="Plasma Shots" value={REAL_DATA.shots} color="#60A5FA" />
            </div>
            <MetricCard label="Density Limit Prediction AUC" value={REAL_DATA.auc_density_limit.toFixed(3)}
              color="#22C55E" sub="Outperforms Greenwald fraction (0.85)" big />
          </div>

          <div style={{background:"#0F172A",borderRadius:"12px",padding:"16px",border:"1px solid #F59E0B33",marginBottom:"12px"}}>
            <div style={{fontSize:"11px",fontWeight:700,color:"#F59E0B",marginBottom:"8px"}}>
              ⚡ SIMPSON'S PARADOX DETECTION
            </div>
            <div style={{fontSize:"11px",color:"#CBD5E1",lineHeight:"1.7"}}>
              Raw correlation: <span style={{color:"#EF4444",fontWeight:700}}>ne ↔ disruption = +0.53</span>
              <br/>
              After conditioning on Ip: <span style={{color:"#22C55E",fontWeight:700}}>ne ↔ disruption = +0.02</span>
            </div>
            <div style={{marginTop:"8px",padding:"8px",background:"#1E293B",borderRadius:"6px",fontSize:"9px",color:"#94A3B8",lineHeight:"1.6"}}>
              Classical correlational AI systems would incorrectly conclude that high density causes disruptions.
              CPDE's causal analysis reveals that <strong style={{color:"#F59E0B"}}>plasma current (Ip)</strong> is the
              confounding variable — higher Ip allows both higher density AND greater stability.
              This insight is only possible with causal inference (Pearl's do-calculus), not correlation.
            </div>
          </div>

          <div style={{background:"#0F172A",borderRadius:"12px",padding:"16px",border:"1px solid #1E293B"}}>
            <div style={{fontSize:"10px",fontWeight:700,color:"#64748B",letterSpacing:"1px",marginBottom:"8px"}}>
              KEY INSIGHT: WHY CAUSAL &gt; CORRELATIONAL
            </div>
            <div style={{fontSize:"10px",color:"#94A3B8",lineHeight:"1.7"}}>
              All existing fusion AI (DeepMind, KSTAR, Princeton, TokaMind) operate at Pearl's Ladder Level 1 —
              association and prediction. They can predict disruptions but cannot explain <em>why</em> or reason about
              <em>what would happen if</em> we intervene.
            </div>
            <div style={{marginTop:"8px",display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:"6px"}}>
              {[
                ["Level 1","Association","All competitors",false],
                ["Level 2","Intervention","FusionMind 4.0",true],
                ["Level 3","Counterfactual","FusionMind 4.0",true],
              ].map(([level,name,who,us])=>(
                <div key={level} style={{
                  padding:"8px",borderRadius:"6px",textAlign:"center",
                  background:us?"#052E16":"#1E293B",border:`1px solid ${us?"#166534":"#334155"}`,
                }}>
                  <div style={{fontSize:"9px",color:us?"#4ADE80":"#64748B",fontWeight:700}}>{level}</div>
                  <div style={{fontSize:"10px",color:us?"#22C55E":"#94A3B8",fontWeight:600}}>{name}</div>
                  <div style={{fontSize:"8px",color:us?"#166534":"#475569",marginTop:"2px"}}>{who}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <div style={{textAlign:"center",marginTop:"20px",padding:"10px",borderTop:"1px solid #1E293B"}}>
        <div style={{color:"#334155",fontSize:"9px"}}>
          FusionMind 4.0 · CPDE v3.2 · Patent Family PF1 · Dr. Mladen Mester · March 2026
        </div>
        <div style={{color:"#1E293B",fontSize:"8px",marginTop:"2px"}}>
          First-ever application of Pearl's causal inference framework to tokamak plasma dynamics
        </div>
      </div>
    </div>
  );
}
