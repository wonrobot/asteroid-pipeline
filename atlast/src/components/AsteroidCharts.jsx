import { useEffect, useState, useRef } from "react";
import Plotly from "plotly.js-dist";

const BAND_COLOR  = { g: "#22c55e", r: "#f97316", i: "#3b82f6" };
const BAND_COLOR2 = { g: "#bbf7d0", r: "#fed7aa", i: "#bfdbfe" };
const ALIAS_HRS   = [12, 24, 48, 4380, 8760];

// 20 distinct colors for nights
const NIGHT_COLORS = [
  "#e63946","#f4a261","#2a9d8f","#457b9d","#e9c46a",
  "#6d6875","#a8dadc","#f77f00","#4cc9f0","#7209b7",
  "#b5e48c","#ff6b6b","#48cae4","#f3722c","#90be6d",
  "#43aa8b","#577590","#f9844a","#4d908e","#277da1",
];

function getNightMap(mjd) {
  const nights = {};
  let nid = 0;
  const sorted = [...new Set(mjd.map(m => Math.floor(m)))].sort((a,b)=>a-b);
  sorted.forEach(d => { nights[d] = nid++; });
  return mjd.map(m => nights[Math.floor(m)]);
}

function PlotDiv({ data, layout, style }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !data) return;
    const base = {
      paper_bgcolor: "transparent",
      plot_bgcolor:  "#fafbff",
      font: { family: "Inter, sans-serif", color: "#475569", size: 11 },
      margin: { t: 40, b: 52, l: 62, r: 20 },
      xaxis: { gridcolor: "#e8edf5", zerolinecolor: "#cbd5e1", linecolor: "#cbd5e1" },
      yaxis: { gridcolor: "#e8edf5", zerolinecolor: "#cbd5e1", linecolor: "#cbd5e1" },
      legend: { bgcolor: "rgba(255,255,255,0.85)", bordercolor: "#e2e8f0", borderwidth: 1, font: { size: 10 } },
      hovermode: "closest",
    };
    Plotly.react(ref.current, data, { ...base, ...layout,
      xaxis: { ...base.xaxis, ...(layout.xaxis||{}) },
      yaxis: { ...base.yaxis, ...(layout.yaxis||{}) },
      yaxis2: layout.yaxis2 || undefined,
    }, { displayModeBar: false, responsive: true });
  }, [data, layout]);
  return <div ref={ref} style={{ width: "100%", ...style }} />;
}

function vline(x, label, color, dash="dash") {
  return {
    type: "line", x0: x, x1: x, y0: 0, y1: 1, yref: "paper",
    line: { color, width: 1.5, dash },
    opacity: 0.85,
  };
}

function aliasShapes(best, periods) {
  const pmin = Math.min(...periods), pmax = Math.max(...periods);
  const shapes = [];
  if (!best) return shapes;
  // P/2 and 2P
  [[best/2, "P/2", "#f59e0b"], [best*2, "2P", "#f59e0b"]].forEach(([p, , c]) => {
    if (p >= pmin && p <= pmax) shapes.push(vline(p, "", c, "dot"));
  });
  // Known calendar aliases
  ALIAS_HRS.forEach(p => {
    if (p >= pmin && p <= pmax) shapes.push(vline(p, "", "#dc2626", "dashdot"));
  });
  // Best period
  if (best >= pmin && best <= pmax) shapes.push(vline(best, "", "#2563eb", "solid"));
  return shapes;
}


export default function AsteroidCharts({ provid }) {
  const [d, setD]     = useState(null);
  const [err, setErr] = useState(false);
  const [tab, setTab] = useState("lc");

  useEffect(() => {
    if (!provid) return;
    setD(null); setErr(false); setTab("lc");
    fetch(`/data/plots/data_${provid.replace(/\s+/g,"_")}.json`)
      .then(r => { if(!r.ok) throw 0; return r.json(); })
      .then(setD).catch(() => setErr(true));
  }, [provid]);

  if (!provid) return null;
  if (err) return <p style={S.msg}>No chart data available.</p>;
  if (!d)  return <p style={S.msg}>Loading charts…</p>;

  
  const nightIdx = getNightMap(d.obs.mjd);
  const nights   = [...new Set(nightIdx)];
  const bands    = [...new Set(d.obs.band)];

  /* ── Lightcurve: one trace per night, colored by night ── */
  const lcTraces = nights.map(ni => {
    const mask = nightIdx.map((n,i) => n===ni ? i : -1).filter(i=>i>=0);
    return {
      x: mask.map(i => d.obs.mjd[i]),
      y: mask.map(i => d.obs.mag[i]),
      error_y: { type:"data", array: mask.map(i=>d.obs.magerr[i]), visible:true, color: NIGHT_COLORS[ni%20]+"66", thickness:1 },
      mode: "markers",
      marker: { color: NIGHT_COLORS[ni%20], size: 4, symbol: mask.map(i => d.obs.band[i]==="g"?"circle":d.obs.band[i]==="r"?"square":"diamond") },
      name: `Night ${ni+1}`,
      type: "scatter",
    };
  });
  // Add detrended baseline hint
  lcTraces.push({
    x: d.obs.mjd, y: d.obs.baseline_mag,
    mode:"markers", marker:{ color:"#94a3b8", size:2.5, opacity:0.4 },
    name:"detrended", type:"scatter", visible:"legendonly",
  });

  /* ── Periodogram: multi-method ── */
  const pg = d.pgram;
  const maxMBLS   = Math.max(...pg.mbls_power);
  const maxMHAOV  = pg.mhaov_power ? Math.max(...pg.mhaov_power) : 1;
  const pgramTraces = [
    { x: pg.periods, y: pg.mbls_power, mode:"lines", line:{color:"#2563eb",width:1.2}, name:"MBLS", type:"scatter" },
  ];
  if (pg.mhaov_power) {
    // Normalise MHAOV to same scale as MBLS for overlay
    const norm = pg.mhaov_power.map(v => v / maxMHAOV * maxMBLS * 0.85);
    pgramTraces.push({ x: pg.periods, y: norm, mode:"lines", line:{color:"#7c3aed",width:1,dash:"dot"}, name:"MHAOV (norm.)", type:"scatter" });
  }
  if (pg.gls_power) {
    const maxGLS = Math.max(...pg.gls_power);
    const norm = pg.gls_power.map(v => v / maxGLS * maxMBLS * 0.7);
    pgramTraces.push({ x: pg.gls_periods, y: norm, mode:"lines", line:{color:"#059669",width:0.8,dash:"dash"}, name:"GLS (norm.)", type:"scatter" });
  }
  if (pg.window_power) {
    const maxW = Math.max(...pg.window_power);
    const norm = pg.window_power.map(v => v / maxW * maxMBLS * 0.35);
    pgramTraces.push({ x: pg.window_periods, y: norm, mode:"lines", line:{color:"#f97316",width:0.7}, name:"Window fn (norm.)", type:"scatter", fill:"tozeroy", fillcolor:"rgba(249,115,22,0.07)" });
  }

  const pgramLayout = {
    title: { text: d.best_period ? `Periodogram — P=${d.best_period.toFixed(3)}h  |  P/2=${(d.best_period/2).toFixed(3)}h  |  2P=${(d.best_period*2).toFixed(3)}h` : 'Periodogram', font:{size:12} },
    xaxis: { title:{text:"Period (hr)"}, type:"log", gridcolor:"#e8edf5" },
    yaxis: { title:{text:"MBLS Power"}, gridcolor:"#e8edf5" },
    shapes: aliasShapes(d.best_period, pg.periods),

  };

  /* ── Phase fold: 2-cycle with residuals ── */
  let foldTraces = null, residTraces = null;
  if (d.fold) {
    const fp = d.fold.fitted_phase, fm = d.fold.fitted_mag;
    foldTraces = bands.map(b => ({
      x: d.fold.phase.filter((_,i)=>d.fold.band[i]===b),
      y: d.fold.mag.filter((_,i)=>d.fold.band[i]===b),
      error_y: { type:"data", array: d.fold.magerr.filter((_,i)=>d.fold.band[i]===b), visible:true, color:BAND_COLOR2[b], thickness:1 },
      mode:"markers", marker:{color:BAND_COLOR[b],size:3.5,opacity:0.8},
      name:b+"-band", type:"scatter",
    }));
    foldTraces.push({ x:fp, y:fm, mode:"lines", line:{color:"#0f172a",width:2}, name:"model", type:"scatter" });

    // Residuals (observed - model, interpolated)
    // Simple: just show error bars at each point as residual proxy
    const interp = (phase) => {
      const idx = fp.reduce((bi,_,i)=>Math.abs(fp[i]-phase)<Math.abs(fp[bi]-phase)?i:bi, 0);
      return fm[idx];
    };
    residTraces = bands.map(b => {
      const phases = d.fold.phase.filter((_,i)=>d.fold.band[i]===b);
      const mags   = d.fold.mag.filter((_,i)=>d.fold.band[i]===b);
      const resids = mags.map((m,i) => m - interp(phases[i]));
      return {
        x: phases, y: resids,
        mode:"markers", marker:{color:BAND_COLOR[b],size:3,opacity:0.7},
        name:b, type:"scatter", showlegend:false,
      };
    });
  }

  const tabs = [
    { id:"lc",    label:"Lightcurve" },
    { id:"pgram", label:"Periodogram" },
    ...(d.fold ? [{ id:"fold", label:"Phase Fold" }] : []),
  ];

  return (
    <div style={{ marginTop:"1rem" }}>
      {/* Tab bar */}
      <div style={S.tabBar}>
        {tabs.map(t => (
          <button key={t.id} style={{...S.tab, ...(tab===t.id?S.tabOn:{})}} onClick={()=>setTab(t.id)}>
            {t.label}
          </button>
        ))}
        {d.best_period && (
          <span style={S.periodPill}>P = {d.best_period.toFixed(4)} hr</span>
        )}
        <span style={S.legend}>
          <span style={{color:"#2563eb"}}>━</span> MBLS &nbsp;
          <span style={{color:"#7c3aed"}}>┅</span> MHAOV &nbsp;
          <span style={{color:"#059669"}}>╌</span> GLS &nbsp;
          <span style={{color:"#f59e0b"}}>┄</span> P/2, 2P &nbsp;
          <span style={{color:"#dc2626"}}>╌╌</span> aliases
        </span>
      </div>

      {tab === "lc" && (
        <PlotDiv data={lcTraces} style={{height:300}} layout={{
          title:{text:"Lightcurve — colored by night, symbol by band (○g □r ◇i)", font:{size:12}},
          xaxis:{title:{text:"MJD"}},
          yaxis:{title:{text:"mag"},autorange:"reversed"},
        }}/>
      )}

      {tab === "pgram" && (
        <PlotDiv data={pgramTraces} style={{height:320}} layout={pgramLayout} />
      )}

      {tab === "fold" && d.fold && (
        <div>
          <PlotDiv data={foldTraces} style={{height:260}} layout={{
            title:{text:`Phase fold — 2 cycles  (P = ${d.best_period?.toFixed(4)} hr)`, font:{size:12}},
            xaxis:{title:{text:"Phase"},range:[0,2],dtick:0.25,gridcolor:"#e8edf5"},
            yaxis:{title:{text:"Δmag (detrended)"},autorange:"reversed"},
            shapes:[{type:"line",x0:1,x1:1,y0:0,y1:1,yref:"paper",line:{color:"#94a3b8",width:1,dash:"dot"}}],
          }}/>
          <PlotDiv data={residTraces} style={{height:140}} layout={{
            title:{text:"Residuals", font:{size:11}},
            xaxis:{title:{text:"Phase"},range:[0,2],dtick:0.25,gridcolor:"#e8edf5"},
            yaxis:{title:{text:"O−C (mag)"},zeroline:true,zerolinecolor:"#475569"},
            margin:{t:28,b:44,l:62,r:20},
          }}/>
        </div>
      )}
    </div>
  );
}

const S = {
  msg:       { color:"#94a3b8", fontSize:"0.78rem", marginTop:12 },
  tabBar:    { display:"flex", gap:"4px", alignItems:"center", marginBottom:"0.5rem", flexWrap:"wrap" },
  tab:       { fontFamily:"Inter,sans-serif", fontSize:"0.72rem", fontWeight:500, padding:"4px 14px", border:"1px solid #dde3f0", borderRadius:5, background:"#f7f8fc", color:"#475569", cursor:"pointer" },
  tabOn:     { background:"#2563eb", color:"#fff", borderColor:"#2563eb" },
  periodPill:{ marginLeft:"auto", fontFamily:"JetBrains Mono,monospace", fontSize:"0.72rem", color:"#2563eb", background:"#dbeafe", padding:"3px 10px", borderRadius:4, fontWeight:600 },
  legend:    { fontSize:"0.65rem", color:"#94a3b8", marginLeft:8 },
};
