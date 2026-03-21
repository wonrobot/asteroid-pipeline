import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist";

const BC = { g:"#22c55e", r:"#f97316", i:"#3b82f6" };
const BAND_FULL = { g:"g-band (green)", r:"r-band (red)", i:"i-band (infrared)" };

function nightMap(mjd) {
  const days = [...new Set(mjd.map(m=>Math.floor(m)))].sort((a,b)=>a-b);
  const map = {}; days.forEach((d,i)=>{map[d]=i;});
  return mjd.map(m=>map[Math.floor(m)]);
}

function Plot({ traces, layout, height=280 }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    try { Plotly.react(ref.current, traces, layout, {displayModeBar:false, responsive:true}); }
    catch(e) { console.warn("Plotly",e); }
  }, [traces, layout]);
  return <div ref={ref} style={{width:"100%", height}} />;
}

const BASE = {
  paper_bgcolor:"transparent", plot_bgcolor:"#fafbff",
  font:{family:"Inter,sans-serif", color:"#475569", size:11},
  legend:{bgcolor:"rgba(255,255,255,0.9)", bordercolor:"#e2e8f0", borderwidth:1, font:{size:10}},
  hovermode:"closest",
};

function makeMarginLayout(t=40,b=48,l=58,r=16) {
  return {margin:{t,b,l,r}};
}

// Vertical marker line as scatter trace
function vline(x, name, color, dash="solid") {
  return {x:[x,x], y:[0,1], yaxis_ref:"paper", mode:"lines",
    line:{color,width:1.8,dash}, name, type:"scatter", showlegend:true,
    hovertemplate:`${name}: ${typeof x==="number"?x.toFixed(3)+"h":x}<extra></extra>`};
}

// Build shared period marker traces
function periodMarkers(P, periods) {
  if (!P) return [];
  const mn = Math.min(...periods), mx = Math.max(...periods);
  const mk = (x,name,color,dash) => x>=mn&&x<=mx ? vline(x,name,color,dash) : null;
  return [
    mk(P,    `P=${P.toFixed(3)}h`,       "#2563eb","solid"),
    mk(P/2,  `P/2=${(P/2).toFixed(3)}h`, "#f59e0b","dot"),
    mk(P*2,  `2P=${(P*2).toFixed(3)}h`,  "#f59e0b","dot"),
    mk(12,   "12h alias",                "#dc2626","dashdot"),
    mk(24,   "24h alias",                "#dc2626","dashdot"),
  ].filter(Boolean);
}

export default function AsteroidCharts({ provid, period }) {
  const [d, setD]     = useState(null);
  const [err, setErr] = useState(false);
  const [tab, setTab] = useState("lc");

  useEffect(() => {
    if (!provid) return;
    setD(null); setErr(false); setTab("lc");
    fetch(`/data/plots/data_${provid.replace(/\s+/g,"_")}.json`)
      .then(r=>{if(!r.ok)throw 0;return r.json();})
      .then(setD).catch(()=>setErr(true));
  }, [provid]);

  if (!provid) return null;
  if (err) return <p style={{color:"#94a3b8",fontSize:"0.78rem",marginTop:12}}>No chart data available.</p>;
  if (!d)  return <p style={{color:"#94a3b8",fontSize:"0.78rem",marginTop:12}}>Loading…</p>;

  const ni    = nightMap(d.obs.mjd);
  const bands = [...new Set(d.obs.band)];
  const P     = d.best_period;
  const pg    = d.pgram;

  /* ── Lightcurve: colored by BAND, symbol by night ── */
  const nightSymbols = ["circle","square","diamond","triangle-up","triangle-down","pentagon","hexagon","star","cross","x"];
  const nights = [...new Set(ni)];
  const lcTraces = bands.map(b => {
    // Split by night for symbol variation
    const subTraces = nights.map(n => {
      const idx = ni.map((v,i)=>v===n&&d.obs.band[i]===b?i:-1).filter(i=>i>=0);
      if (!idx.length) return null;
      return {
        x: idx.map(i=>d.obs.mjd[i]), y: idx.map(i=>d.obs.mag[i]),
        error_y:{type:"data",array:idx.map(i=>d.obs.magerr[i]),visible:true,color:BC[b]+"55",thickness:1},
        mode:"markers",
        marker:{color:BC[b], size:4, symbol:nightSymbols[n%10], opacity:0.85},
        name:`${b}-band · Night ${n+1}`, type:"scatter",
        legendgroup:b,
        showlegend: n===nights.filter(nn=>idx.some((_,i)=>ni[idx[0]]===nn))[0],
      };
    }).filter(Boolean);
    return subTraces;
  }).flat();

  // Simpler: one trace per band (all nights merged), night shown by opacity gradient
  const lcSimple = bands.map(b => ({
    x: d.obs.mjd.filter((_,i)=>d.obs.band[i]===b),
    y: d.obs.mag.filter((_,i)=>d.obs.band[i]===b),
    error_y:{type:"data",
      array:d.obs.magerr.filter((_,i)=>d.obs.band[i]===b),
      visible:true, color:BC[b]+"55", thickness:1},
    mode:"markers", marker:{color:BC[b], size:4, opacity:0.85},
    name:BAND_FULL[b]||b, type:"scatter",
  }));

  const lcLayout = {
    ...BASE, ...makeMarginLayout(),
    title:{text:"Lightcurve", font:{size:12,color:"#0f172a"}},
    xaxis:{title:{text:"MJD"},gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
    yaxis:{title:{text:"mag"},autorange:"reversed",gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
  };

  /* ── Periodogram — 3 stacked panels sharing period markers ── */
  const markers = periodMarkers(P, pg.periods);
  const xaxis_shared = {title:{text:"Period (hr)"},type:"log",gridcolor:"#e8edf5",linecolor:"#cbd5e1",
    range: pg.periods ? [Math.log10(Math.min(...pg.periods)), Math.log10(Math.max(...pg.periods))] : undefined};

  // Panel 1 — Tier 1 fast scan
  const t1max = Math.max(...pg.mbls_power);
  const t1Traces = [
    {x:pg.gls_periods||pg.periods, y:pg.gls_power||[], mode:"lines", line:{color:"#059669",width:1,dash:"dash"}, name:"GLS", type:"scatter"},
    {x:pg.periods, y:pg.mbls_power, mode:"lines", line:{color:"#2563eb",width:1.2}, name:"MBLS (T1)", type:"scatter"},
    ...markers,
  ];

  // Panel 2 — Tier 2
  const t2Traces = [];
  if (pg.mhaov_power) {
    const s = t1max/Math.max(...pg.mhaov_power)*0.9;
    t2Traces.push({x:pg.periods, y:pg.mhaov_power.map(v=>v*s), mode:"lines", line:{color:"#7c3aed",width:1.2}, name:"MHAOV (norm.)", type:"scatter"});
  }
  const t2mbls = pg.mbls_power; // already on same scale
  t2Traces.push({x:pg.periods, y:t2mbls, mode:"lines", line:{color:"#2563eb",width:1}, name:"MBLS (T2)", type:"scatter"});
  t2Traces.push(...markers);

  // Panel 3 — CE / window
  const t3Traces = [];
  if (pg.ce_periods && pg.ce_scores) {
    t3Traces.push({x:pg.ce_periods, y:pg.ce_scores, mode:"lines", line:{color:"#0891b2",width:1.2}, name:"Conditional Entropy", type:"scatter"});
  }
  if (pg.window_periods && pg.window_power) {
    const wmax = Math.max(...pg.window_power);
    t3Traces.push({x:pg.window_periods, y:pg.window_power.map(v=>v/wmax*t1max*0.5),
      mode:"lines", line:{color:"#f97316",width:0.8}, fill:"tozeroy", fillcolor:"rgba(249,115,22,0.06)",
      name:"Window fn (norm.)", type:"scatter"});
  }
  if (t3Traces.length === 0) {
    t3Traces.push({x:[],y:[], type:"scatter", name:"No T3 data"});
  }
  t3Traces.push(...markers);

  const pgramBase = {...BASE, ...makeMarginLayout(28,40,58,16)};

  /* ── Phase fold ── */
  const foldTraces = d.fold ? [
    ...bands.map(b=>({
      x:d.fold.phase.filter((_,i)=>d.fold.band[i]===b),
      y:d.fold.mag.filter((_,i)=>d.fold.band[i]===b),
      error_y:{type:"data",array:d.fold.magerr.filter((_,i)=>d.fold.band[i]===b),visible:true,color:BC[b]+"55",thickness:1},
      mode:"markers", marker:{color:BC[b],size:3.5,opacity:0.85},
      name:BAND_FULL[b]||b, type:"scatter",
    })),
    {x:d.fold.fitted_phase, y:d.fold.fitted_mag, mode:"lines",
      line:{color:"#0f172a",width:2}, name:"model", type:"scatter"},
  ] : null;

  const residTraces = d.fold ? bands.map(b=>{
    const fp=d.fold.fitted_phase, fm=d.fold.fitted_mag;
    const interp=ph=>fm[fp.reduce((bi,_,i)=>Math.abs(fp[i]-ph)<Math.abs(fp[bi]-ph)?i:bi,0)];
    const phases=d.fold.phase.filter((_,i)=>d.fold.band[i]===b);
    const mags=d.fold.mag.filter((_,i)=>d.fold.band[i]===b);
    return {x:phases, y:mags.map((m,i)=>m-interp(phases[i])),
      mode:"markers", marker:{color:BC[b],size:3,opacity:0.7},
      name:b, type:"scatter", showlegend:false};
  }) : null;

  const tabs = [
    {id:"lc",    label:"Lightcurve"},
    {id:"pgram", label:"Periodogram"},
    ...(d.fold?[{id:"fold",label:"Phase Fold"}]:[]),
  ];

  const S = {
    bar:{display:"flex",gap:4,alignItems:"center",marginBottom:8,flexWrap:"wrap"},
    tab:{fontFamily:"Inter,sans-serif",fontSize:"0.72rem",fontWeight:500,padding:"4px 14px",
      border:"1px solid #dde3f0",borderRadius:5,background:"#f7f8fc",color:"#475569",cursor:"pointer"},
    on:{background:"#2563eb",color:"#fff",borderColor:"#2563eb"},
    pill:{marginLeft:"auto",fontFamily:"JetBrains Mono,monospace",fontSize:"0.72rem",color:"#2563eb",
      background:"#dbeafe",padding:"3px 10px",borderRadius:4,fontWeight:600},
    pLabel:{fontSize:"0.65rem",color:"#94a3b8",padding:"2px 6px",background:"#f1f5f9",borderRadius:3,fontFamily:"JetBrains Mono,monospace"},
    pgTitle:{fontSize:"0.68rem",fontWeight:600,color:"#475569",textTransform:"uppercase",
      letterSpacing:"0.06em",padding:"6px 0 2px 4px"},
    divider:{borderBottom:"1px dashed #e2e8f0",margin:"2px 0"},
  };

  return (
    <div style={{marginTop:"1rem"}}>
      <div style={S.bar}>
        {tabs.map(t=><button key={t.id} style={{...S.tab,...(tab===t.id?S.on:{})}} onClick={()=>setTab(t.id)}>{t.label}</button>)}
        {P && <span style={S.pill}>P = {P.toFixed(4)} hr</span>}
      </div>

      {tab==="lc" && <>
        <Plot traces={lcSimple} layout={lcLayout} height={290}/>
        <div style={{fontSize:"0.7rem",color:"#94a3b8",padding:"4px 8px",display:"flex",gap:16,flexWrap:"wrap"}}>
          {bands.map(b=><span key={b} style={{display:"flex",alignItems:"center",gap:4}}>
            <span style={{width:10,height:10,borderRadius:"50%",background:BC[b],display:"inline-block"}}/>
            {BAND_FULL[b]||b}
          </span>)}
          <span style={{marginLeft:"auto"}}>Symbol shape varies by night</span>
        </div>
      </>}

      {tab==="pgram" && <>
        <div style={S.pgTitle}>Tier 1 — Fast scan (GLS · MBLS)</div>
        <Plot traces={t1Traces} layout={{...pgramBase,
          title:{text:"", font:{size:11}},
          xaxis:{...xaxis_shared, showticklabels:false, title:{text:""}},
          yaxis:{title:{text:"Power"},gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
        }} height={180}/>
        <div style={S.divider}/>
        <div style={S.pgTitle}>Tier 2 — High-order (MHAOV · MBLS)</div>
        <Plot traces={t2Traces} layout={{...pgramBase,
          xaxis:{...xaxis_shared, showticklabels:false, title:{text:""}},
          yaxis:{title:{text:"Power"},gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
        }} height={180}/>
        <div style={S.divider}/>
        <div style={S.pgTitle}>Tier 3 — {pg.ce_periods?"Conditional Entropy (CE)":"Window function"}</div>
        <Plot traces={t3Traces} layout={{...pgramBase, ...makeMarginLayout(28,48,58,16),
          xaxis:{...xaxis_shared, title:{text:"Period (hr)"}},
          yaxis:{title:{text:pg.ce_periods?"CE score":"Power"},gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
        }} height={180}/>
        <div style={{fontSize:"0.68rem",color:"#94a3b8",padding:"4px 8px",display:"flex",gap:16,flexWrap:"wrap"}}>
          <span><span style={{color:"#2563eb"}}>━</span> P={P?.toFixed(3)}h</span>
          <span><span style={{color:"#f59e0b"}}>┄</span> P/2, 2P</span>
          <span><span style={{color:"#dc2626"}}>╌╌</span> 12h/24h aliases</span>
          <span><span style={{color:"#7c3aed"}}>┅</span> MHAOV</span>
          <span><span style={{color:"#059669"}}>╌</span> GLS</span>
          <span><span style={{color:"#0891b2"}}>━</span> CE</span>
        </div>
      </>}

      {tab==="fold" && d.fold && <>
        <Plot traces={foldTraces} layout={{...BASE,...makeMarginLayout(),
          title:{text:`Phase fold — 2 cycles  (P = ${P?.toFixed(4)} hr)`,font:{size:12,color:"#0f172a"}},
          xaxis:{title:{text:"Phase"},range:[0,2],dtick:0.25,gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
          yaxis:{title:{text:"Δmag (detrended)"},autorange:"reversed",gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
        }} height={270}/>
        <Plot traces={residTraces} layout={{...BASE,...makeMarginLayout(24,44,58,16),
          title:{text:"Residuals (O−C)",font:{size:11,color:"#0f172a"}},
          xaxis:{title:{text:"Phase"},range:[0,2],dtick:0.25,gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
          yaxis:{title:{text:"O−C (mag)"},zeroline:true,zerolinecolor:"#475569",gridcolor:"#e8edf5"},
        }} height={150}/>
      </>}
    </div>
  );
}
