import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist";

const BC = { g:"#22c55e", r:"#f97316", i:"#3b82f6" };
const BAND_FULL = { g:"g-band (green)", r:"r-band (orange)", i:"i-band (blue)" };

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
  showlegend: false,
  hovermode:"closest",
};

const PLOT_MARGIN = {t:24, b:44, l:58, r:24};

// Log axis config with clean tick formatting
const LOG_XAXIS = () => ({
  title:{text:"Period (hr)"},
  type:"log",
  gridcolor:"#e8edf5", linecolor:"#cbd5e1",
  tickformat:".3~g",
  range:[Math.log10(0.3), Math.log10(25)],
  tickvals:[0.5,1,2,3,4,5,6,7,8,9,10,12,15,20,24],
  ticktext:["0.5","1","2","3","4","5","6","7","8","9","10","12","15","20","24"],
});

// Period marker as scatter trace with text label at top
function pline(x, label, color, dash, maxPow) {
  return {
    x:[x, x], y:[0, 1],
    mode:"lines+text",
    line:{color, width: dash==="solid"?2:1.5, dash},
    text:["", label],
    textposition:"top center",
    textfont:{size:9, color},
    name:label, type:"scatter", showlegend:false,
    hovertemplate:`${label}<extra></extra>`,
  };
}

function periodMarkers(P, periods, maxPow) {
  if (!P) return [];
  const mn = Math.min(...periods), mx = Math.max(...periods);
  const mk = (x, label, color, dash) =>
    x>=mn && x<=mx ? pline(x, label, color, dash, maxPow) : null;
  return [
    mk(P,   `P=${P.toFixed(3)}h`,      "#111827","solid"),
    mk(P/2, `P/2=${(P/2).toFixed(3)}h`,"#d97706","dot"),
    mk(P*2, `2P=${(P*2).toFixed(3)}h`, "#d97706","dot"),
  ].filter(Boolean);
}

// Colored text label row (no line icons)
function LegendRow({ items }) {
  return (
    <div style={{display:"flex",gap:14,flexWrap:"wrap",padding:"2px 4px 6px",alignItems:"center"}}>
      {items.map(({color,label})=>(
        <span key={label} style={{fontSize:"0.68rem",fontWeight:600,color,letterSpacing:"0.01em"}}>
          {label}
        </span>
      ))}
    </div>
  );
}

// Stat line below panel
function StatLine({ items }) {
  return (
    <div style={{fontSize:"0.68rem",color:"#64748b",padding:"2px 4px 6px",
      fontFamily:"JetBrains Mono,monospace",display:"flex",gap:20,flexWrap:"wrap"}}>
      {items.map(({label,value})=>(
        <span key={label}>{label}: <strong style={{color:"#0f172a"}}>{value}</strong></span>
      ))}
    </div>
  );
}

// Tier section wrapper
function TierSection({ title, period, children }) {
  return (
    <div style={{marginBottom:4}}>
      <div style={{display:"flex",alignItems:"baseline",gap:10,padding:"6px 4px 0"}}>
        <span style={{fontSize:"0.65rem",fontWeight:700,color:"#475569",
          textTransform:"uppercase",letterSpacing:"0.08em"}}>{title}</span>

      </div>
      {children}
    </div>
  );
}

export default function AsteroidCharts({ provid }) {
  const [d, setD]     = useState(null);
  const [err, setErr] = useState(false);
  const [tab, setTab] = useState("fold");

  useEffect(() => {
    if (!provid) return;
    setD(null); setErr(false); setTab("fold");
    fetch(`/data/plots/data_${provid.replace(/\s+/g,"_")}.json`)
      .then(r=>{if(!r.ok)throw 0;return r.json();})
      .then(setD).catch(()=>setErr(true));
  }, [provid]);

  if (!provid) return null;
  if (err) return <p style={{color:"#94a3b8",fontSize:"0.78rem",marginTop:12}}>No chart data available.</p>;
  if (!d)  return <p style={{color:"#94a3b8",fontSize:"0.78rem",marginTop:12}}>Loading…</p>;

  const bands  = [...new Set(d.obs.band)];
  const P      = d.best_period;
  const pg     = d.pgram;

  /* ── Lightcurve ── */
  const lcTraces = bands.map(b=>({
    x: d.obs.mjd.filter((_,i)=>d.obs.band[i]===b),
    y: d.obs.mag.filter((_,i)=>d.obs.band[i]===b),
    error_y:{type:"data",array:d.obs.magerr.filter((_,i)=>d.obs.band[i]===b),
      visible:true,color:BC[b]+"55",thickness:1},
    mode:"markers", marker:{color:BC[b],size:4,symbol:"circle",opacity:0.85},
    name:BAND_FULL[b]||b, type:"scatter", showlegend:false,
  }));

  /* ── Periodogram shared ── */
  const maxMBLS = Math.max(...pg.mbls_power);
  const normTo1 = (arr, max) => arr.map(v => v / (max || 1));
  const xaxis   = LOG_XAXIS();
  const markers = periodMarkers(P, pg.periods, maxMBLS);

  // Method disagreement lines for R=0
  const METHOD_COLORS = {mhaov:"#9333ea",mbls:"#ea580c",ce:"#0e7490",gls:"#059669"};
  const methodLines = (() => {
    if (P || !d.method_periods) return [];
    const mn=Math.min(...pg.periods), mx=Math.max(...pg.periods);
    return Object.entries(d.method_periods)
      .filter(([,v])=>v&&v>=mn&&v<=mx)
      .map(([name,val])=>pline(val,`${name.toUpperCase()} ${val.toFixed(3)}h`,
        METHOD_COLORS[name]||"#94a3b8","dashdot",maxMBLS));
  })();

  /* T1 — Generalised Lomb-Scargle + Multi-Band Lomb-Scargle */
  const maxGLS = pg.gls_power ? Math.max(...pg.gls_power) : 1;
  const t1Traces = [
    ...(P?markers:methodLines),
    {x:pg.gls_periods||pg.periods, y:normTo1(pg.gls_power||[], maxGLS), mode:"lines",
      line:{color:"#0d9488",width:1,dash:"dash"}, name:"GLS", type:"scatter",showlegend:false},
    {x:pg.periods, y:normTo1(pg.mbls_power, maxMBLS), mode:"lines",
      line:{color:"#1d4ed8",width:1.3}, name:"MBLS", type:"scatter",showlegend:false},
  ];

  /* T2 — Multi-Harmonic AoV + Multi-Band Lomb-Scargle */
  const maxMHAOV = pg.mhaov_power ? Math.max(...pg.mhaov_power) : 1;
  const t2Traces = [
    ...(P?markers.map(m=>({...m,showlegend:false})):methodLines),
    {x:pg.periods, y:normTo1(pg.mhaov_power||[], maxMHAOV), mode:"lines",
      line:{color:"#9333ea",width:1.3,dash:"dot"}, name:"MHAOV", type:"scatter",showlegend:false},
    {x:pg.periods, y:normTo1(pg.mbls_power, maxMBLS), mode:"lines",
      line:{color:"#ea580c",width:1.3}, name:"MBLS", type:"scatter",showlegend:false},
  ];
  const pval = pg.p_value;

  /* T3 — Conditional Entropy + Window function */
  const maxCE = pg.ce_scores ? Math.max(...pg.ce_scores) : 1;
  const minCE = pg.ce_scores ? Math.min(...pg.ce_scores) : 0;
  const ceDetected = pg.ce_scores && minCE < 0.95; // CE=1.0 means no detection
  const alias12in = 12 >= 0.3 && 12 <= 25;
  const alias24in = 24 >= 0.3 && 24 <= 25; // line at 23.5 to avoid label cutoff
  const t3Traces = [
    ...(P?markers.map(m=>({...m,showlegend:false})):methodLines),
    // 12h/24h alias lines for T3 context
    ...(alias12in?[pline(12,"12h alias","#dc2626","dashdot",maxMBLS)]:[] ),
    ...(alias24in?[pline(23.5,"24h alias","#dc2626","dashdot",maxMBLS)]:[] ),
    ...(pg.window_periods&&pg.window_power ? [{
      x:pg.window_periods,
      y:normTo1(pg.window_power, Math.max(...pg.window_power)),
      mode:"lines", line:{color:"#f59e0b",width:1},
      fill:"tozeroy", fillcolor:"rgba(245,158,11,0.08)",
      name:"Window", type:"scatter",showlegend:false,
    }] : []),
    ...(ceDetected && pg.ce_periods&&pg.ce_scores ? [{
      x:pg.ce_periods, y:normTo1(pg.ce_scores, maxCE),
      mode:"lines", line:{color:"#0e7490",width:1.3},
      name:"CE", type:"scatter",showlegend:false,
    }] : []),
  ];

  /* ── Phase fold ── */
  const foldTraces = d.fold ? [
    ...bands.map(b=>({
      x:d.fold.phase.filter((_,i)=>d.fold.band[i]===b),
      y:d.fold.mag.filter((_,i)=>d.fold.band[i]===b),
      error_y:{type:"data",array:d.fold.magerr.filter((_,i)=>d.fold.band[i]===b),
        visible:true,color:BC[b]+"55",thickness:1},
      mode:"markers",marker:{color:BC[b],size:3.5,opacity:0.85},
      name:BAND_FULL[b]||b,type:"scatter",showlegend:false,
    })),
    {x:d.fold.fitted_phase,y:d.fold.fitted_mag,mode:"lines",
      line:{color:"#111827",width:2},name:"model",type:"scatter",showlegend:false},
  ] : null;

  const residTraces = d.fold ? bands.map(b=>{
    const fp=d.fold.fitted_phase,fm=d.fold.fitted_mag;
    const interp=ph=>fm[fp.reduce((bi,_,i)=>Math.abs(fp[i]-ph)<Math.abs(fp[bi]-ph)?i:bi,0)];
    const phases=d.fold.phase.filter((_,i)=>d.fold.band[i]===b);
    const mags=d.fold.mag.filter((_,i)=>d.fold.band[i]===b);
    return {x:phases,y:mags.map((m,i)=>m-interp(phases[i])),
      mode:"markers",marker:{color:BC[b],size:3,opacity:0.7},
      name:b,type:"scatter",showlegend:false};
  }) : null;

  const tabs = [
    ...(d.fold?[{id:"fold",label:"Phase Fold"}]:[]),
    {id:"pgram",label:"Periodogram"},
    {id:"lc",label:"Lightcurve"},
  ];

  const S = {
    bar:{display:"flex",gap:4,alignItems:"center",marginBottom:8,flexWrap:"wrap"},
    tab:{fontFamily:"Inter,sans-serif",fontSize:"0.72rem",fontWeight:500,padding:"4px 14px",
      border:"1px solid #dde3f0",borderRadius:5,background:"#f7f8fc",color:"#475569",cursor:"pointer"},
    on:{background:"#2563eb",color:"#fff",borderColor:"#2563eb"},
    pill:{marginLeft:"auto",fontFamily:"JetBrains Mono,monospace",fontSize:"0.72rem",
      color:"#2563eb",background:"#dbeafe",padding:"3px 10px",borderRadius:4,fontWeight:600},
    divider:{borderBottom:"1px dashed #e2e8f0",margin:"4px 0"},
  };

  const sharedMarkerLegend = []; // removed — labels shown on lines directly

  return (
    <div style={{marginTop:"1rem"}}>
      <div style={S.bar}>
        {tabs.map(t=>(
          <button key={t.id} style={{...S.tab,...(tab===t.id?S.on:{})}} onClick={()=>setTab(t.id)}>
            {t.label}
          </button>
        ))}
        {P && <span style={S.pill}>P = {P.toFixed(4)} hr</span>}
      </div>

      {/* ── Lightcurve ── */}
      {tab==="lc" && <>
        <LegendRow items={bands.map(b=>({color:BC[b],label:BAND_FULL[b]||b}))} />
        <Plot traces={lcTraces} layout={{
          ...BASE, margin:PLOT_MARGIN,
          title:{text:"Lightcurve",font:{size:12,color:"#0f172a"}},
          xaxis:{title:{text:"MJD"},gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
          yaxis:{title:{text:"mag"},autorange:"reversed",gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
        }} height={290}/>
      </>}

      {/* ── Periodogram ── */}
      {tab==="pgram" && <>
        <TierSection title="Tier 1: Fast Scan (GLS · MBLS)" period={P}>
          <LegendRow items={[
            {color:"#1d4ed8",label:"Multi-Band Lomb-Scargle (MBLS)"},
            {color:"#0d9488",label:"Generalised Lomb-Scargle (GLS, normalised)"},
          ]}/>
          <Plot traces={t1Traces} layout={{
            ...BASE, margin:PLOT_MARGIN,
            xaxis:{...xaxis},
            yaxis:{title:{text:"Normalised Power"},range:[0,1.05],gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
          }} height={200}/>

        </TierSection>

        <div style={S.divider}/>

        <TierSection title="Tier 2: High-Order Methods (MHAOV · MBLS)" period={P}>
          <LegendRow items={[
            {color:"#ea580c",label:"Multi-Band Lomb-Scargle (MBLS)"},
            {color:"#9333ea",label:"Multi-Harmonic Analysis of Variance (MHAOV, normalised)"},
          ]}/>
          <Plot traces={t2Traces} layout={{
            ...BASE, margin:PLOT_MARGIN,
            xaxis:{...xaxis},
            yaxis:{title:{text:"Normalised Power"},range:[0,1.05],gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
          }} height={200}/>
          {pval!=null && d.r_code!==0 && (
            <StatLine items={[{
              label:"Detection significance",
              value: pval < 1e-10 ? `★★★ very high  (p = ${pval.toExponential(2)})` :
                     pval < 0.001 ? `★★ significant  (p = ${pval.toExponential(2)})` :
                     pval < 0.05  ? `★ marginal  (p = ${pval.toExponential(2)})` :
                                    `not significant  (p = ${pval.toExponential(2)})`
            }]}/>
          )}
        </TierSection>

        <div style={S.divider}/>

        <TierSection title="Tier 3: Conditional Entropy (CE) · Window Function" period={P}>
          <LegendRow items={[
            ...(ceDetected?[{color:"#0e7490",label:"Conditional Entropy (CE, normalised)"}]:[]),
            {color:"#f59e0b",label:"Window function (normalised)"},
            {color:"#dc2626",label:"12h / 24h calendar aliases"},
          ]}/>
          <Plot traces={t3Traces} layout={{
            ...BASE, margin:PLOT_MARGIN,
            xaxis:{...xaxis},
            yaxis:{title:{text:"Normalised Power / CE"},range:[0,1.05],gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
          }} height={200}/>
          {pg.ce_scores && (
            <StatLine items={[
              ...(ceDetected ? [{label:"CE minimum",value:minCE.toFixed(4)}] : [{label:"CE",value:"no detection"}]),
              {label:"Note",value:"window fn peaks at 12/24h reflect observing cadence"},
            ]}/>
          )}
        </TierSection>
      </>}

      {/* ── Phase fold ── */}
      {tab==="fold" && d.fold && <>
        <LegendRow items={[
          ...bands.map(b=>({color:BC[b],label:BAND_FULL[b]||b})),
          {color:"#111827",label:"Fitted model"},
        ]}/>
        <div style={{display:"flex",flexDirection:"column",gap:0}}>
          <Plot traces={foldTraces} layout={{
            ...BASE, margin:{...PLOT_MARGIN,b:8},
            title:{text:`Phase fold — 2 cycles  (P = ${P?.toFixed(4)} hr)`,font:{size:12,color:"#0f172a"}},
            xaxis:{title:{text:""},range:[0,2],dtick:0.25,
              gridcolor:"#e8edf5",linecolor:"#cbd5e1",showticklabels:false},
            yaxis:{title:{text:"Δmag (detrended)"},autorange:"reversed",
              gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
          }} height={260}/>
          <Plot traces={residTraces} layout={{
            ...BASE, margin:{...PLOT_MARGIN,t:8},
            xaxis:{title:{text:"Phase"},range:[0,2],dtick:0.25,
              gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
            yaxis:{title:{text:"O−C (mag)"},zeroline:true,
              zerolinecolor:"#475569",gridcolor:"#e8edf5"},
          }} height={150}/>
        </div>
      </>}
    </div>
  );
}
