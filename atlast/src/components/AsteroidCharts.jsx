import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist";

const BC = { g:"#22c55e", r:"#f97316", i:"#3b82f6" };
const BAND_FULL = { g:"g-band", r:"r-band", i:"i-band" };
const BAND_LONG = { g:"g-band (green, 480nm)", r:"r-band (orange, 620nm)", i:"i-band (blue, 750nm)" };

function nightMap(mjd) {
  const days = [...new Set(mjd.map(m=>Math.floor(m)))].sort((a,b)=>a-b);
  const map = {}; days.forEach((d,i)=>{map[d]=i;});
  return mjd.map(m=>map[Math.floor(m)]);
}

function PlotDiv({ traces, layout, height=280 }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    try { Plotly.react(ref.current, traces, layout, {displayModeBar:false, responsive:true}); }
    catch(e) { console.warn("Plotly",e); }
  }, [traces, layout]);
  return <div ref={ref} style={{width:"100%", height}} />;
}

const BASE_LAYOUT = {
  paper_bgcolor:"transparent",
  plot_bgcolor:"#fafbff",
  font:{family:"Inter, sans-serif", color:"#64748b", size:11},
  showlegend:false,
  hovermode:"closest",
  margin:{t:16, b:44, l:62, r:24},
};

const YAXIS_BASE = {gridcolor:"#e8edf5", linecolor:"#cbd5e1", zeroline:false};
const XAXIS_BASE = {gridcolor:"#e8edf5", linecolor:"#cbd5e1"};

const LOG_XAXIS = {
  ...XAXIS_BASE,
  title:{text:"Period (hr)"},
  type:"log",
  tickformat:".3~g",
  range:[Math.log10(0.3), Math.log10(25)],
  tickvals:[0.5,1,2,3,4,5,6,7,8,9,10,12,15,20,24],
  ticktext:["0.5","1","2","3","4","5","6","7","8","9","10","12","15","20","24"],
};

// Vertical line as scatter trace (no text — labels shown in div above)
function vline(x, color, dash, maxY=1) {
  return {
    x:[x,x], y:[0, maxY],
    mode:"lines", type:"scatter", showlegend:false,
    line:{color, width: dash==="solid"?2.5:1.5, dash},
    hovertemplate:`${x.toFixed(3)} hr<extra></extra>`,
  };
}

function normArr(arr, max) {
  if (!arr || !arr.length) return [];
  return arr.map(v => v / (max || 1));
}

// Section header component
function SectionTitle({ tier, title, subtitle }) {
  return (
    <div style={{padding:"10px 2px 4px", borderTop: tier>1?"1px dashed #e2e8f0":"none", marginTop: tier>1?8:0}}>
      <div style={{display:"flex", alignItems:"baseline", gap:8, flexWrap:"wrap"}}>
        <span style={{fontSize:"0.6rem", fontWeight:700, color:"#94a3b8",
          textTransform:"uppercase", letterSpacing:"0.1em", fontFamily:"Inter,sans-serif"}}>
          Tier {tier}
        </span>
        <span style={{fontSize:"0.72rem", fontWeight:600, color:"#334155",
          fontFamily:"Inter,sans-serif"}}>
          {title}
        </span>
        {subtitle && (
          <span style={{fontSize:"0.65rem", color:"#94a3b8", fontFamily:"Inter,sans-serif"}}>
            {subtitle}
          </span>
        )}
      </div>
    </div>
  );
}

// Colored method label row
function MethodLabels({ items }) {
  return (
    <div style={{display:"flex", gap:16, flexWrap:"wrap", padding:"2px 2px 4px", alignItems:"center"}}>
      {items.map(({color, label, style})=>(
        <span key={label} style={{fontSize:"0.68rem", fontWeight:600,
          color, fontFamily:"Inter,sans-serif", ...(style||{})}}>
          {label}
        </span>
      ))}
    </div>
  );
}

// Period marker info bar
function PeriodBar({ P }) {
  if (!P) return null;
  return (
    <div style={{display:"flex", gap:16, padding:"3px 2px 5px", flexWrap:"wrap"}}>
      <span style={{fontSize:"0.65rem", fontFamily:"JetBrains Mono,monospace", color:"#111827", display:"flex", alignItems:"center", gap:5}}>
        <span style={{display:"inline-block", width:16, height:2, background:"#111827", borderRadius:1}}/>
        P = {P.toFixed(4)} hr
      </span>
      <span style={{fontSize:"0.65rem", fontFamily:"JetBrains Mono,monospace", color:"#d97706", display:"flex", alignItems:"center", gap:5}}>
        <span style={{display:"inline-block", width:16, height:2, background:"#d97706", borderRadius:1, opacity:0.7}}/>
        P/2 = {(P/2).toFixed(4)} hr
      </span>
      <span style={{fontSize:"0.65rem", fontFamily:"JetBrains Mono,monospace", color:"#d97706", display:"flex", alignItems:"center", gap:5}}>
        <span style={{display:"inline-block", width:16, height:2, background:"#d97706", borderRadius:1, opacity:0.7}}/>
        2P = {(P*2).toFixed(4)} hr
      </span>
    </div>
  );
}

// Stat line
function StatLine({ items }) {
  return (
    <div style={{display:"flex", gap:20, flexWrap:"wrap", padding:"3px 2px 2px"}}>
      {items.map(({label,value,highlight})=>(
        <span key={label} style={{fontSize:"0.67rem", color:"#64748b", fontFamily:"Inter,sans-serif"}}>
          {label}:{" "}
          <strong style={{color: highlight||"#0f172a"}}>{value}</strong>
        </span>
      ))}
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

  // Period vertical lines
  const pLines = P ? [
    vline(P,     "#111827","solid"),
    vline(P/2,   "#d97706","dot"),
    vline(P*2,   "#d97706","dot"),
  ].filter(l=>l.x[0]>=0.3&&l.x[0]<=25) : [];

  // Method disagreement lines for R=0
  const METHOD_C = {mhaov:"#9333ea",mbls:"#ea580c",ce:"#0e7490",gls:"#0d9488"};
  const methodLines = !P && d.method_periods
    ? Object.entries(d.method_periods)
        .filter(([,v])=>v&&v>=0.3&&v<=25)
        .map(([n,v])=>vline(v, METHOD_C[n]||"#94a3b8","dashdot"))
    : [];

  const activeLines = P ? pLines : methodLines;

  /* ── T1 ── */
  const maxMBLS  = Math.max(...pg.mbls_power);
  const maxGLS   = pg.gls_power ? Math.max(...pg.gls_power) : 1;
  const t1Traces = [
    ...activeLines,
    {x:pg.gls_periods||pg.periods, y:normArr(pg.gls_power||[], maxGLS),
      mode:"lines", line:{color:"#0d9488",width:1,dash:"dash"}, type:"scatter"},
    {x:pg.periods, y:normArr(pg.mbls_power, maxMBLS),
      mode:"lines", line:{color:"#1d4ed8",width:1.4}, type:"scatter"},
  ];

  /* ── T2 ── */
  const maxMHAOV = pg.mhaov_power ? Math.max(...pg.mhaov_power) : 1;
  const t2Traces = [
    ...activeLines,
    {x:pg.periods, y:normArr(pg.mhaov_power||[], maxMHAOV),
      mode:"lines", line:{color:"#9333ea",width:1.2,dash:"dot"}, type:"scatter"},
    {x:pg.periods, y:normArr(pg.mbls_power, maxMBLS),
      mode:"lines", line:{color:"#ea580c",width:1.4}, type:"scatter"},
  ];
  const pval = pg.p_value;
  const sigLabel = pval==null ? null
    : pval < 1e-10 ? `★★★ very high  (p = ${pval.toExponential(2)})`
    : pval < 0.001 ? `★★ significant  (p = ${pval.toExponential(2)})`
    : pval < 0.05  ? `★ marginal  (p = ${pval.toExponential(2)})`
    :                `not significant  (p = ${pval.toExponential(2)})`;

  /* ── T3 ── */
  const maxCE  = pg.ce_scores ? Math.max(...pg.ce_scores) : 1;
  const minCE  = pg.ce_scores ? Math.min(...pg.ce_scores) : 0;
  const ceOK   = pg.ce_scores && minCE < 0.95;
  const maxWin = pg.window_power ? Math.max(...pg.window_power) : 1;
  const t3Traces = [
    ...activeLines,
    vline(12, "#dc2626","dashdot"), vline(24, "#dc2626","dashdot"),
    ...(pg.window_periods&&pg.window_power ? [{
      x:pg.window_periods, y:normArr(pg.window_power, maxWin),
      mode:"lines", line:{color:"#f59e0b",width:1},
      fill:"tozeroy", fillcolor:"rgba(245,158,11,0.08)", type:"scatter",
    }] : []),
    ...(ceOK&&pg.ce_periods ? [{
      x:pg.ce_periods, y:normArr(pg.ce_scores, maxCE),
      mode:"lines", line:{color:"#0e7490",width:1.4}, type:"scatter",
    }] : []),
  ];

  /* ── Phase fold ── */
  const foldTraces = d.fold ? [
    ...bands.map(b=>({
      x:d.fold.phase.filter((_,i)=>d.fold.band[i]===b),
      y:d.fold.mag.filter((_,i)=>d.fold.band[i]===b),
      error_y:{type:"data",
        array:d.fold.magerr.filter((_,i)=>d.fold.band[i]===b),
        visible:true,color:BC[b]+"55",thickness:1},
      mode:"markers",marker:{color:BC[b],size:3.5,opacity:0.85},
      type:"scatter",showlegend:false,
    })),
    {x:d.fold.fitted_phase,y:d.fold.fitted_mag,
      mode:"lines",line:{color:"#111827",width:2},type:"scatter",showlegend:false},
  ] : null;

  const residTraces = d.fold ? bands.map(b=>{
    const fp=d.fold.fitted_phase,fm=d.fold.fitted_mag;
    const interp=ph=>fm[fp.reduce((bi,_,i)=>Math.abs(fp[i]-ph)<Math.abs(fp[bi]-ph)?i:bi,0)];
    const phases=d.fold.phase.filter((_,i)=>d.fold.band[i]===b);
    const mags=d.fold.mag.filter((_,i)=>d.fold.band[i]===b);
    return {x:phases,y:mags.map((m,i)=>m-interp(phases[i])),
      mode:"markers",marker:{color:BC[b],size:3,opacity:0.7},
      type:"scatter",showlegend:false};
  }) : null;

  /* ── Lightcurve ── */
  const lcTraces = bands.map(b=>({
    x:d.obs.mjd.filter((_,i)=>d.obs.band[i]===b),
    y:d.obs.mag.filter((_,i)=>d.obs.band[i]===b),
    error_y:{type:"data",
      array:d.obs.magerr.filter((_,i)=>d.obs.band[i]===b),
      visible:true,color:BC[b]+"55",thickness:1},
    mode:"markers",marker:{color:BC[b],size:4,symbol:"circle",opacity:0.85},
    type:"scatter",showlegend:false,
  }));

  const tabs = [
    ...(d.fold?[{id:"fold",label:"Phase Fold"}]:[]),
    {id:"pgram",label:"Periodogram"},
    {id:"lc",label:"Lightcurve"},
  ];

  const S = {
    bar:{display:"flex",gap:4,alignItems:"center",marginBottom:12,flexWrap:"wrap"},
    tab:{fontFamily:"Inter,sans-serif",fontSize:"0.72rem",fontWeight:500,
      padding:"5px 16px",border:"1px solid #dde3f0",borderRadius:6,
      background:"#f7f8fc",color:"#475569",cursor:"pointer",transition:"all 0.1s"},
    on:{background:"#2563eb",color:"#fff",borderColor:"#2563eb"},
    pill:{marginLeft:"auto",fontFamily:"JetBrains Mono,monospace",fontSize:"0.72rem",
      color:"#2563eb",background:"#dbeafe",padding:"3px 10px",borderRadius:4,fontWeight:600},
  };

  const pgLayout = (extra={}) => ({
    ...BASE_LAYOUT,
    xaxis:{...LOG_XAXIS},
    yaxis:{...YAXIS_BASE,title:{text:"Normalised power"},range:[0,1.05]},
    ...extra,
  });

  return (
    <div style={{marginTop:"1rem"}}>
      <div style={S.bar}>
        {tabs.map(t=>(
          <button key={t.id}
            style={{...S.tab,...(tab===t.id?S.on:{})}}
            onClick={()=>setTab(t.id)}>
            {t.label}
          </button>
        ))}
        {P && <span style={S.pill}>P = {P.toFixed(4)} hr</span>}
      </div>

      {/* ══ PHASE FOLD ══ */}
      {tab==="fold" && d.fold && <>
        <SectionTitle tier={null} title="Phase Fold"
          subtitle={`2 cycles · P = ${P?.toFixed(4)} hr`} />
        <MethodLabels items={[
          ...bands.map(b=>({color:BC[b],label:BAND_LONG[b]||b})),
          {color:"#111827",label:"Fitted model"},
        ]}/>
        <div style={{display:"flex",flexDirection:"column"}}>
          <PlotDiv traces={foldTraces} height={260} layout={{
            ...BASE_LAYOUT,
            margin:{t:16,b:6,l:62,r:24},
            xaxis:{...XAXIS_BASE,range:[0,2],dtick:0.25,showticklabels:false,title:{text:""}},
            yaxis:{...YAXIS_BASE,title:{text:"Δmag (detrended)"},autorange:"reversed"},
          }}/>
          <PlotDiv traces={residTraces} height={130} layout={{
            ...BASE_LAYOUT,
            margin:{t:6,b:44,l:62,r:24},
            xaxis:{...XAXIS_BASE,range:[0,2],dtick:0.25,title:{text:"Phase"}},
            yaxis:{...YAXIS_BASE,title:{text:"O−C (mag)"},
              zeroline:true,zerolinecolor:"#94a3b8",zerolinewidth:1},
          }}/>
        </div>
      </>}

      {/* ══ PERIODOGRAM ══ */}
      {tab==="pgram" && <>
        <SectionTitle tier={1} title="Fast Scan"
          subtitle="Generalised Lomb-Scargle (GLS) · Multi-Band Lomb-Scargle (MBLS)" />
        <MethodLabels items={[
          {color:"#1d4ed8",label:"Multi-Band Lomb-Scargle (MBLS)"},
          {color:"#0d9488",label:"Generalised Lomb-Scargle (GLS, normalised)"},
        ]}/>
        <PeriodBar P={P}/>
        <PlotDiv traces={t1Traces} height={190} layout={pgLayout()}/>

        <SectionTitle tier={2} title="High-Order Methods"
          subtitle="Multi-Harmonic Analysis of Variance (MHAOV) · Multi-Band Lomb-Scargle (MBLS)" />
        <MethodLabels items={[
          {color:"#ea580c",label:"Multi-Band Lomb-Scargle (MBLS)"},
          {color:"#9333ea",label:"Multi-Harmonic Analysis of Variance (MHAOV, normalised)"},
        ]}/>
        <PeriodBar P={P}/>
        <PlotDiv traces={t2Traces} height={190} layout={pgLayout()}/>
        {sigLabel && d.r_code!==0 && (
          <StatLine items={[{label:"Detection significance",value:sigLabel,
            highlight: pval<1e-10?"#16a34a":pval<0.001?"#d97706":"#dc2626"}]}/>
        )}

        <SectionTitle tier={3} title="Conditional Entropy (CE) · Window Function"
          subtitle="Cadence aliases at 12h/24h shown in red" />
        <MethodLabels items={[
          ...(ceOK?[{color:"#0e7490",label:"Conditional Entropy (CE, normalised)"}]:[]),
          {color:"#f59e0b",label:"Window function (normalised)"},
          {color:"#dc2626",label:"12h · 24h calendar aliases"},
        ]}/>
        <PeriodBar P={P}/>
        <PlotDiv traces={t3Traces} height={190} layout={pgLayout({
          yaxis:{...YAXIS_BASE,title:{text:"Normalised power / CE"},range:[0,1.05]},
        })}/>
        {pg.ce_scores && (
          <StatLine items={ceOK
            ? [{label:"CE minimum",value:minCE.toFixed(4)}]
            : [{label:"Conditional Entropy",value:"no detection — period not constrained by this method"}]}
          />
        )}
      </>}

      {/* ══ LIGHTCURVE ══ */}
      {tab==="lc" && <>
        <SectionTitle tier={null} title="Lightcurve"
          subtitle="Raw photometry colored by filter" />
        <MethodLabels items={bands.map(b=>({color:BC[b],label:BAND_LONG[b]||b}))}/>
        <PlotDiv traces={lcTraces} height={290} layout={{
          ...BASE_LAYOUT,
          xaxis:{...XAXIS_BASE,title:{text:"MJD"}},
          yaxis:{...YAXIS_BASE,title:{text:"Magnitude"},autorange:"reversed"},
        }}/>
      </>}
    </div>
  );
}
