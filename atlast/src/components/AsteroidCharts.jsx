import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist";

const BC = { g:"#22c55e", r:"#f97316", i:"#3b82f6",
  Lg:"#22c55e", Lr:"#f97316", Li:"#3b82f6" };
const BAND_LONG = { g:"g-band (480nm)", r:"r-band (620nm)", i:"i-band (750nm)",
  Lg:"g-band (480nm)", Lr:"r-band (620nm)", Li:"i-band (750nm)" };
const getBandLabel = b => BAND_LONG[b] || BAND_LONG[b?.toLowerCase()] || b;
const getBandColor = b => BC[b] || BC[b?.toLowerCase()] || "#94a3b8";

function normArr(arr, max) {
  if (!arr||!arr.length) return [];
  return arr.map(v=>v/(max||1));
}

function PlotDiv({ traces, layout, height }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    try { Plotly.react(ref.current, traces, layout, {displayModeBar:false,responsive:true}); }
    catch(e) {}
  }, [traces, layout]);
  return <div ref={ref} style={{width:"100%",height}} />;
}

const BASE = {
  paper_bgcolor:"transparent", plot_bgcolor:"#fafbff",
  font:{family:"Inter,sans-serif",color:"#94a3b8",size:11},
  showlegend:false, hovermode:"closest",
  margin:{t:12,b:40,l:56,r:20},
};
const XY = {gridcolor:"#f1f5f9",linecolor:"#e2e8f0",zeroline:false};
const LOGX = {
  ...XY, type:"log", title:{text:"Period (hr)"},
  tickvals:[0.5,1,2,3,5,8,10,12,15,20,24],
  ticktext:["0.5","1","2","3","5","8","10","12","15","20","24"],
  range:[Math.log10(0.3),Math.log10(25)],
};

function vline(x,color,dash) {
  return {x:[x,x],y:[0,1],mode:"lines",type:"scatter",showlegend:false,
    line:{color,width:dash==="solid"?2:1.2,dash},
    hovertemplate:`${x.toFixed(3)}h<extra></extra>`};
}

// Tiny colored dot + label
function Chip({color,label}) {
  return (
    <span style={{display:"inline-flex",alignItems:"center",gap:5,
      fontSize:"0.66rem",color:"#64748b",fontFamily:"Inter,sans-serif"}}>
      <span style={{width:8,height:8,borderRadius:"50%",
        background:color,flexShrink:0,display:"inline-block"}}/>
      {label}
    </span>
  );
}

function Row({children,gap=12}) {
  return <div style={{display:"flex",gap,flexWrap:"wrap",
    alignItems:"center",padding:"2px 0 6px"}}>{children}</div>;
}

// P marker bar — shown once per tab, not per panel
function PMarkers({P}) {
  if (!P) return null;
  const mk = (val,color,label) => (
    <span key={label} style={{display:"inline-flex",alignItems:"center",gap:5,
      fontSize:"0.65rem",fontFamily:"JetBrains Mono,monospace",color}}>
      <svg width="18" height="8" style={{flexShrink:0}}>
        <line x1="0" y1="4" x2="18" y2="4" stroke={color} strokeWidth="1.8"
          strokeDasharray={color==="#1e3a5f"?"0":"3,2"}/>
      </svg>
      {label}
    </span>
  );
  return (
    <div style={{display:"flex",gap:14,flexWrap:"wrap",padding:"0 0 4px"}}>
      {mk(P,      "#1e3a5f",`P = ${P.toFixed(3)} hr`)}
      {mk(P/2,    "#d97706",`P/2 = ${(P/2).toFixed(3)} hr`)}
      {mk(P*2,    "#d97706",`2P = ${(P*2).toFixed(3)} hr`)}
    </div>
  );
}

// Thin tier label
function TierLabel({n,name}) {
  return (
    <div style={{display:"flex",alignItems:"baseline",gap:7,padding:"10px 0 2px",
      borderTop:n>1?"1px solid #f1f5f9":"none",marginTop:n>1?4:0}}>
      <span style={{fontSize:"0.58rem",fontWeight:700,color:"#cbd5e1",
        textTransform:"uppercase",letterSpacing:"0.12em",
        fontFamily:"Inter,sans-serif"}}>T{n}</span>
      <span style={{fontSize:"0.68rem",fontWeight:600,color:"#475569",
        fontFamily:"Inter,sans-serif"}}>{name}</span>
    </div>
  );
}

export default function AsteroidCharts({provid}) {
  const [d,setD]   = useState(null);
  const [err,setErr] = useState(false);
  const [tab,setTab] = useState("fold");

  useEffect(()=>{
    if (!provid) return;
    setD(null); setErr(false); setTab("fold");
    fetch(`/data/plots/data_${provid.replace(/\s+/g,"_")}.json`)
      .then(r=>{if(!r.ok)throw 0;return r.json();})
      .then(setD).catch(()=>setErr(true));
  },[provid]);

  if (!provid) return null;
  if (err) return <p style={{color:"#94a3b8",fontSize:"0.75rem",marginTop:10}}>No data.</p>;
  if (!d)  return <p style={{color:"#94a3b8",fontSize:"0.75rem",marginTop:10}}>Loading…</p>;

  const bands = [...new Set(d.obs.band)];
  const P     = d.best_period;
  const pg    = d.pgram;

  // Shared period lines
  const pLines = P ? [
    vline(P,    "#1e3a5f","solid"),
    vline(P/2,  "#d97706","dot"),
    vline(P*2,  "#d97706","dot"),
  ].filter(l=>l.x[0]>=0.3&&l.x[0]<=25) : [];

  // R=0 method disagreement lines
  const MC = {mhaov:"#9333ea",mbls:"#ea580c",ce:"#0e7490",gls:"#0d9488"};
  const mLines = !P&&d.method_periods
    ? Object.entries(d.method_periods)
        .filter(([,v])=>v&&v>=0.3&&v<=25)
        .map(([n,v])=>vline(v,MC[n]||"#94a3b8","dashdot"))
    : [];
  const aLines = P ? pLines : mLines;

  const maxMBLS  = Math.max(...pg.mbls_power);
  const maxGLS   = pg.gls_power?Math.max(...pg.gls_power):1;
  const maxMHAOV = pg.mhaov_power?Math.max(...pg.mhaov_power):1;
  const maxCE    = pg.ce_scores?Math.max(...pg.ce_scores):1;
  const minCE    = pg.ce_scores?Math.min(...pg.ce_scores):0;
  const ceOK     = pg.ce_scores&&minCE<0.95;
  const pval     = pg.p_value;

  const t1 = [...aLines,
    {x:pg.gls_periods||pg.periods,y:normArr(pg.gls_power||[],maxGLS),
      mode:"lines",line:{color:"#0d9488",width:1,dash:"dash"},type:"scatter"},
    {x:pg.periods,y:normArr(pg.mbls_power,maxMBLS),
      mode:"lines",line:{color:"#1d4ed8",width:1.4},type:"scatter"},
  ];
  const t2 = [...aLines,
    {x:pg.periods,y:normArr(pg.mhaov_power||[],maxMHAOV),
      mode:"lines",line:{color:"#9333ea",width:1.2,dash:"dot"},type:"scatter"},
    {x:pg.periods,y:normArr(pg.mbls_power,maxMBLS),
      mode:"lines",line:{color:"#ea580c",width:1.4},type:"scatter"},
  ];
  const maxWin = pg.window_power?Math.max(...pg.window_power):1;
  const t3 = [...aLines,
    vline(12,"#ef4444","dashdot"), vline(23.5,"#ef4444","dashdot"),
    ...(pg.window_periods&&pg.window_power?[{
      x:pg.window_periods,y:normArr(pg.window_power,maxWin),
      mode:"lines",line:{color:"#f59e0b",width:1},
      fill:"tozeroy",fillcolor:"rgba(245,158,11,0.07)",type:"scatter"}]:[]),
    ...(ceOK&&pg.ce_periods?[{
      x:pg.ce_periods,y:normArr(pg.ce_scores,maxCE),
      mode:"lines",line:{color:"#0e7490",width:1.4},type:"scatter"}]:[]),
  ];

  const pgLayout = {
    ...BASE,
    xaxis:{...LOGX},
    yaxis:{...XY,range:[0,1.05],title:{text:""}},
  };

  // Phase fold
  const foldT = d.fold?[
    ...bands.map(b=>({
      x:d.fold.phase.filter((_,i)=>d.fold.band[i]===b),
      y:d.fold.mag.filter((_,i)=>d.fold.band[i]===b),
      error_y:{type:"data",array:d.fold.magerr.filter((_,i)=>d.fold.band[i]===b),
        visible:true,color:getBandColor(b)+"44",thickness:1},
      mode:"markers",marker:{color:getBandColor(b),size:3.5,opacity:0.85},
      type:"scatter",showlegend:false})),
    {x:d.fold.fitted_phase,y:d.fold.fitted_mag,
      mode:"lines",line:{color:"#1e3a5f",width:2},type:"scatter",showlegend:false},
  ]:null;

  const residT = d.fold?bands.map(b=>{
    const fp=d.fold.fitted_phase,fm=d.fold.fitted_mag;
    const interp=ph=>fm[fp.reduce((bi,_,i)=>Math.abs(fp[i]-ph)<Math.abs(fp[bi]-ph)?i:bi,0)];
    const phases=d.fold.phase.filter((_,i)=>d.fold.band[i]===b);
    const mags=d.fold.mag.filter((_,i)=>d.fold.band[i]===b);
    return {x:phases,y:mags.map((m,i)=>m-interp(phases[i])),
      mode:"markers",marker:{color:getBandColor(b),size:3,opacity:0.7},
      type:"scatter",showlegend:false};
  }):null;

  const mjd0 = Math.min(...d.obs.mjd);
  const mjdToDate = mjd => new Date((mjd - 40587) * 86400000).toISOString();
  const lcT = bands.map(b=>({
    x:d.obs.mjd.filter((_,i)=>d.obs.band[i]===b).map(mjdToDate),
    y:d.obs.mag.filter((_,i)=>d.obs.band[i]===b),
    error_y:{type:"data",array:d.obs.magerr.filter((_,i)=>d.obs.band[i]===b),
      visible:true,color:getBandColor(b)+"44",thickness:1},
    mode:"markers",marker:{color:getBandColor(b),size:4,symbol:"circle",opacity:0.85},
    type:"scatter",showlegend:false}));

  const tabs=[
    ...(d.fold?[{id:"fold",label:"Phase Fold"}]:[]),
    {id:"pgram",label:"Periodogram"},
    {id:"lc",label:"Lightcurve"},
  ];

  const S={
    bar:{display:"flex",gap:4,alignItems:"center",marginBottom:14,flexWrap:"wrap",
      paddingTop:14,borderTop:"1px solid #f1f5f9",marginTop:8},
    tab:{fontFamily:"Inter,sans-serif",fontSize:"0.72rem",fontWeight:500,
      padding:"5px 16px",border:"1px solid #e2e8f0",borderRadius:6,
      background:"#fff",color:"#64748b",cursor:"pointer"},
    on:{background:"#2563eb",color:"#fff",borderColor:"#2563eb"},
    pill:{marginLeft:"auto",fontFamily:"JetBrains Mono,monospace",fontSize:"0.7rem",
      color:"#2563eb",background:"#eff6ff",padding:"3px 10px",borderRadius:4,fontWeight:600},
  };

  const sigStars = pval==null?null
    :pval<1e-10?"★★★":pval<0.001?"★★":pval<0.05?"★":"—";
  const sigColor = pval==null?"#94a3b8"
    :pval<1e-10?"#16a34a":pval<0.001?"#d97706":"#dc2626";

  return (
    <div>
      {/* Tab bar */}
      <div style={S.bar}>
        {tabs.map(t=>(
          <button key={t.id} style={{...S.tab,...(tab===t.id?S.on:{})}}
            onClick={()=>setTab(t.id)}>{t.label}</button>
        ))}

      </div>

      {/* PHASE FOLD */}
      {tab==="fold"&&d.fold&&<>
        <Row>
          {bands.map(b=><Chip key={b} color={getBandColor(b)} label={getBandLabel(b)}/>)}
          <Chip color="#1e3a5f" label="fitted model"/>
        </Row>
        <div style={{display:"flex",flexDirection:"column"}}>
          <PlotDiv traces={foldT} height={250} layout={{
            ...BASE,margin:{t:12,b:4,l:56,r:20},
            xaxis:{...XY,range:[0,2],dtick:0.25,showticklabels:false,title:{text:""}},
            yaxis:{...XY,title:{text:"Δmag"},autorange:"reversed"},
          }}/>
          <PlotDiv traces={residT} height={110} layout={{
            ...BASE,margin:{t:4,b:40,l:56,r:20},
            xaxis:{...XY,range:[0,2],dtick:0.25,title:{text:"Phase"}},
            yaxis:{...XY,title:{text:"O−C"},
              zeroline:true,zerolinecolor:"#cbd5e1",zerolinewidth:1},
          }}/>
        </div>
      </>}

      {/* PERIODOGRAM */}
      {tab==="pgram"&&<>
        <PMarkers P={P}/>

        <TierLabel n={1} name="Fast Scan"/>
        <Row>
          <Chip color="#1d4ed8" label="Multi-Band Lomb-Scargle (MBLS)"/>
          <Chip color="#0d9488" label="Generalised Lomb-Scargle (GLS)"/>
        </Row>
        <PlotDiv traces={t1} height={175} layout={pgLayout}/>

        <TierLabel n={2} name="High-Order Methods"/>
        <Row>
          <Chip color="#ea580c" label="Multi-Band Lomb-Scargle (MBLS)"/>
          <Chip color="#9333ea" label="Multi-Harmonic AoV (MHAOV)"/>
          {sigStars&&d.r_code!==0&&(
            <span style={{marginLeft:"auto",fontSize:"0.65rem",
              color:sigColor,fontFamily:"JetBrains Mono,monospace",fontWeight:600}}>
              {sigStars} p = {pval?.toExponential(2)}
            </span>
          )}
        </Row>
        <PlotDiv traces={t2} height={175} layout={pgLayout}/>

        <TierLabel n={3} name="Conditional Entropy · Window Function"/>
        <Row>
          {ceOK&&<Chip color="#0e7490" label="Conditional Entropy (CE)"/>}
          <Chip color="#f59e0b" label="Window function "/>
          <Chip color="#ef4444" label="12h · 24h aliases"/>
          {!ceOK&&(
            <span style={{fontSize:"0.65rem",color:"#94a3b8",
              fontFamily:"Inter,sans-serif",fontStyle:"italic"}}>
              CE: no detection
            </span>
          )}
        </Row>
        <PlotDiv traces={t3} height={175} layout={{...pgLayout,
          yaxis:{...XY,range:[0,1.05],title:{text:""}}}}/>
      </>}

      {/* LIGHTCURVE */}
      {tab==="lc"&&<>
        <Row>
          {bands.map(b=><Chip key={b} color={getBandColor(b)} label={getBandLabel(b)}/>)}
        </Row>
        <PlotDiv traces={lcT} height={270} layout={{
          ...BASE,
          xaxis:{...XY,title:{text:"Date (UTC)"},type:"date",tickformat:"%b %d"},
          yaxis:{...XY,title:{text:"Magnitude"},autorange:"reversed"},
        }}/>
      </>}
    </div>
  );
}
