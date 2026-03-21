import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist";

const T = {
  accent:"#2563eb", accentSoft:"#dbeafe",
  green:"#16a34a",  greenSoft:"#dcfce7",
  amber:"#d97706",  amberSoft:"#fef3c7",
  red:"#dc2626",    redSoft:"#fee2e2",
  purple:"#7c3aed", purpleSoft:"#ede9fe",
  yellow:"#a16207", yellowSoft:"#fef9c3",
  textPri:"#0f172a", textSec:"#475569", textDim:"#94a3b8",
  border:"#dde3f0", bg:"#f7f8fc", bgCard:"#ffffff",
  shadow:"0 1px 3px rgba(0,0,0,0.08)",
};

const R_COLOR = {"3":T.green,"2":T.amber,"1":T.yellow,"0":T.red,"-1":T.purple};
const R_SOFT  = {"3":T.greenSoft,"2":T.amberSoft,"1":T.yellowSoft,"0":T.redSoft,"-1":T.purpleSoft};
const R_LABEL = {"3":"Confirmed","2":"Moderate","1":"Tentative","0":"No period","-1":"Alias"};

const BASE = {
  paper_bgcolor:"transparent", plot_bgcolor:"#fafbff",
  font:{family:"Inter,sans-serif",color:"#64748b",size:11},
  showlegend:false, hovermode:"closest",
  margin:{t:16,b:44,l:58,r:16},
};
const GRID = {gridcolor:"#f1f5f9",linecolor:"#e2e8f0",zeroline:false};
const BARRIER = {type:"line",x0:2.2,x1:2.2,y0:0,y1:1,yref:"paper",
  line:{color:T.red,width:1.5,dash:"dash"},opacity:0.6};
const BARRIER_ANN = {x:2.4,y:0.97,xref:"x",yref:"paper",
  text:"2.2 hr spin barrier",showarrow:false,
  font:{size:9,color:T.red},xanchor:"left"};

function PlotDiv({traces,layout,height,onReady}) {
  const ref = useRef(null);
  useEffect(()=>{
    if (!ref.current||!traces) return;
    try {
      Plotly.react(ref.current,traces,{...BASE,...layout},
        {displayModeBar:false,responsive:true});
      if (onReady) onReady(ref.current);
    } catch(e){}
  },[traces,layout]);
  return <div ref={ref} style={{width:"100%",height}}/>;
}

function Card({title,subtitle,children,style,muted}) {
  return (
    <div style={{background:muted?"#f8fafc":T.bgCard,
      border:`1px solid ${T.border}`,borderRadius:10,
      overflow:"hidden",boxShadow:T.shadow,...style}}>
      {(title||subtitle)&&(
        <div style={{padding:"16px 20px 0"}}>
          {title&&<div style={{fontSize:"0.8rem",fontWeight:600,
            color:muted?T.textDim:T.textPri,fontFamily:"Inter,sans-serif"}}>{title}</div>}
          {subtitle&&<div style={{fontSize:"0.64rem",color:T.textDim,marginTop:3}}>{subtitle}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

function RDot({r}) {
  return <span style={{display:"inline-flex",alignItems:"center",gap:5,
    fontSize:"0.65rem",color:T.textSec}}>
    <span style={{width:8,height:8,borderRadius:"50%",background:R_COLOR[r],
      flexShrink:0,display:"inline-block"}}/>
    {R_LABEL[r]}
  </span>;
}

// ── 1. Spin rate histogram ─────────────────────────────────
function SpinHistogram({catalog}) {
  const periods = catalog
    .filter(r=>r.final_period_hr&&r.final_period_hr!=="")
    .map(r=>parseFloat(r.final_period_hr))
    .filter(p=>!isNaN(p)&&p>0&&p<=25);

  return (
    <Card title="Rotation Period Distribution"
      subtitle={`${periods.length} asteroids with detected periods`}>
      <PlotDiv height={220} traces={[{
        x:periods, type:"histogram",
        xbins:{start:0,end:25,size:0.5},
        marker:{color:T.accentSoft,line:{color:T.accent,width:1}},
        hovertemplate:"%{y} asteroids · %{x:.1f}–%{xend:.1f} hr bin<extra></extra>",
      }]} layout={{
        xaxis:{...GRID,title:{text:"Rotation period (hr)"},range:[0,25]},
        yaxis:{...GRID,title:{text:"Count"}},
        shapes:[BARRIER],
        annotations:[BARRIER_ANN],
      }}/>
    </Card>
  );
}

// ── 2. Spin spectrum dot strip ─────────────────────────────
// Each asteroid = one dot, x=period, y=jittered, size=amplitude
function SpinSpectrum({catalog,onSelect}) {
  const rows = catalog.filter(r=>
    r.final_period_hr&&r.final_period_hr!==""&&String(r.r_code)!=="0"
  );

  // Jitter y slightly so dots don't overlap
  const seed = (i) => (Math.sin(i*127.1)*43758.5453)%1;
  const groups = ["3","2","1","-1"];
  const traces = groups.map(rc => {
    const sub = rows.filter(r=>String(r.r_code)===rc);
    return {
      x: sub.map(r=>parseFloat(r.final_period_hr)),
      y: sub.map((_,i)=>0.5+seed(i)*0.5),
      text: sub.map(r=>r.provid),
      customdata: sub,
      mode:"markers",
      marker:{
        color: R_COLOR[rc],
        size: sub.map(r=>r.t2_amplitude_mag
          ? Math.max(6, Math.min(18, parseFloat(r.t2_amplitude_mag)*10))
          : 8),
        opacity:0.8,
        line:{color:"rgba(255,255,255,0.5)",width:0.8},
      },
      name:R_LABEL[rc], type:"scatter",
      hovertemplate:"<b>%{text}</b><br>P = %{x:.3f} hr<extra></extra>",
    };
  }).filter(t=>t.x.length>0);

  const ref = useRef(null);
  useEffect(()=>{
    if (!ref.current) return;
    try {
      Plotly.react(ref.current, traces, {
        ...BASE,
        margin:{t:16,b:44,l:16,r:16},
        xaxis:{...GRID,title:{text:"Rotation period (hr)"},range:[0,25]},
        yaxis:{...GRID,visible:false,range:[0,1.2]},
        shapes:[BARRIER],
        annotations:[BARRIER_ANN],
      },{displayModeBar:false,responsive:true});
      ref.current.on("plotly_click",e=>{
        const row=e.points[0]?.customdata;
        if(row) onSelect(row);
      });
    } catch(e){}
  },[catalog]);

  return (
    <Card title="Spin Spectrum"
      subtitle="Each dot = one asteroid · dot size ∝ lightcurve amplitude · click to explore">
      <div style={{display:"flex",gap:12,flexWrap:"wrap",padding:"6px 20px 0"}}>
        {groups.map(rc=><RDot key={rc} r={rc}/>)}
        <span style={{marginLeft:"auto",fontSize:"0.63rem",color:T.textDim}}>
          size ∝ amplitude
        </span>
      </div>
      <div ref={ref} style={{width:"100%",height:160}}/>
    </Card>
  );
}

// ── 3. Period vs Amplitude scatter ────────────────────────
function AmpVsPeriod({catalog,onSelect}) {
  const rows = catalog.filter(r=>
    r.final_period_hr&&r.final_period_hr!==""&&
    r.t2_amplitude_mag&&r.t2_amplitude_mag!==""&&
    String(r.r_code)!=="0"
  );
  const groups=["3","2","1","-1"];
  const traces=groups.map(rc=>{
    const sub=rows.filter(r=>String(r.r_code)===rc);
    return {
      x:sub.map(r=>parseFloat(r.final_period_hr)),
      y:sub.map(r=>parseFloat(r.t2_amplitude_mag)),
      text:sub.map(r=>r.provid),
      customdata:sub,
      mode:"markers",
      marker:{color:R_COLOR[rc],size:8,opacity:0.85,
        line:{color:"rgba(255,255,255,0.5)",width:0.8}},
      name:R_LABEL[rc],type:"scatter",
      hovertemplate:"<b>%{text}</b><br>P = %{x:.3f} hr<br>Amp = %{y:.3f} mag<extra></extra>",
    };
  }).filter(t=>t.x.length>0);

  const ref = useRef(null);
  useEffect(()=>{
    if (!ref.current) return;
    try {
      Plotly.react(ref.current,traces,{
        ...BASE,
        margin:{t:16,b:44,l:58,r:16},
        xaxis:{...GRID,title:{text:"Rotation period (hr)"},range:[0,25]},
        yaxis:{...GRID,title:{text:"Lightcurve amplitude (mag)"}},
        shapes:[{...BARRIER,y0:undefined,y1:undefined,
          line:{...BARRIER.line,opacity:0.3}}],
      },{displayModeBar:false,responsive:true});
      ref.current.on("plotly_click",e=>{
        const row=e.points[0]?.customdata;
        if(row) onSelect(row);
      });
    } catch(e){}
  },[catalog]);

  return (
    <Card title="Rotation Period vs Shape Elongation"
      subtitle="Lightcurve amplitude proxies elongation — larger amplitude = more elongated body · click any point">
      <div style={{display:"flex",gap:12,flexWrap:"wrap",padding:"6px 20px 0"}}>
        {groups.map(rc=><RDot key={rc} r={rc}/>)}
      </div>
      <div ref={ref} style={{width:"100%",height:260}}/>
    </Card>
  );
}

// ── 4. Color index placeholder ────────────────────────────
function ColorIndexPlaceholder() {
  return (
    <Card title="Color Index Diagram" subtitle="g−r vs r−i · taxonomic classification proxy" muted>
      <div style={{height:220,display:"flex",flexDirection:"column",
        alignItems:"center",justifyContent:"center",gap:10,
        padding:"0 32px",textAlign:"center"}}>
        <div style={{fontSize:"2rem",opacity:0.2}}>◉</div>
        <div style={{fontSize:"0.75rem",fontWeight:600,color:T.textDim}}>
          Coming in next release
        </div>
        <div style={{fontSize:"0.65rem",color:T.textDim,lineHeight:1.6,maxWidth:260}}>
          Multi-band color indices (g−r, r−i) will reveal asteroid taxonomy —
          C-type, S-type, and other spectral classes — directly from LSST photometry.
        </div>
      </div>
    </Card>
  );
}

// ── Main ──────────────────────────────────────────────────
export default function Explore({catalog,onSelect}) {
  if (!catalog.length) return (
    <div style={{padding:"48px 0",textAlign:"center",color:T.textDim}}>
      Loading…
    </div>
  );

  return (
    <div style={{display:"flex",flexDirection:"column",gap:20}}>

      {/* Row 1: spin histogram + spin spectrum stacked */}
      <SpinHistogram catalog={catalog}/>
      <SpinSpectrum catalog={catalog} onSelect={onSelect}/>

      {/* Row 2: period vs amplitude + color placeholder */}
      <div style={{display:"grid",gridTemplateColumns:"1.4fr 1fr",gap:16}}>
        <AmpVsPeriod catalog={catalog} onSelect={onSelect}/>
        <ColorIndexPlaceholder/>
      </div>

      {/* Data note */}
      <div style={{padding:"10px 16px",background:T.bgCard,
        border:`1px solid ${T.border}`,borderRadius:8,
        fontSize:"0.67rem",color:T.textDim,lineHeight:1.7,
        display:"flex",justifyContent:"space-between",alignItems:"center"}}>
        <span>
          Rubin/LSST commissioning · Apr–May 2025 · 76 asteroids ·
          1.45M observations · 12-day baseline · g, r, i bands
        </span>
        <a href="https://github.com/wonrobot/asteroid-pipeline"
          target="_blank" rel="noreferrer"
          style={{color:T.accent,textDecoration:"none",flexShrink:0,marginLeft:16}}>
          GitHub ↗
        </a>
      </div>
    </div>
  );
}
