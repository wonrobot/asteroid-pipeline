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
const R_LABEL = {"3":"Confirmed (R=3)","2":"Moderate (R=2)","1":"Tentative (R=1)","0":"No period (R=0)","-1":"Alias (R=−1)"};

const BASE_LAYOUT = {
  paper_bgcolor:"transparent", plot_bgcolor:"#fafbff",
  font:{family:"Inter,sans-serif",color:"#64748b",size:11},
  showlegend:false, hovermode:"closest",
  margin:{t:12,b:44,l:58,r:16},
};
const GRID = {gridcolor:"#f1f5f9",linecolor:"#e2e8f0"};

function PlotDiv({traces,layout,height,onClick}) {
  const ref = useRef(null);
  useEffect(()=>{
    if (!ref.current||!traces) return;
    try {
      Plotly.react(ref.current,traces,{...BASE_LAYOUT,...layout},
        {displayModeBar:false,responsive:true});
      if (onClick) ref.current.on("plotly_click",e=>{
        const row=e.points[0]?.customdata;
        if(row) onClick(row);
      });
    } catch(e){}
  },[traces,layout]);
  return <div ref={ref} style={{width:"100%",height}}/>;
}

function Card({title,subtitle,children,style}) {
  return (
    <div style={{background:T.bgCard,border:`1px solid ${T.border}`,
      borderRadius:10,overflow:"hidden",boxShadow:T.shadow,...style}}>
      {(title||subtitle)&&(
        <div style={{padding:"14px 18px 0"}}>
          {title&&<div style={{fontSize:"0.78rem",fontWeight:600,
            color:T.textPri,fontFamily:"Inter,sans-serif"}}>{title}</div>}
          {subtitle&&<div style={{fontSize:"0.63rem",color:T.textDim,marginTop:2}}>{subtitle}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

// ── 1. Summary stat cards ──────────────────────────────────
function SummaryCards({catalog}) {
  const wp  = catalog.filter(r=>r.final_period_hr&&r.final_period_hr!=="");
  const con = catalog.filter(r=>r.reliability==="confirmed");
  const fast= wp.filter(r=>parseFloat(r.final_period_hr)<2.2);
  const r3  = catalog.filter(r=>String(r.r_code)==="3");
  const med = [...wp.map(r=>parseFloat(r.final_period_hr))].sort((a,b)=>a-b);
  const medVal = med[Math.floor(med.length/2)]?.toFixed(2);
  const cards=[
    {n:catalog.length,  label:"Observed",       sub:"total asteroids",   c:T.accent},
    {n:wp.length,       label:"Periods found",   sub:`of ${catalog.length} observed`, c:T.green},
    {n:con.length,      label:"Confirmed",       sub:"high confidence",   c:T.amber},
    {n:r3.length,       label:"R=3 detections",  sub:"all 3 methods agree",c:T.green},
    {n:fast.length,     label:"Fast rotators",   sub:"P < 2.2 hr",        c:T.red},
    {n:medVal?`${medVal}h`:"—",label:"Median period",sub:"detected objects",c:T.purple,str:true},
  ];
  return (
    <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(145px,1fr))",
      gap:12,marginBottom:20}}>
      {cards.map(({n,label,sub,c,str})=>(
        <div key={label} style={{background:T.bgCard,border:`1px solid ${T.border}`,
          borderRadius:10,padding:"14px 16px",boxShadow:T.shadow}}>
          <div style={{fontFamily:"JetBrains Mono,monospace",
            fontSize:str?"1.15rem":"1.5rem",fontWeight:700,color:c,lineHeight:1}}>{n}</div>
          <div style={{fontSize:"0.72rem",fontWeight:600,color:T.textPri,
            marginTop:4,fontFamily:"Inter,sans-serif"}}>{label}</div>
          <div style={{fontSize:"0.61rem",color:T.textDim,marginTop:2}}>{sub}</div>
        </div>
      ))}
    </div>
  );
}

// ── 2. Spin rate histogram ─────────────────────────────────
function SpinHistogram({catalog}) {
  const periods = catalog
    .filter(r=>r.final_period_hr&&r.final_period_hr!=="")
    .map(r=>parseFloat(r.final_period_hr))
    .filter(p=>!isNaN(p)&&p>0&&p<=25);

  const traces = [{
    x:periods, type:"histogram",
    xbins:{start:0,end:25,size:0.5},
    marker:{color:T.accentSoft,line:{color:T.accent,width:1}},
    hovertemplate:"%{y} asteroids in this bin<extra></extra>",
  }];

  return (
    <Card title="Spin Rate Distribution"
      subtitle={`${periods.length} objects with detected periods · 0.5 hr bins`}>
      <PlotDiv traces={traces} height={210} layout={{
        xaxis:{...GRID,title:{text:"Rotation period (hr)"},range:[0,25]},
        yaxis:{...GRID,title:{text:"Count"}},
        shapes:[{type:"line",x0:2.2,x1:2.2,y0:0,y1:1,yref:"paper",
          line:{color:T.red,width:1.5,dash:"dash"},opacity:0.7}],
        annotations:[{x:2.4,y:0.97,xref:"x",yref:"paper",
          text:"2.2 hr spin barrier",showarrow:false,
          font:{size:9,color:T.red},xanchor:"left"}],
      }}/>
    </Card>
  );
}

// ── 3. Period vs SNR scatter ───────────────────────────────
function PeriodVsSNR({catalog,onSelect}) {
  const rows = catalog.filter(r=>
    r.final_period_hr&&r.final_period_hr!==""&&
    r.t1_snr&&r.t1_snr!==""&&String(r.r_code)!=="0"
  );
  const groups=["3","2","1","-1"];
  const traces=groups.map(rc=>{
    const sub=rows.filter(r=>String(r.r_code)===rc);
    return {
      x:sub.map(r=>parseFloat(r.final_period_hr)),
      y:sub.map(r=>parseFloat(r.t1_snr)),
      text:sub.map(r=>r.provid),
      customdata:sub,
      mode:"markers",
      marker:{color:R_COLOR[rc],size:7,opacity:0.85,
        line:{color:"rgba(255,255,255,0.5)",width:0.8}},
      name:R_LABEL[rc],type:"scatter",
      hovertemplate:"<b>%{text}</b><br>P = %{x:.3f} hr<br>SNR = %{y:.1f}<extra></extra>",
    };
  }).filter(t=>t.x.length>0);

  return (
    <Card title="Period vs Signal-to-Noise Ratio"
      subtitle="Higher SNR = more robust detection · click any point to view detail">
      <div style={{padding:"4px 18px 0",display:"flex",gap:12,flexWrap:"wrap"}}>
        {groups.filter(rc=>traces.find(t=>t.name===R_LABEL[rc])).map(rc=>(
          <span key={rc} style={{display:"flex",alignItems:"center",gap:5,
            fontSize:"0.65rem",color:T.textSec}}>
            <span style={{width:8,height:8,borderRadius:"50%",
              background:R_COLOR[rc],display:"inline-block"}}/>
            {R_LABEL[rc]}
          </span>
        ))}
      </div>
      <PlotDiv traces={traces} height={230} layout={{
        margin:{t:8,b:44,l:58,r:16},
        xaxis:{...GRID,title:{text:"Period (hr)"},range:[0,25]},
        yaxis:{...GRID,title:{text:"SNR"}},
        shapes:[{type:"line",x0:2.2,x1:2.2,y0:0,y1:1,yref:"paper",
          line:{color:T.red,width:1,dash:"dash"},opacity:0.4}],
      }} onClick={onSelect}/>
    </Card>
  );
}

// ── 4. Amplitude vs period ─────────────────────────────────
function AmpVsPeriod({catalog,onSelect}) {
  const rows=catalog.filter(r=>
    r.final_period_hr&&r.final_period_hr!==""&&
    r.t2_amplitude_mag&&r.t2_amplitude_mag!==""
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
      marker:{color:R_COLOR[rc],size:7,opacity:0.85,
        line:{color:"rgba(255,255,255,0.5)",width:0.8}},
      name:R_LABEL[rc],type:"scatter",
      hovertemplate:"<b>%{text}</b><br>P = %{x:.3f} hr<br>Amp = %{y:.3f} mag<extra></extra>",
    };
  }).filter(t=>t.x.length>0);

  return (
    <Card title="Lightcurve Amplitude vs Period"
      subtitle="Shape elongation proxy · larger amplitude = more elongated body">
      <PlotDiv traces={traces} height={220} layout={{
        xaxis:{...GRID,title:{text:"Period (hr)"},range:[0,25]},
        yaxis:{...GRID,title:{text:"Amplitude (mag)"}},
      }} onClick={onSelect}/>
    </Card>
  );
}

// ── 5. R-code bar chart ────────────────────────────────────
function RCodeBars({catalog}) {
  const counts=["3","2","1","0","-1"].map(r=>({
    r,n:catalog.filter(row=>String(row.r_code)===r).length,
    color:R_COLOR[r],soft:R_SOFT[r],label:R_LABEL[r],
  }));
  const max=Math.max(...counts.map(c=>c.n));
  return (
    <Card title="Detection Reliability" subtitle="R-code breakdown across all observed objects">
      <div style={{padding:"12px 18px 16px",display:"flex",flexDirection:"column",gap:9}}>
        {counts.map(({r,n,color,soft,label})=>(
          <div key={r} style={{display:"flex",alignItems:"center",gap:10}}>
            <span style={{fontSize:"0.63rem",fontFamily:"JetBrains Mono,monospace",
              fontWeight:700,color,background:soft,padding:"1px 6px",
              borderRadius:3,minWidth:30,textAlign:"center"}}>R={r}</span>
            <div style={{flex:1,background:"#f1f5f9",borderRadius:4,height:16,overflow:"hidden"}}>
              <div style={{width:`${(n/max)*100}%`,height:"100%",
                background:color,opacity:0.75,borderRadius:4,
                transition:"width 0.5s ease"}}/>
            </div>
            <span style={{fontSize:"0.7rem",fontFamily:"JetBrains Mono,monospace",
              color:T.textSec,minWidth:20,textAlign:"right"}}>{n}</span>
            <span style={{fontSize:"0.62rem",color:T.textDim,minWidth:100}}>
              {label.split(" ").slice(1).join(" ")}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ── 6. Fast rotators compact table (collapsible) ───────────
function FastRotators({catalog,onSelect}) {
  const [open,setOpen]=useState(false);
  const fast=catalog
    .filter(r=>r.final_period_hr&&parseFloat(r.final_period_hr)<2.2)
    .sort((a,b)=>parseFloat(a.final_period_hr)-parseFloat(b.final_period_hr));
  if (!fast.length) return null;

  const ultrafast=fast.filter(r=>parseFloat(r.final_period_hr)<0.5).length;

  return (
    <Card style={{marginBottom:16}}>
      <div
        onClick={()=>setOpen(o=>!o)}
        style={{display:"flex",alignItems:"center",gap:10,padding:"12px 18px",
          cursor:"pointer",userSelect:"none"}}>
        <span style={{fontSize:"0.85rem"}}>⚡</span>
        <div style={{flex:1}}>
          <span style={{fontSize:"0.78rem",fontWeight:600,color:T.textPri,
            fontFamily:"Inter,sans-serif"}}>
            Fast Rotators — P &lt; 2.2 hr spin barrier
          </span>
          <span style={{fontSize:"0.65rem",color:T.textDim,marginLeft:10}}>
            {fast.length} objects · {ultrafast} ultra-fast (P &lt; 0.5 hr)
          </span>
        </div>
        <span style={{fontSize:"0.72rem",color:T.textDim,transform:open?"rotate(90deg)":"none",
          transition:"transform 0.2s"}}>▶</span>
      </div>
      {open&&(
        <div style={{borderTop:`1px solid ${T.border}`}}>
          <div style={{display:"grid",
            gridTemplateColumns:"repeat(auto-fill,minmax(240px,1fr))"}}>
            {fast.map(row=>(
              <div key={row.provid}
                onClick={()=>onSelect(row)}
                style={{display:"flex",alignItems:"center",gap:10,
                  padding:"7px 18px",cursor:"pointer",
                  borderBottom:`1px solid ${T.border}`,
                  transition:"background 0.1s"}}
                onMouseEnter={e=>e.currentTarget.style.background="#f0f4ff"}
                onMouseLeave={e=>e.currentTarget.style.background="transparent"}>
                <span style={{fontFamily:"JetBrains Mono,monospace",
                  fontSize:"0.75rem",fontWeight:600,color:T.textPri,minWidth:84}}>
                  {row.provid}
                </span>
                <span style={{fontFamily:"JetBrains Mono,monospace",
                  fontSize:"0.75rem",color:T.accent,minWidth:60}}>
                  {parseFloat(row.final_period_hr).toFixed(3)} hr
                </span>
                <span style={{fontSize:"0.63rem",fontWeight:600,
                  color:R_COLOR[String(row.r_code)],
                  background:R_SOFT[String(row.r_code)],
                  padding:"1px 5px",borderRadius:3,
                  fontFamily:"JetBrains Mono,monospace"}}>
                  R={row.r_code}
                </span>
                <span style={{fontSize:"0.62rem",color:T.textDim}}>
                  {parseFloat(row.final_period_hr)<0.5?"ultra-fast":"superfast"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

// ── 7. Regime + pipeline note ──────────────────────────────
function DataNote({catalog}) {
  const dense=catalog.filter(r=>r.regime==="dense").length;
  const sparse=catalog.filter(r=>r.regime==="sparse").length;
  return (
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
      <Card title="Observation Regime" subtitle="Data density classification">
        <div style={{padding:"10px 18px 14px",display:"flex",flexDirection:"column",gap:8}}>
          {[["dense",dense],["sparse",sparse]].map(([regime,n])=>(
            <div key={regime} style={{display:"flex",alignItems:"center",gap:10}}>
              <span style={{fontSize:"0.65rem",fontWeight:500,color:T.textSec,
                minWidth:50,fontFamily:"Inter,sans-serif"}}>{regime}</span>
              <div style={{flex:1,background:"#f1f5f9",borderRadius:4,height:14,overflow:"hidden"}}>
                <div style={{width:`${(n/catalog.length)*100}%`,height:"100%",
                  background:T.accent,opacity:0.6,borderRadius:4}}/>
              </div>
              <span style={{fontSize:"0.68rem",fontFamily:"JetBrains Mono,monospace",
                color:T.textSec,minWidth:20,textAlign:"right"}}>{n}</span>
            </div>
          ))}
        </div>
      </Card>
      <Card title="Pipeline" subtitle="Method coverage">
        <div style={{padding:"10px 18px 14px",display:"flex",flexDirection:"column",gap:6}}>
          {[
            ["GLS + MBLS (Tier 1)","Fast scan, all objects"],
            ["MHAOV + MBLS (Tier 2)","High-order, all objects"],
            ["CE (Tier 3)","Conditional entropy, P > 1hr"],
          ].map(([name,desc])=>(
            <div key={name} style={{fontSize:"0.65rem",fontFamily:"Inter,sans-serif"}}>
              <span style={{fontWeight:600,color:T.textSec}}>{name}</span>
              <span style={{color:T.textDim}}> — {desc}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ── Main ───────────────────────────────────────────────────
export default function Explore({catalog,onSelect}) {
  if (!catalog.length) return (
    <div style={{padding:"48px 0",textAlign:"center",color:T.textDim}}>
      Loading catalog…
    </div>
  );

  return (
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      <SummaryCards catalog={catalog}/>
      <FastRotators catalog={catalog} onSelect={onSelect}/>

      {/* Row 1: spin histogram + reliability */}
      <div style={{display:"grid",gridTemplateColumns:"1.2fr 1fr",gap:16}}>
        <SpinHistogram catalog={catalog}/>
        <RCodeBars catalog={catalog}/>
      </div>

      {/* Row 2: period vs SNR full width */}
      <PeriodVsSNR catalog={catalog} onSelect={onSelect}/>

      {/* Row 3: amplitude vs period + regime */}
      <div style={{display:"grid",gridTemplateColumns:"1.4fr 1fr",gap:16}}>
        <AmpVsPeriod catalog={catalog} onSelect={onSelect}/>
        <DataNote catalog={catalog}/>
      </div>

      {/* Footer note */}
      <div style={{padding:"10px 14px",background:T.bgCard,
        border:`1px solid ${T.border}`,borderRadius:8,
        fontSize:"0.67rem",color:T.textDim,lineHeight:1.7}}>
        <strong style={{color:T.textSec}}>Data:</strong>{" "}
        Rubin/LSST commissioning, Apr–May 2025 · 76 asteroids · 1.45M observations ·
        12-day baseline · g,r,i bands · 3-tier pipeline (GLS, MBLS, MHAOV, CE).{" "}
        <a href="https://github.com/wonrobot/asteroid-pipeline" target="_blank"
          rel="noreferrer"
          style={{color:T.accent,textDecoration:"none",float:"right"}}>
          GitHub ↗
        </a>
      </div>
    </div>
  );
}
