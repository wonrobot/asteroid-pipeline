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

const R_COLOR  = {"3":T.green,  "2":T.amber,  "1":T.yellow, "0":T.red,   "-1":T.purple};
const R_SOFT   = {"3":T.greenSoft,"2":T.amberSoft,"1":T.yellowSoft,"0":T.redSoft,"-1":T.purpleSoft};
const R_LABEL  = {"3":"R=3 Confirmed","2":"R=2 Moderate","1":"R=1 Tentative","0":"R=0 No period","-1":"R=−1 Alias"};

function usePlotly(ref, traces, layout, deps) {
  useEffect(() => {
    if (!ref.current || !traces) return;
    try {
      Plotly.react(ref.current, traces, {
        paper_bgcolor:"transparent", plot_bgcolor:"#fafbff",
        font:{family:"Inter,sans-serif", color:"#64748b", size:11},
        showlegend:false, hovermode:"closest",
        margin:{t:32,b:48,l:58,r:16},
        ...layout,
      }, {displayModeBar:false, responsive:true});
    } catch(e) {}
  }, deps);
}

function PlotCard({ title, subtitle, children, style }) {
  return (
    <div style={{background:T.bgCard, border:`1px solid ${T.border}`,
      borderRadius:10, overflow:"hidden",
      boxShadow:T.shadow, ...style}}>
      <div style={{padding:"14px 18px 0"}}>
        <div style={{fontSize:"0.78rem",fontWeight:600,color:T.textPri,
          fontFamily:"Inter,sans-serif"}}>{title}</div>
        {subtitle&&<div style={{fontSize:"0.65rem",color:T.textDim,marginTop:2}}>{subtitle}</div>}
      </div>
      {children}
    </div>
  );
}

// ── Spin rate histogram ──────────────────────────────────────
function SpinHistogram({ catalog, onSelect }) {
  const ref = useRef(null);
  const periods = catalog
    .filter(r=>r.final_period_hr&&r.final_period_hr!=="")
    .map(r=>parseFloat(r.final_period_hr))
    .filter(p=>!isNaN(p)&&p>0&&p<25);

  usePlotly(ref, [{
    x: periods, type:"histogram",
    xbins:{start:0,end:25,size:0.5},
    marker:{color:T.accentSoft, line:{color:T.accent,width:1}},
    hovertemplate:"Period %{x:.1f}–%{x:.1f}h<br>%{y} asteroids<extra></extra>",
  }], {
    title:{text:"",},
    xaxis:{title:{text:"Rotation period (hr)"},gridcolor:"#f1f5f9",linecolor:"#e2e8f0"},
    yaxis:{title:{text:"Count"},gridcolor:"#f1f5f9",linecolor:"#e2e8f0"},
    shapes:[
      // Spin barrier at 2.2hr
      {type:"line",x0:2.2,x1:2.2,y0:0,y1:1,yref:"paper",
        line:{color:T.red,width:1.5,dash:"dash"},opacity:0.7},
    ],
    annotations:[
      {x:2.2,y:1,xref:"x",yref:"paper",text:"2.2hr barrier",
        showarrow:false,font:{size:9,color:T.red},
        xanchor:"left",yanchor:"top",xshift:4},
    ],
  }, [catalog]);

  return (
    <PlotCard title="Spin Rate Distribution"
      subtitle={`${periods.length} objects with detected periods`}>
      <div ref={ref} style={{width:"100%",height:220}}/>
    </PlotCard>
  );
}

// ── Period vs H scatter ──────────────────────────────────────
function PeriodVsH({ catalog, onSelect }) {
  const ref = useRef(null);
  const groups = ["-1","0","1","2","3"];

  const traces = groups.map(r => {
    const rows = catalog.filter(row=>String(row.r_code)===r&&row.final_period_hr&&row.final_period_hr!=="");
    return {
      x: rows.map(row=>parseFloat(row.final_period_hr)),
      y: rows.map(row=>parseFloat(row.h_mag||row.H||20)),
      text: rows.map(row=>row.provid),
      customdata: rows,
      mode:"markers",
      marker:{color:R_COLOR[r],size:7,opacity:0.85,
        line:{color:"rgba(255,255,255,0.6)",width:0.8}},
      name:R_LABEL[r], type:"scatter",
      hovertemplate:"<b>%{text}</b><br>P = %{x:.3f} hr<br>H = %{y:.1f}<extra></extra>",
    };
  }).filter(t=>t.x.length>0);

  useEffect(()=>{
    if (!ref.current) return;
    try {
      Plotly.react(ref.current, traces, {
        paper_bgcolor:"transparent", plot_bgcolor:"#fafbff",
        font:{family:"Inter,sans-serif",color:"#64748b",size:11},
        showlegend:true,
        legend:{bgcolor:"rgba(255,255,255,0.9)",bordercolor:"#e2e8f0",
          borderwidth:1,font:{size:10},x:1,xanchor:"right",y:1},
        hovermode:"closest",
        margin:{t:32,b:48,l:58,r:140},
        xaxis:{title:{text:"Period (hr)"},gridcolor:"#f1f5f9",linecolor:"#e2e8f0",
          range:[0,25]},
        yaxis:{title:{text:"Absolute magnitude H"},gridcolor:"#f1f5f9",
          linecolor:"#e2e8f0",autorange:"reversed"},
        shapes:[
          {type:"line",x0:2.2,x1:2.2,y0:0,y1:1,yref:"paper",
            line:{color:T.red,width:1,dash:"dash"},opacity:0.5},
        ],
      },{displayModeBar:false,responsive:true});

      ref.current.on("plotly_click", e=>{
        const row = e.points[0]?.customdata;
        if (row) onSelect(row);
      });
    } catch(e) {}
  }, [catalog]);

  return (
    <PlotCard title="Period vs Absolute Magnitude (H)"
      subtitle="Click any point to view asteroid detail · H proxy from designation order">
      <div ref={ref} style={{width:"100%",height:280}}/>
    </PlotCard>
  );
}

// ── Amplitude vs period ──────────────────────────────────────
function AmpVsPeriod({ catalog, onSelect }) {
  const ref = useRef(null);

  useEffect(()=>{
    const rows = catalog.filter(r=>
      r.final_period_hr&&r.final_period_hr!==""&&
      r.t2_amplitude_mag&&r.t2_amplitude_mag!==""
    );
    const groups = ["-1","0","1","2","3"];
    const traces = groups.map(rc=>{
      const sub = rows.filter(r=>String(r.r_code)===rc);
      return {
        x:sub.map(r=>parseFloat(r.final_period_hr)),
        y:sub.map(r=>parseFloat(r.t2_amplitude_mag)),
        text:sub.map(r=>r.provid),
        customdata:sub,
        mode:"markers",
        marker:{color:R_COLOR[rc],size:7,opacity:0.85,
          line:{color:"rgba(255,255,255,0.6)",width:0.8}},
        name:R_LABEL[rc],type:"scatter",
        hovertemplate:"<b>%{text}</b><br>P = %{x:.3f} hr<br>Amp = %{y:.3f} mag<extra></extra>",
      };
    }).filter(t=>t.x.length>0);

    if (!ref.current) return;
    try {
      Plotly.react(ref.current, traces, {
        paper_bgcolor:"transparent",plot_bgcolor:"#fafbff",
        font:{family:"Inter,sans-serif",color:"#64748b",size:11},
        showlegend:false, hovermode:"closest",
        margin:{t:32,b:48,l:58,r:16},
        xaxis:{title:{text:"Period (hr)"},gridcolor:"#f1f5f9",
          linecolor:"#e2e8f0",range:[0,25]},
        yaxis:{title:{text:"Amplitude (mag)"},gridcolor:"#f1f5f9",linecolor:"#e2e8f0"},
      },{displayModeBar:false,responsive:true});
      ref.current.on("plotly_click",e=>{
        const row=e.points[0]?.customdata;
        if(row) onSelect(row);
      });
    } catch(e){}
  },[catalog]);

  return (
    <PlotCard title="Lightcurve Amplitude vs Period"
      subtitle="Elongated/tumbling bodies tend toward larger amplitudes">
      <div ref={ref} style={{width:"100%",height:240}}/>
    </PlotCard>
  );
}

// ── R-code breakdown bar ──────────────────────────────────────
function RCodeBar({ catalog }) {
  const counts = ["3","2","1","0","-1"].map(r=>({
    r, n: catalog.filter(row=>String(row.r_code)===r).length,
    label:R_LABEL[r], color:R_COLOR[r], soft:R_SOFT[r],
  }));
  const max = Math.max(...counts.map(c=>c.n));
  return (
    <PlotCard title="Reliability Distribution" subtitle="R-code breakdown">
      <div style={{padding:"12px 18px 16px",display:"flex",flexDirection:"column",gap:8}}>
        {counts.map(({r,n,label,color,soft})=>(
          <div key={r} style={{display:"flex",alignItems:"center",gap:10}}>
            <span style={{fontSize:"0.65rem",fontFamily:"JetBrains Mono,monospace",
              fontWeight:600,color,background:soft,padding:"1px 6px",
              borderRadius:3,minWidth:32,textAlign:"center"}}>R={r}</span>
            <div style={{flex:1,background:"#f1f5f9",borderRadius:3,height:14,overflow:"hidden"}}>
              <div style={{width:`${(n/max)*100}%`,height:"100%",
                background:color,borderRadius:3,opacity:0.8,
                transition:"width 0.4s ease"}}/>
            </div>
            <span style={{fontSize:"0.68rem",fontFamily:"JetBrains Mono,monospace",
              color:T.textSec,minWidth:24,textAlign:"right"}}>{n}</span>
            <span style={{fontSize:"0.63rem",color:T.textDim,minWidth:80}}>{label.split(" ").slice(1).join(" ")}</span>
          </div>
        ))}
      </div>
    </PlotCard>
  );
}

// ── Regime breakdown ──────────────────────────────────────────
function RegimeBreakdown({ catalog }) {
  const regimes = [...new Set(catalog.map(r=>r.regime).filter(Boolean))];
  const counts = regimes.map(regime=>({
    regime,
    n: catalog.filter(r=>r.regime===regime).length,
    confirmed: catalog.filter(r=>r.regime===regime&&r.reliability==="confirmed").length,
  })).sort((a,b)=>b.n-a.n);
  const max = Math.max(...counts.map(c=>c.n));

  return (
    <PlotCard title="Observation Regime" subtitle="Data density classification">
      <div style={{padding:"12px 18px 16px",display:"flex",flexDirection:"column",gap:8}}>
        {counts.map(({regime,n,confirmed})=>(
          <div key={regime} style={{display:"flex",alignItems:"center",gap:10}}>
            <span style={{fontSize:"0.65rem",color:T.textSec,minWidth:80,
              fontFamily:"Inter,sans-serif",fontWeight:500}}>{regime}</span>
            <div style={{flex:1,background:"#f1f5f9",borderRadius:3,height:14,
              overflow:"hidden",position:"relative"}}>
              <div style={{width:`${(n/max)*100}%`,height:"100%",
                background:T.accentSoft,borderRadius:3}}/>
              <div style={{width:`${(confirmed/max)*100}%`,height:"50%",
                background:T.accent,borderRadius:"3px 3px 0 0",
                position:"absolute",top:0,left:0,opacity:0.7}}/>
            </div>
            <span style={{fontSize:"0.68rem",fontFamily:"JetBrains Mono,monospace",
              color:T.textSec,minWidth:16}}>{n}</span>
            <span style={{fontSize:"0.63rem",color:T.textDim}}>
              ({confirmed} confirmed)
            </span>
          </div>
        ))}
      </div>
    </PlotCard>
  );
}

// ── Fast rotator highlight ────────────────────────────────────
function FastRotators({ catalog, onSelect }) {
  const fast = catalog
    .filter(r=>r.final_period_hr&&parseFloat(r.final_period_hr)<2.2)
    .sort((a,b)=>parseFloat(a.final_period_hr)-parseFloat(b.final_period_hr));

  if (!fast.length) return null;

  return (
    <PlotCard title={`⚡ Fast Rotators — P < 2.2 hr spin barrier`}
      subtitle={`${fast.length} objects spinning faster than the rubble-pile disruption limit`}>
      <div style={{padding:"8px 18px 14px",display:"flex",flexDirection:"column",gap:4}}>
        {fast.map(row=>(
          <div key={row.provid}
            onClick={()=>onSelect(row)}
            style={{display:"flex",alignItems:"center",gap:12,padding:"6px 8px",
              borderRadius:6,cursor:"pointer",transition:"background 0.1s",
              background:"transparent"}}
            onMouseEnter={e=>e.currentTarget.style.background="#f0f4ff"}
            onMouseLeave={e=>e.currentTarget.style.background="transparent"}>
            <span style={{fontFamily:"JetBrains Mono,monospace",fontSize:"0.78rem",
              fontWeight:600,color:T.textPri,minWidth:90}}>{row.provid}</span>
            <span style={{fontFamily:"JetBrains Mono,monospace",fontSize:"0.78rem",
              color:T.accent,minWidth:70}}>
              {parseFloat(row.final_period_hr).toFixed(3)} hr
            </span>
            <span style={{fontSize:"0.68rem",fontWeight:600,
              color:R_COLOR[String(row.r_code)],
              background:R_SOFT[String(row.r_code)],
              padding:"1px 6px",borderRadius:3,fontFamily:"JetBrains Mono,monospace"}}>
              R={row.r_code}
            </span>
            <span style={{fontSize:"0.68rem",color:T.textDim}}>
              {parseFloat(row.final_period_hr)<0.5?"ultra-fast":"superfast"}
            </span>
          </div>
        ))}
      </div>
    </PlotCard>
  );
}

// ── Science summary cards ─────────────────────────────────────
function SummaryCards({ catalog }) {
  const withPeriod = catalog.filter(r=>r.final_period_hr&&r.final_period_hr!=="");
  const confirmed  = catalog.filter(r=>r.reliability==="confirmed");
  const fast       = withPeriod.filter(r=>parseFloat(r.final_period_hr)<2.2);
  const median     = [...withPeriod.map(r=>parseFloat(r.final_period_hr))]
    .sort((a,b)=>a-b)[Math.floor(withPeriod.length/2)]?.toFixed(2);
  const r3         = catalog.filter(r=>String(r.r_code)==="3");

  const cards = [
    {n:catalog.length,   label:"Observed",      sub:"total asteroids",   color:T.accent},
    {n:withPeriod.length,label:"Periods found",  sub:`of ${catalog.length} observed`, color:T.green},
    {n:confirmed.length, label:"Confirmed",      sub:"R=3 or R=2",        color:T.amber},
    {n:fast.length,      label:"Fast rotators",  sub:"P < 2.2 hr",        color:T.red},
    {n:median?`${median}h`:"—", label:"Median period", sub:"detected objects", color:T.purple, isStr:true},
    {n:r3.length,        label:"High confidence",sub:"R=3 detections",    color:T.green},
  ];

  return (
    <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(150px,1fr))",
      gap:12,marginBottom:24}}>
      {cards.map(({n,label,sub,color,isStr})=>(
        <div key={label} style={{background:T.bgCard,border:`1px solid ${T.border}`,
          borderRadius:10,padding:"14px 16px",boxShadow:T.shadow}}>
          <div style={{fontFamily:"JetBrains Mono,monospace",fontSize:isStr?"1.2rem":"1.5rem",
            fontWeight:700,color,lineHeight:1}}>{n}</div>
          <div style={{fontSize:"0.72rem",fontWeight:600,color:T.textPri,
            marginTop:4,fontFamily:"Inter,sans-serif"}}>{label}</div>
          <div style={{fontSize:"0.62rem",color:T.textDim,marginTop:2}}>{sub}</div>
        </div>
      ))}
    </div>
  );
}

// ── Main Explore component ────────────────────────────────────
export default function Explore({ catalog, onSelect }) {
  if (!catalog.length) return (
    <div style={{padding:"48px 0",textAlign:"center",color:T.textDim}}>
      Loading catalog…
    </div>
  );

  return (
    <div>
      <SummaryCards catalog={catalog}/>

      {/* Fast rotators first — most interesting */}
      <FastRotators catalog={catalog} onSelect={onSelect}/>

      {/* Main science plots */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginTop:16}}>
        <SpinHistogram catalog={catalog} onSelect={onSelect}/>
        <RCodeBar catalog={catalog}/>
      </div>

      <div style={{marginTop:16}}>
        <PeriodVsH catalog={catalog} onSelect={onSelect}/>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginTop:16}}>
        <AmpVsPeriod catalog={catalog} onSelect={onSelect}/>
        <RegimeBreakdown catalog={catalog}/>
      </div>

      <div style={{marginTop:20,padding:"12px 16px",background:T.bgCard,
        border:`1px solid ${T.border}`,borderRadius:8,
        fontSize:"0.68rem",color:T.textDim,lineHeight:1.6}}>
        <strong style={{color:T.textSec}}>Data:</strong> Rubin/LSST commissioning photometry, Apr–May 2025.
        76 asteroids · 1.45M observations · 12-day baseline · g,r,i bands.
        Pipeline: 3-tier multi-method (GLS, MBLS, MHAOV, CE) with R-code reliability system.
        <span style={{float:"right"}}>
          <a href="https://github.com/wonrobot/asteroid-pipeline" target="_blank"
            rel="noreferrer" style={{color:T.accent,textDecoration:"none"}}>
            GitHub ↗
          </a>
        </span>
      </div>
    </div>
  );
}
