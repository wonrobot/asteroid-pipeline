import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist";

const BC = { g:"#22c55e", r:"#f97316", i:"#3b82f6" };
const NC = ["#e63946","#f4a261","#2a9d8f","#457b9d","#e9c46a","#6d6875","#a8dadc","#f77f00","#4cc9f0","#7209b7","#b5e48c","#ff6b6b","#48cae4","#f3722c","#90be6d","#43aa8b","#577590","#f9844a","#4d908e","#277da1"];

function nightMap(mjd) {
  const days = [...new Set(mjd.map(m => Math.floor(m)))].sort((a,b)=>a-b);
  const map = {};
  days.forEach((d,i) => { map[d] = i; });
  return mjd.map(m => map[Math.floor(m)]);
}

function Plot({ traces, layout }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    try {
      Plotly.react(ref.current, traces, layout, { displayModeBar:false, responsive:true });
    } catch(e) { console.warn("Plotly error", e); }
  }, [traces, layout]);
  return <div ref={ref} style={{width:"100%", height:280}} />;
}

const BASE = {
  paper_bgcolor:"transparent", plot_bgcolor:"#fafbff",
  font:{family:"Inter,sans-serif", color:"#475569", size:11},
  margin:{t:44,b:52,l:62,r:20},
  legend:{bgcolor:"rgba(255,255,255,0.85)", bordercolor:"#e2e8f0", borderwidth:1, font:{size:10}},
  hovermode:"closest",
};

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
  if (err) return <p style={{color:"#94a3b8",fontSize:"0.78rem",marginTop:12}}>No chart data available.</p>;
  if (!d)  return <p style={{color:"#94a3b8",fontSize:"0.78rem",marginTop:12}}>Loading…</p>;

  const ni    = nightMap(d.obs.mjd);
  const nights = [...new Set(ni)];
  const bands  = [...new Set(d.obs.band)];
  const P      = d.best_period;

  /* Lightcurve — one trace per night */
  const lcTraces = nights.map(n => {
    const idx = ni.map((v,i)=>v===n?i:-1).filter(i=>i>=0);
    return {
      x: idx.map(i=>d.obs.mjd[i]), y: idx.map(i=>d.obs.mag[i]),
      error_y:{type:"data", array:idx.map(i=>d.obs.magerr[i]), visible:true, color:NC[n%20]+"55", thickness:1},
      mode:"markers",
      marker:{color:NC[n%20], size:4,
        symbol:idx.map(i=>d.obs.band[i]==="g"?"circle":d.obs.band[i]==="r"?"square":"diamond")},
      name:`Night ${n+1}`, type:"scatter",
    };
  });

  /* Periodogram */
  const pg = d.pgram;
  const maxP = Math.max(...pg.mbls_power);
  const pgramTraces = [
    {x:pg.periods, y:pg.mbls_power, mode:"lines", line:{color:"#2563eb",width:1.2}, name:"MBLS", type:"scatter"},
  ];
  if (pg.mhaov_power) {
    const s = maxP / Math.max(...pg.mhaov_power) * 0.85;
    pgramTraces.push({x:pg.periods, y:pg.mhaov_power.map(v=>v*s), mode:"lines", line:{color:"#7c3aed",width:1,dash:"dot"}, name:"MHAOV (norm.)", type:"scatter"});
  }
  if (pg.gls_power && pg.gls_periods) {
    const s = maxP / Math.max(...pg.gls_power) * 0.7;
    pgramTraces.push({x:pg.gls_periods, y:pg.gls_power.map(v=>v*s), mode:"lines", line:{color:"#059669",width:0.8,dash:"dash"}, name:"GLS (norm.)", type:"scatter"});
  }
  if (pg.window_power && pg.window_periods) {
    const s = maxP / Math.max(...pg.window_power) * 0.3;
    pgramTraces.push({x:pg.window_periods, y:pg.window_power.map(v=>v*s), mode:"lines", line:{color:"#f97316",width:0.7}, fill:"tozeroy", fillcolor:"rgba(249,115,22,0.07)", name:"Window (norm.)", type:"scatter"});
  }
  // Period marker as a scatter trace (vertical line workaround)
  if (P) {
    pgramTraces.push({x:[P,P], y:[0,maxP], mode:"lines", line:{color:"#2563eb",width:2,dash:"solid"}, name:`P=${P.toFixed(3)}h`, type:"scatter", showlegend:true});
    if (P/2 > Math.min(...pg.periods))
      pgramTraces.push({x:[P/2,P/2], y:[0,maxP], mode:"lines", line:{color:"#f59e0b",width:1.5,dash:"dot"}, name:`P/2=${(P/2).toFixed(3)}h`, type:"scatter"});
    if (P*2 < Math.max(...pg.periods))
      pgramTraces.push({x:[P*2,P*2], y:[0,maxP], mode:"lines", line:{color:"#f59e0b",width:1.5,dash:"dot"}, name:`2P=${(P*2).toFixed(3)}h`, type:"scatter"});
    [12,24].forEach(alias => {
      const mn = Math.min(...pg.periods), mx = Math.max(...pg.periods);
      if (alias>=mn && alias<=mx)
        pgramTraces.push({x:[alias,alias], y:[0,maxP], mode:"lines", line:{color:"#dc2626",width:1,dash:"dashdot"}, name:`${alias}h alias`, type:"scatter"});
    });
  }

  /* Phase fold */
  const foldTraces = d.fold ? [
    ...bands.map(b => ({
      x: d.fold.phase.filter((_,i)=>d.fold.band[i]===b),
      y: d.fold.mag.filter((_,i)=>d.fold.band[i]===b),
      error_y:{type:"data", array:d.fold.magerr.filter((_,i)=>d.fold.band[i]===b), visible:true, color:BC[b]+"55", thickness:1},
      mode:"markers", marker:{color:BC[b],size:3.5,opacity:0.85},
      name:b+"-band", type:"scatter",
    })),
    {x:d.fold.fitted_phase, y:d.fold.fitted_mag, mode:"lines", line:{color:"#0f172a",width:2}, name:"model", type:"scatter"},
  ] : null;

  const residTraces = d.fold ? bands.map(b => {
    const fp = d.fold.fitted_phase, fm = d.fold.fitted_mag;
    const interp = ph => fm[fp.reduce((bi,_,i)=>Math.abs(fp[i]-ph)<Math.abs(fp[bi]-ph)?i:bi,0)];
    const phases = d.fold.phase.filter((_,i)=>d.fold.band[i]===b);
    const mags   = d.fold.mag.filter((_,i)=>d.fold.band[i]===b);
    return {x:phases, y:mags.map((m,i)=>m-interp(phases[i])), mode:"markers", marker:{color:BC[b],size:3,opacity:0.7}, name:b, type:"scatter", showlegend:false};
  }) : null;

  const tabs = [
    {id:"lc", label:"Lightcurve"},
    {id:"pgram", label:"Periodogram"},
    ...(d.fold ? [{id:"fold", label:"Phase Fold"}] : []),
  ];

  const S = {
    bar:{display:"flex",gap:4,alignItems:"center",marginBottom:8,flexWrap:"wrap"},
    tab:{fontFamily:"Inter,sans-serif",fontSize:"0.72rem",fontWeight:500,padding:"4px 14px",border:"1px solid #dde3f0",borderRadius:5,background:"#f7f8fc",color:"#475569",cursor:"pointer"},
    on:{background:"#2563eb",color:"#fff",borderColor:"#2563eb"},
    pill:{marginLeft:"auto",fontFamily:"JetBrains Mono,monospace",fontSize:"0.72rem",color:"#2563eb",background:"#dbeafe",padding:"3px 10px",borderRadius:4,fontWeight:600},
  };

  return (
    <div style={{marginTop:"1rem"}}>
      <div style={S.bar}>
        {tabs.map(t=><button key={t.id} style={{...S.tab,...(tab===t.id?S.on:{})}} onClick={()=>setTab(t.id)}>{t.label}</button>)}
        {P && <span style={S.pill}>P = {P.toFixed(4)} hr</span>}
      </div>

      {tab==="lc" && <Plot traces={lcTraces} layout={{...BASE,
        title:{text:"Lightcurve (colored by night, ○g □r ◇i)",font:{size:12,color:"#0f172a"}},
        xaxis:{title:{text:"MJD"},gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
        yaxis:{title:{text:"mag"},autorange:"reversed",gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
      }}/>}

      {tab==="pgram" && <Plot traces={pgramTraces} layout={{...BASE, height:300,
        title:{text:"Periodogram — MBLS · MHAOV · GLS · window function",font:{size:12,color:"#0f172a"}},
        xaxis:{title:{text:"Period (hr)"},type:"log",gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
        yaxis:{title:{text:"Power (MBLS scale)"},gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
      }}/>}

      {tab==="fold" && d.fold && <>
        <Plot traces={foldTraces} layout={{...BASE,
          title:{text:`Phase fold — 2 cycles  (P = ${P?.toFixed(4)} hr)`,font:{size:12,color:"#0f172a"}},
          xaxis:{title:{text:"Phase"},range:[0,2],dtick:0.25,gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
          yaxis:{title:{text:"Δmag (detrended)"},autorange:"reversed",gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
        }}/>
        <Plot traces={residTraces} layout={{...BASE, margin:{t:28,b:44,l:62,r:20},
          title:{text:"Residuals (O−C)",font:{size:11,color:"#0f172a"}},
          xaxis:{title:{text:"Phase"},range:[0,2],dtick:0.25,gridcolor:"#e8edf5",linecolor:"#cbd5e1"},
          yaxis:{title:{text:"O−C (mag)"},zeroline:true,zerolinecolor:"#475569",gridcolor:"#e8edf5"},
        }}/>
      </>}
    </div>
  );
}
