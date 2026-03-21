import { useState, useEffect, useRef } from 'react'
import Papa from 'papaparse'
import AsteroidCharts from './components/AsteroidCharts'
import Explore from './components/Explore'

/* ─────────────────────────────────────────────
   Design tokens — light scientific theme
   Inspired by Nature / A&A journal aesthetics
───────────────────────────────────────────── */
const T = {
  bg:         '#f7f8fc',
  bgCard:     '#ffffff',
  bgHover:    '#f0f4ff',
  bgSelected: '#e8efff',
  border:     '#dde3f0',
  borderStr:  '#b8c4e0',
  accent:     '#2563eb',
  accentSoft: '#dbeafe',
  green:      '#16a34a',
  greenSoft:  '#dcfce7',
  amber:      '#d97706',
  amberSoft:  '#fef3c7',
  red:        '#dc2626',
  redSoft:    '#fee2e2',
  purple:     '#7c3aed',
  purpleSoft: '#ede9fe',
  textPri:    '#0f172a',
  textSec:    '#475569',
  textDim:    '#94a3b8',
  shadow:     '0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)',
  shadowMd:   '0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.05)',
}

const css = `
  @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: ${T.bg};
    color: ${T.textPri};
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: ${T.bg}; }
  ::-webkit-scrollbar-thumb { background: ${T.border}; border-radius: 3px; }

  .root { max-width: 1120px; margin: 0 auto; padding: 0 24px 80px; }

  /* Header */
  .header {
    padding: 32px 0 28px;
    border-bottom: 1px solid ${T.border};
    margin-bottom: 28px;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 24px;
    align-items: center;
  }
  .logo-row { display: flex; align-items: baseline; gap: 14px; margin-bottom: 8px; }
  .logo {
    font-family: 'Lora', serif;
    font-size: 2.8rem;
    font-weight: 600;
    color: ${T.textPri};
    letter-spacing: -0.02em;
  }
  .logo span { color: ${T.accent}; }
  .logo-version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: ${T.textDim};
    background: ${T.border};
    padding: 2px 7px;
    border-radius: 3px;
    letter-spacing: 0.05em;
  }
  .header-tagline {
    font-size: 0.82rem;
    color: ${T.textDim};
    line-height: 1.5;
    max-width: 480px;
    margin-top: 2px;
  }
  .stats-grid {
    display: flex;
    gap: 0;
    border: 1px solid ${T.border};
    border-radius: 8px;
    overflow: hidden;
    background: ${T.bgCard};
    box-shadow: ${T.shadow};
    height: fit-content;
  }
  .stat-cell {
    padding: 12px 18px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    border-right: 1px solid ${T.border};
    min-width: 80px;
  }
  .stat-cell:last-child { border-right: none; }
  .stat-n {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: ${T.accent};
    line-height: 1;
  }
  .stat-l {
    font-size: 0.63rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: ${T.textDim};
    white-space: nowrap;
  }

  /* Detail panel */
  .detail {
    background: ${T.bgCard};
    border: 1px solid ${T.borderStr};
    border-top: 3px solid ${T.accent};
    border-radius: 0 0 8px 8px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: ${T.shadowMd};
    animation: fadeSlide 0.18s ease;
  }
  @keyframes fadeSlide {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .detail-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 16px;
  }
  .detail-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: ${T.textPri};
  }
  .detail-period-hero {
    font-family: 'Lora', serif;
    font-size: 0.95rem;
    color: ${T.textSec};
    margin-top: 3px;
  }
  .detail-period-hero strong {
    font-size: 1.4rem;
    color: ${T.accent};
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
  }
  .btn-close {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    padding: 5px 12px;
    border: 1px solid ${T.border};
    background: ${T.bg};
    color: ${T.textSec};
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.12s;
    flex-shrink: 0;
  }
  .btn-close:hover { border-color: ${T.accent}; color: ${T.accent}; background: ${T.accentSoft}; }
  .detail-fields {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 14px 20px;
    padding: 16px 0;
    border-top: 1px solid ${T.border};
    border-bottom: 1px solid ${T.border};
    margin-bottom: 14px;
  }
  .df { display: flex; flex-direction: column; gap: 3px; }
  .df-l {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: ${T.textDim};
    font-weight: 500;
  }
  .df-v {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: ${T.textPri};
  }
  .detail-note {
    font-size: 0.8rem;
    color: ${T.textSec};
    line-height: 1.6;
    display: flex;
    gap: 8px;
    align-items: flex-start;
  }
  .detail-note::before { content: 'ℹ'; color: ${T.accent}; flex-shrink: 0; }
  .chart-ph {
    background: ${T.bg};
    border: 1px dashed ${T.borderStr};
    border-radius: 6px;
    padding: 28px;
    text-align: center;
    color: ${T.textDim};
    font-size: 0.78rem;
    margin-top: 14px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
  }

  /* Controls */
  .controls {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
    flex-wrap: wrap;
    align-items: center;
  }
  .filter-grp {
    display: flex;
    gap: 2px;
    background: ${T.bgCard};
    border: 1px solid ${T.border};
    border-radius: 6px;
    padding: 3px;
    box-shadow: ${T.shadow};
  }
  .fbtn {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 5px 13px;
    border: none;
    background: none;
    color: ${T.textSec};
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.12s;
    white-space: nowrap;
  }
  .fbtn:hover { background: ${T.bgHover}; color: ${T.accent}; }
  .fbtn.on { background: ${T.accent}; color: #fff; }

  /* Period slider */
  .slider-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    background: ${T.bgCard};
    border: 1px solid ${T.border};
    border-radius: 6px;
    padding: 6px 14px;
    box-shadow: ${T.shadow};
    flex: 1;
    min-width: 260px;
    max-width: 380px;
  }
  .slider-lbl { font-size: 0.68rem; color: ${T.textDim}; white-space: nowrap; text-transform: uppercase; letter-spacing: 0.06em; }
  .slider-val { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: ${T.accent}; white-space: nowrap; min-width: 76px; text-align: right; }
  input[type=range] { flex: 1; accent-color: ${T.accent}; cursor: pointer; }
  .count-label { margin-left: auto; font-size: 0.72rem; color: ${T.textDim}; font-family: 'JetBrains Mono', monospace; white-space: nowrap; }

  /* Legend */
  .legend {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 12px;
    padding: 9px 14px;
    background: ${T.bgCard};
    border: 1px solid ${T.border};
    border-radius: 6px;
    box-shadow: ${T.shadow};
    align-items: center;
  }
  .legend-lbl { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.08em; color: ${T.textDim}; font-weight: 600; }

  /* Table */
  .tbl-wrap {
    background: ${T.bgCard};
    border: 1px solid ${T.border};
    border-radius: 8px;
    overflow: hidden;
    box-shadow: ${T.shadow};
  }
  table { width: 100%; border-collapse: collapse; }
  thead tr { background: ${T.bg}; border-bottom: 1px solid ${T.borderStr}; }
  th {
    padding: 11px 16px;
    text-align: left;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: ${T.textDim};
    font-weight: 600;
  }
  tbody tr { border-bottom: 1px solid ${T.border}; cursor: pointer; transition: background 0.08s; }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: ${T.bgHover}; }
  tbody tr.sel { background: ${T.bgSelected}; }
  td { padding: 10px 16px; }
  .td-id { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 500; color: ${T.textPri}; }
  .td-p  { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: ${T.accent}; font-weight: 500; }
  .td-null { font-size: 0.8rem; color: ${T.textDim}; }
  .td-sec  { font-size: 0.78rem; color: ${T.textSec}; }

  /* R badge */
  .rb { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600; padding: 2px 7px; border-radius: 4px; letter-spacing: 0.04em; }
  .rb3  { background: ${T.greenSoft};  color: ${T.green};  }
  .rb2  { background: ${T.amberSoft};  color: ${T.amber};  }
  .rb1  { background: #fef9c3;         color: #a16207;     }
  .rb0  { background: ${T.redSoft};    color: ${T.red};    }
  .rbm1 { background: ${T.purpleSoft}; color: ${T.purple}; }

  /* Rel chip */
  .rc { display: inline-block; font-size: 0.68rem; padding: 1px 7px; border: 1px solid ${T.border}; border-radius: 3px; color: ${T.textSec}; background: ${T.bg}; font-family: 'JetBrains Mono', monospace; }

  /* Footer */
  .footer { margin-top: 56px; padding-top: 20px; border-top: 1px solid ${T.border}; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 0.72rem; color: ${T.textDim}; }
  .footer a { color: ${T.accent}; text-decoration: none; }
  .footer a:hover { text-decoration: underline; }

  @media (max-width: 700px) {
    .header { grid-template-columns: 1fr; }
    .stats-grid { width: 100%; }
    th:nth-child(4), td:nth-child(4),
    th:nth-child(5), td:nth-child(5) { display: none; }
  }
`

function RBadge({ code }) {
  const n = parseInt(code)
  const cls = n >= 3 ? 'rb rb3' : n === 2 ? 'rb rb2' : n === 1 ? 'rb rb1' : n === -1 ? 'rb rbm1' : 'rb rb0'
  return <span className={cls}>R={code}</span>
}

function scienceNote(row) {
  const p = parseFloat(row.final_period_hr)
  if (!p) return 'Insufficient signal for period detection with the 12-day baseline. Follow-up observations recommended.'
  if (p < 2.2) return `Superfast rotator (P = ${p.toFixed(3)} hr) — spinning faster than the 2.2 hr spin barrier. These objects are likely monolithic rather than rubble piles; a cohesive body is required to avoid rotational breakup.`
  if (p < 6)   return `Period of ${p.toFixed(3)} hr is typical for main-belt asteroids. The double-hump lightcurve is consistent with an elongated triaxial ellipsoid shape.`
  return `Long rotation period of ${p.toFixed(3)} hr. Slow rotators may have experienced YORP braking over their lifetime, or represent a primordial slow-spin population.`
}

const FILTERS = [
  { id: 'all',  label: 'All'           },
  { id: 'conf', label: 'Confirmed'     },
  { id: 'tent', label: 'Tentative'     },
  { id: 'fup',  label: 'Follow-up'     },
  { id: 'r3',   label: 'R=3 only'      },
  { id: 'fast', label: 'Fast (<2 hr)'  },
]

export default function App() {
  const [catalog,  setCatalog]  = useState([])
  const [page, setPage] = useState('explore')
  const [selected, setSelected] = useState(null)
  const [filter,   setFilter]   = useState('all')
  const [minP,     setMinP]     = useState(0)
  const [maxP,     setMaxP]     = useState(24)
  const detailRef = useRef(null)

  useEffect(() => {
    fetch('/data/catalog_92k.csv')
      .then(r => r.text())
      .then(text => {
        const { data } = Papa.parse(text, { header: true, skipEmptyLines: true })
        setCatalog(data)
      })
      .catch(err => console.error('Failed to load catalog:', err))
  }, [])

  const handleSelect = row => {
    setSelected(prev => prev?.provid === row.provid ? null : row)
    setTimeout(() => detailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 40)
  }

  const filtered = catalog.filter(row => {
    const p = parseFloat(row.final_period_hr)
    if (!isNaN(p) && (p < minP || p > maxP)) return false
    if (filter === 'conf') return row.reliability === 'confirmed'
    if (filter === 'tent') return row.reliability?.includes('tentative')
    if (filter === 'fup')  return row.reliability === 'followup_needed'
    if (filter === 'r3')   return row.r_code === '3'
    if (filter === 'fast') return !isNaN(p) && p < 2.0
    return true
  })

  const published = catalog.filter(r => r.final_period_hr && r.final_period_hr !== '')
  const confirmed = catalog.filter(r => r.reliability === 'confirmed')

  return (
    <>
      <style>{css}</style>
      <div className="root">

        <header className="header">
          <div>
            <div style={{display:'flex',alignItems:'baseline',gap:12,marginBottom:6}}>
              <h1 className="logo" onClick={()=>{setPage("explore");setSelected(null);}} style={{cursor:"pointer"}}>ATL<span>AST</span></h1>
            </div>
            <p className="header-tagline">
              Asteroid Temporal Lightcurve Analysis &amp; Spin Tracking
            </p>
            <p style={{fontSize:'0.72rem', color:T.textDim, marginTop:3, textAlign:'left'}}>
              Rubin/LSST · First Look commissioning · Apr–May 2025
            </p>
          </div>
          <div style={{display:'flex',flexDirection:'column',gap:10,alignItems:'flex-end'}}>
            <div className="stats-grid">
              <div className="stat-cell">
                <span className="stat-n">{catalog.length}</span>
                <span className="stat-l">Observed</span>
              </div>
              <div className="stat-cell">
                <span className="stat-n">{published.length}</span>
                <span className="stat-l">Published</span>
              </div>
              <div className="stat-cell">
                <span className="stat-n">{confirmed.length}</span>
                <span className="stat-l">Confirmed</span>
              </div>
            </div>
            <div style={{display:'flex',gap:4,background:'#f1f5f9',
              borderRadius:8,padding:3,border:`1px solid ${T.border}`}}>
              {[['explore','Population'],['catalog','Catalog'],['methods','Methodology'],['about','About']].map(([id,label])=>(
                <button key={id}
                  onClick={()=>{setPage(id);setSelected(null);}}
                  style={{fontFamily:"Inter,sans-serif",fontSize:"0.72rem",
                    fontWeight:500,padding:"5px 16px",border:"none",
                    borderRadius:6,cursor:"pointer",transition:"all 0.12s",
                    background: page===id ? T.accent : "transparent",
                    color: page===id ? "#fff" : T.textSec}}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        </header>

        <div ref={detailRef} />

        {selected && (
          <div className="detail">
{/* ── Header row ── */}
            <div className="detail-head">
              <div>
                <div className="detail-id">{selected.provid}</div>
                <div className="detail-period-hero">
                  {selected.final_period_hr
                    ? <>Period: <strong>{parseFloat(selected.final_period_hr).toFixed(4)}</strong> hr</>
                    : 'No period detected'}
                </div>
              </div>
              <button className="btn-close" onClick={() => setSelected(null)}>✕ close</button>
            </div>

            {/* ── Status cluster: R-code · reliability · regime ── */}
            <div style={{display:'flex', gap:'8px', alignItems:'center', marginBottom:'12px', flexWrap:'wrap'}}>
              <RBadge code={selected.r_code} />
              {selected.reliability && (
                <span style={{fontSize:'0.72rem', fontWeight:500, color: selected.reliability==='confirmed'?T.green:selected.reliability==='followup_needed'?T.red:T.amber,
                  background: selected.reliability==='confirmed'?T.greenSoft:selected.reliability==='followup_needed'?T.redSoft:T.amberSoft,
                  padding:'2px 8px', borderRadius:4}}>
                  {selected.reliability.replace('_',' ')}
                </span>
              )}
              {selected.regime && (
                <span style={{fontSize:'0.72rem', color:T.textDim, background:T.bg, border:`1px solid ${T.border}`, padding:'2px 8px', borderRadius:4}}>
                  {selected.regime}
                </span>
              )}
              {/* Spin barrier warning */}
              {selected.final_period_hr && parseFloat(selected.final_period_hr) < 2.2 && (
                <span style={{fontSize:'0.72rem', fontWeight:600, color:'#dc2626', background:'#fee2e2',
                  border:'1px solid #fca5a5', padding:'2px 10px', borderRadius:4, marginLeft:'auto'}}>
                  ⚡ Superfast rotator — P &lt; 2.2 hr spin barrier
                </span>
              )}
              {selected.final_period_hr && parseFloat(selected.final_period_hr) < 0.5 && (
                <span style={{fontSize:'0.72rem', fontWeight:600, color:'#7c3aed', background:'#ede9fe',
                  border:'1px solid #c4b5fd', padding:'2px 10px', borderRadius:4}}>
                  ⚡ Ultra-fast (cf. Greenstreet et al. 2026)
                </span>
              )}
            </div>

            {/* ── Detection cluster ── */}
            <div className="detail-fields" style={{gridTemplateColumns:'repeat(auto-fill, minmax(130px,1fr))'}}>
              {[
                ['Period (hr)',   selected.final_period_hr ? parseFloat(selected.final_period_hr).toFixed(4) : '—'],
                ['Amplitude',    selected.t2_amplitude_mag ? `${parseFloat(selected.t2_amplitude_mag).toFixed(3)} mag` : '—'],
                ['SNR',          selected.t1_snr ? parseFloat(selected.t1_snr).toFixed(1) : '—'],
              ].map(([l,v]) => (
                <div className="df" key={l}><span className="df-l">{l}</span><span className="df-v">{v}</span></div>
              ))}
            </div>

            {/* ── Observation cluster ── */}
            <div style={{display:'flex', gap:'6px', flexWrap:'wrap', margin:'10px 0', alignItems:'center'}}>
              <span style={{fontSize:'0.62rem', textTransform:'uppercase', letterSpacing:'0.08em', color:T.textDim, fontWeight:600, marginRight:4}}>Observations</span>
              {[
                [selected.t1_n_obs, 'obs'],
                [selected.n_nights, 'nights'],
                [selected.baseline_hr ? `${(parseFloat(selected.baseline_hr)/24).toFixed(1)}d` : null, 'baseline'],
                [selected.alias_risk, 'alias risk'],
              ].filter(([v])=>v).map(([v,l])=>(
                <span key={l} style={{fontSize:'0.75rem', fontFamily:"'JetBrains Mono',monospace", color:T.textPri,
                  background:T.bg, border:`1px solid ${T.border}`, borderRadius:4, padding:'2px 8px'}}>
                  <strong>{v}</strong> <span style={{color:T.textDim,fontSize:'0.68rem'}}>{l}</span>
                </span>
              ))}
            </div>

            {/* ── LCDB comparison ── */}
            {selected.lcdb_period_hr && (
              <div style={{fontSize:'0.75rem', padding:'6px 10px', background:'#f0fdf4', border:'1px solid #bbf7d0',
                borderRadius:5, marginBottom:'10px', display:'flex', gap:12, flexWrap:'wrap', alignItems:'center'}}>
                <span style={{fontWeight:600, color:T.green}}>LCDB match</span>
                <span style={{fontFamily:"'JetBrains Mono',monospace"}}>Published: {parseFloat(selected.lcdb_period_hr).toFixed(4)} hr</span>
                {selected.lcdb_delta_pct && (
                  <span style={{color: Math.abs(parseFloat(selected.lcdb_delta_pct))<10?T.green:T.red}}>
                    Δ = {parseFloat(selected.lcdb_delta_pct).toFixed(1)}%
                  </span>
                )}
                {selected.lcdb_u_code && <span style={{color:T.textDim}}>U={selected.lcdb_u_code}</span>}
              </div>
            )}

            {/* ── Method agreement chips ── */}
            {(selected.t2_mhaov_period_hr || selected.t2_mbls_period_hr || selected.t2_ce_period_hr) && (
              <div style={{display:'flex', gap:6, flexWrap:'wrap', marginBottom:'10px', alignItems:'center'}}>
                <span style={{fontSize:'0.62rem', textTransform:'uppercase', letterSpacing:'0.08em', color:T.textDim, fontWeight:600, marginRight:4}}>Methods</span>
                {[
                  ['MHAOV', selected.t2_mhaov_period_hr],
                  ['MBLS',  selected.t2_mbls_period_hr],
                  ['CE',    selected.t2_ce_period_hr],
                ].filter(([,v])=>v).map(([name,val])=>(
                  <span key={name} style={{fontSize:'0.7rem', fontFamily:"'JetBrains Mono',monospace",
                    background:T.accentSoft, color:T.accent, border:`1px solid ${T.border}`,
                    borderRadius:4, padding:'2px 8px'}}>
                    {name} {parseFloat(val).toFixed(3)}h
                  </span>
                ))}
              </div>
            )}

            {/* reliability_notes — shown only if present and not generic */}
            {selected.reliability_notes && selected.reliability_notes !== 'nan' && !selected.reliability_notes.includes('High confidence') && (
              <div style={{fontSize:'0.72rem', color:T.textSec, padding:'5px 10px',
                background:T.bg, border:`1px solid ${T.border}`, borderRadius:4, lineHeight:1.5}}>
                📋 {selected.reliability_notes}
              </div>
            )}

            <div style={{marginTop:'16px'}}><AsteroidCharts provid={selected.provid} /></div>
          </div>
        )}

        {page === "catalog" && <div className="legend">
          <span className="legend-lbl">R-code:</span>
          {[['3','High confidence'],['2','Moderate confidence'],['1','Tentative'],['0','No period'],[-1,'Alias suspected']].map(([r, desc]) => (
            <span key={r} title={desc} style={{ cursor: 'help' }}><RBadge code={r} /></span>
          ))}
          <span style={{ fontSize: '0.65rem', color: T.textDim, marginLeft: 4 }}>hover for meaning</span>
        </div>}

        {page === "catalog" && <div className="controls">
          <div className="filter-grp">
            {FILTERS.map(f => (
              <button key={f.id} className={`fbtn ${filter === f.id ? 'on' : ''}`} onClick={() => setFilter(f.id)}>
                {f.label}
              </button>
            ))}
          </div>

          <div className="slider-wrap">
            <span className="slider-lbl">Period</span>
            <input type="range" min={0} max={24} step={0.5} value={minP}
              onChange={e => setMinP(Math.min(parseFloat(e.target.value), maxP - 0.5))} />
            <input type="range" min={0} max={24} step={0.5} value={maxP}
              onChange={e => setMaxP(Math.max(parseFloat(e.target.value), minP + 0.5))} />
            <span className="slider-val">{minP}–{maxP} hr</span>
          </div>

          <span className="count-label">{filtered.length} / {catalog.length}</span>
        </div>}

        {page === 'explore' && (
          <Explore catalog={catalog} onSelect={(row)=>{
            setSelected(row);
            setPage('catalog');
            setTimeout(()=>detailRef.current?.scrollIntoView({behavior:'smooth',block:'start'}),40);
          }}/>
        )}

        {page === 'methods' && (
          <div style={{padding:'64px 0',textAlign:'center'}}>
            <div style={{fontSize:'2rem',marginBottom:16,opacity:0.15}}>⚗</div>
            <div style={{fontSize:'0.9rem',fontWeight:600,color:T.textSec,marginBottom:8}}>Methodology</div>
            <div style={{fontSize:'0.75rem',color:T.textDim,maxWidth:480,margin:'0 auto',lineHeight:1.8}}>
              Pipeline documentation — 3-tier detection system (GLS, MBLS, MHAOV, CE),
              R-code reliability framework, and validation results.
              <br/><span style={{color:T.accent}}>Coming in next release.</span>
            </div>
          </div>
        )}

        {page === 'about' && (
          <div style={{padding:'64px 0',textAlign:'center'}}>
            <div style={{fontSize:'2rem',marginBottom:16,opacity:0.15}}>◎</div>
            <div style={{fontSize:'0.9rem',fontWeight:600,color:T.textSec,marginBottom:8}}>About ATLAST</div>
            <div style={{fontSize:'0.75rem',color:T.textDim,maxWidth:480,margin:'0 auto',lineHeight:1.8}}>
              ATLAST is an open-source asteroid rotation period catalog built on
              Rubin/LSST commissioning photometry, using a three-tier multi-method
              detection pipeline validated against the LCDB.
              <br/><br/>
              <a href="https://github.com/wonrobot/asteroid-pipeline" target="_blank"
                rel="noreferrer" style={{color:T.accent,textDecoration:'none'}}>
                GitHub ↗
              </a>
              {' · '}Greenstreet et al. 2026
            </div>
          </div>
        )}

        {page === 'catalog' && <div className="tbl-wrap">
          <table>
            <thead>
              <tr>
                <th>Designation</th>
                <th>Period (hr)</th>
                <th>R-code</th>
                <th>Reliability</th>
                <th>Regime</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr><td colSpan={5} style={{ padding: 48, textAlign: 'center', color: T.textDim, fontSize: '0.85rem' }}>
                  No objects match current filters.
                </td></tr>
              ) : filtered.map((row, i) => (
                <tr key={i} className={selected?.provid === row.provid ? 'sel' : ''} onClick={() => handleSelect(row)}>
                  <td className="td-id">{row.provid}</td>
                  <td>
                    {row.final_period_hr
                      ? <span className="td-p">{parseFloat(row.final_period_hr).toFixed(3)}</span>
                      : <span className="td-null">—</span>}
                  </td>
                  <td><RBadge code={row.r_code} /></td>
                  <td><span className="rc">{row.reliability || '—'}</span></td>
                  <td className="td-sec">{row.regime || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>}

        <footer className="footer">
          <span>ATLAST · Rubin/LSST commissioning data · Apr–May 2025 · 3-tier pipeline</span>
          <span>
            <a href="https://github.com/wonrobot/asteroid-pipeline" target="_blank" rel="noreferrer">GitHub</a>
            {' · '}Rubin Observatory First Look
          </span>
        </footer>

      </div>
    </>
  )
}
