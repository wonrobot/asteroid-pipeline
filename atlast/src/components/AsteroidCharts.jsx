import { useEffect, useState } from 'react'
import Plot from 'plotly.js-dist'

export default function AsteroidCharts({ provid }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const slug = provid.split(' ').join('_')

  useEffect(() => {
    setLoading(true)
    setData(null)
    fetch(`/data/plots/data_${slug}.json`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [provid])

  useEffect(() => {
    if (!data) return

    const layout_base = {
      paper_bgcolor: '#fff',
      plot_bgcolor:  '#fafafa',
      margin: { t: 36, r: 16, b: 48, l: 56 },
      height: 240,
      font: { family: 'Inter, sans-serif', size: 11 },
      legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
    }
    const config = { displayModeBar: false, responsive: true }
    const grid = { gridcolor: '#eee', zerolinecolor: '#dde3f0' }

    // Lightcurve
    if (data.obs) {
      const obs = data.obs
      const bandColors = { Lr: '#e85d5d', Lg: '#16a34a', Li: '#d97706' }
      const bands = [...new Set(obs.bands)]
      const traces = bands.map(b => {
        const idx = obs.bands.reduce((a, x, i) => (x === b ? [...a, i] : a), [])
        return {
          x: idx.map(i => obs.t_hrs[i]),
          y: idx.map(i => obs.y_dt[i]),
          mode: 'markers', name: b,
          marker: { color: bandColors[b] || '#aaa', size: 3.5, opacity: 0.75 }
        }
      })
      Plot.newPlot('chart-lc', traces, {
        ...layout_base,
        title: { text: 'Lightcurve (detrended)', font: { size: 12 } },
        xaxis: { title: 'Time (hr)', ...grid },
        yaxis: { title: 'Δmag', autorange: 'reversed', ...grid },
      }, config)
    }

    // Periodogram
    if (data.pgram) {
      const pg = data.pgram
      const traces = []
      if (pg.test_periods && pg.gls_power)
        traces.push({ x: pg.test_periods, y: pg.gls_power, name: 'GLS',
          line: { color: '#2563eb', width: 1.2 }, mode: 'lines' })
      if (pg.test_periods && pg.mhaov_power)
        traces.push({ x: pg.test_periods, y: pg.mhaov_power, name: 'MHAOV',
          line: { color: '#16a34a', width: 1.2 }, mode: 'lines' })
      if (pg.best_period)
        traces.push({ x: [pg.best_period, pg.best_period], y: [0, 1],
          name: 'Adopted', line: { color: '#d97706', width: 2, dash: 'dash' }, mode: 'lines' })
      Plot.newPlot('chart-pgram', traces, {
        ...layout_base,
        title: { text: 'Periodogram', font: { size: 12 } },
        xaxis: { title: 'Period (hr)', ...grid },
        yaxis: { title: 'Power', ...grid },
      }, config)
    }

    // Phase fold
    if (data.fold) {
      const f = data.fold
      const traces = []
      if (f.phase && f.mag)
        traces.push({ x: f.phase, y: f.mag, mode: 'markers', name: 'Data',
          marker: { color: '#2563eb', size: 3, opacity: 0.65 } })
      if (f.phase_fit && f.mag_fit)
        traces.push({ x: f.phase_fit, y: f.mag_fit, mode: 'lines', name: 'Model',
          line: { color: '#d97706', width: 2 } })
      Plot.newPlot('chart-fold', traces, {
        ...layout_base,
        title: { text: 'Phase fold', font: { size: 12 } },
        xaxis: { title: 'Phase', range: [0, 1], ...grid },
        yaxis: { title: 'Δmag', autorange: 'reversed', ...grid },
      }, config)
    }

  }, [data])

  const boxStyle = {
    background: '#fff',
    borderRadius: 6,
    border: '1px solid #dde3f0',
    overflow: 'hidden',
    minHeight: 240,
  }

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8',
      fontFamily: 'JetBrains Mono, monospace', fontSize: '0.78rem' }}>
      Loading data for {provid}...
    </div>
  )

  if (!data) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8',
      fontFamily: 'JetBrains Mono, monospace', fontSize: '0.78rem' }}>
      No precomputed data for {provid}.<br/>
      <span style={{ fontSize: '0.7rem' }}>Run the pipeline to generate plot files.</span>
    </div>
  )

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 14 }}>
      <div id="chart-lc"    style={boxStyle} />
      <div id="chart-pgram" style={boxStyle} />
      <div id="chart-fold"  style={{ ...boxStyle, gridColumn: '1 / -1' }} />
    </div>
  )
}
