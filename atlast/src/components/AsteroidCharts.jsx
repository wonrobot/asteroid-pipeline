import { useEffect, useState, useRef } from "react";
import Plotly from "plotly.js-dist";

const BAND_COLOR = { g: "#4ade80", r: "#f87171", i: "#60a5fa" };

function PlotDiv({ id, data, layout }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    Plotly.react(ref.current, data, {
      paper_bgcolor: "transparent",
      plot_bgcolor:  "#f7f8fc",
      font: { family: "Inter, sans-serif", color: "#475569", size: 11 },
      margin: { t: 36, b: 48, l: 56, r: 16 },
      xaxis: { gridcolor: "#dde3f0", zerolinecolor: "#b8c4e0", ...layout.xaxis },
      yaxis: { gridcolor: "#dde3f0", zerolinecolor: "#b8c4e0", ...layout.yaxis },
      title: { font: { size: 12, color: "#0f172a" }, ...layout.title },
      legend: { font: { size: 10 } },
      showlegend: true,
    }, { displayModeBar: false, responsive: true });
  }, [data, layout]);
  return <div ref={ref} style={{ width: "100%", height: 260 }} />;
}

export default function AsteroidCharts({ provid }) {
  const [data, setData]   = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!provid) return;
    setData(null); setError(null);
    const slug = provid.replace(/\s+/g, "_");
    fetch(`/data/plots/data_${slug}.json`)
      .then(r => { if (!r.ok) throw new Error("missing"); return r.json(); })
      .then(setData)
      .catch(() => setError(true));
  }, [provid]);

  if (!provid) return null;
  if (error)   return <p style={{ color: "#94a3b8", fontSize: "0.78rem", marginTop: 12 }}>No chart data available.</p>;
  if (!data)   return <p style={{ color: "#94a3b8", fontSize: "0.78rem", marginTop: 12 }}>Loading charts…</p>;

  // Lightcurve traces — one per band
  const bands = [...new Set(data.obs.band)];
  const lcTraces = bands.map(b => ({
    x: data.obs.mjd.filter((_, i) => data.obs.band[i] === b),
    y: data.obs.mag.filter((_, i) => data.obs.band[i] === b),
    error_y: {
      type: "data",
      array: data.obs.magerr.filter((_, i) => data.obs.band[i] === b),
      visible: true, color: BAND_COLOR[b] + "88",
    },
    mode: "markers",
    marker: { color: BAND_COLOR[b], size: 3.5 },
    name: b + "-band",
    type: "scatter",
  }));

  // Periodogram
  const pgramTraces = [{
    x: data.pgram.periods,
    y: data.pgram.power,
    mode: "lines",
    line: { color: "#2563eb", width: 1 },
    name: "MBLS",
    type: "scatter",
  }];
  if (data.best_period) {
    const maxPow = Math.max(...data.pgram.power);
    pgramTraces.push({
      x: [data.best_period, data.best_period],
      y: [0, maxPow],
      mode: "lines",
      line: { color: "#dc2626", width: 1.5, dash: "dash" },
      name: `P = ${data.best_period.toFixed(3)} hr`,
      type: "scatter",
    });
  }

  // Phase fold
  const foldTraces = data.fold ? bands.map(b => ({
    x: data.fold.phase.filter((_, i) => data.fold.band[i] === b),
    y: data.fold.mag.filter((_, i) => data.fold.band[i] === b),
    mode: "markers",
    marker: { color: BAND_COLOR[b], size: 3.5 },
    name: b + "-band",
    type: "scatter",
  })) : null;

  if (data.fold) {
    foldTraces.push({
      x: data.fold.fitted_phase,
      y: data.fold.fitted_mag,
      mode: "lines",
      line: { color: "#0f172a", width: 1.5 },
      name: "fit",
      type: "scatter",
    });
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", marginTop: "1rem" }}>
      <PlotDiv id="lc"    data={lcTraces}    layout={{ title: { text: "Lightcurve" }, xaxis: { title: { text: "MJD" } }, yaxis: { title: { text: "mag" }, autorange: "reversed" } }} />
      <PlotDiv id="pgram" data={pgramTraces} layout={{ title: { text: "Periodogram (MBLS)" }, xaxis: { title: { text: "Period (hr)" }, type: "log" }, yaxis: { title: { text: "Power" } } }} />
      {foldTraces && (
        <PlotDiv id="fold" data={foldTraces} layout={{ title: { text: `Phase fold  (P = ${data.best_period?.toFixed(3)} hr)` }, xaxis: { title: { text: "Phase" }, range: [0,1] }, yaxis: { title: { text: "Δmag (detrended)" }, autorange: "reversed" } }} />
      )}
    </div>
  );
}
