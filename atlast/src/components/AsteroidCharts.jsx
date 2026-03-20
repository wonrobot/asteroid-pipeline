import { useEffect, useState } from "react";
import Plot from "react-plotly.js";

export default function AsteroidCharts({ provid }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!provid) return;
    const slug = provid.replace(/\s+/g, "_");
    fetch(`/data/plots/data_${slug}.json`)
      .then(r => {
        if (!r.ok) throw new Error("No data for " + provid);
        return r.json();
      })
      .then(setData)
      .catch(e => setError(e.message));
  }, [provid]);

  if (!provid) return null;
  if (error) return <p style={{color:"#888"}}>No chart data available for {provid}.</p>;
  if (!data) return <p style={{color:"#888"}}>Loading charts…</p>;

  const plotStyle = { width: "100%", height: 280 };
  const config = { displayModeBar: false, responsive: true };
  const baseLayout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "Inter, sans-serif", color: "#cdd6f4", size: 12 },
    margin: { t: 36, b: 48, l: 56, r: 16 },
    xaxis: { gridcolor: "#313244", zerolinecolor: "#45475a" },
    yaxis: { gridcolor: "#313244", zerolinecolor: "#45475a" },
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem", marginTop: "1rem" }}>

      {/* Lightcurve */}
      <Plot
        style={plotStyle}
        config={config}
        data={[{
          x: data.obs.mjd,
          y: data.obs.mag,
          error_y: { type: "data", array: data.obs.magerr, visible: true, color: "#585b70" },
          mode: "markers",
          marker: { color: data.obs.band.map(b => b==="g"?"#a6e3a1":b==="r"?"#f38ba8":"#89b4fa"), size: 4 },
          type: "scatter",
          name: "Observations",
        }]}
        layout={{
          ...baseLayout,
          title: { text: "Lightcurve", font: { size: 13, color: "#cdd6f4" } },
          yaxis: { ...baseLayout.yaxis, autorange: "reversed", title: { text: "mag" } },
          xaxis: { ...baseLayout.xaxis, title: { text: "MJD" } },
        }}
      />

      {/* Periodogram */}
      <Plot
        style={plotStyle}
        config={config}
        data={[{
          x: data.pgram.periods,
          y: data.pgram.power,
          mode: "lines",
          line: { color: "#89b4fa", width: 1.2 },
          type: "scatter",
          name: "MBLS Power",
        },
        data.best_period ? {
          x: [data.best_period, data.best_period],
          y: [0, Math.max(...data.pgram.power)],
          mode: "lines",
          line: { color: "#f38ba8", width: 1.5, dash: "dash" },
          name: `P = ${data.best_period.toFixed(3)} hr`,
        } : null].filter(Boolean)}
        layout={{
          ...baseLayout,
          title: { text: "Periodogram (MBLS)", font: { size: 13, color: "#cdd6f4" } },
          xaxis: { ...baseLayout.xaxis, title: { text: "Period (hr)" }, type: "log" },
          yaxis: { ...baseLayout.yaxis, title: { text: "Power" } },
        }}
      />

      {/* Phase fold */}
      {data.fold && (
        <Plot
          style={plotStyle}
          config={config}
          data={[{
            x: data.fold.phase,
            y: data.fold.mag,
            mode: "markers",
            marker: { color: data.fold.band.map(b => b==="g"?"#a6e3a1":b==="r"?"#f38ba8":"#89b4fa"), size: 4 },
            type: "scatter",
            name: "Folded",
          }]}
          layout={{
            ...baseLayout,
            title: { text: `Phase Fold  (P = ${data.best_period?.toFixed(3)} hr)`, font: { size: 13, color: "#cdd6f4" } },
            yaxis: { ...baseLayout.yaxis, autorange: "reversed", title: { text: "mag" } },
            xaxis: { ...baseLayout.xaxis, title: { text: "Phase" }, range: [0, 1] },
          }}
        />
      )}
    </div>
  );
}
