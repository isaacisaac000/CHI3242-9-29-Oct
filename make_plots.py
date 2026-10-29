import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from qhchina import load_fonts

load_fonts()

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

COLLECTION_ORDER = ["呐喊", "彷徨", "故事新编", "沈从文"]
COLORS = {
    "呐喊": "#1f77b4",
    "彷徨": "#d62728",
    "故事新编": "#2ca02c",
    "沈从文": "#9467bd",
}

with open("story_coordinates.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 43
x = [float(r["PC1"]) for r in rows]
y = [float(r["PC2"]) for r in rows]
cols = [r["collection"] for r in rows]

# --- PNG ---
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
centroids = {}
for collection in COLLECTION_ORDER:
    pairs = [(xi, yi) for xi, yi, c in zip(x, y, cols) if c == collection]
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    centroids[collection] = (sum(xs) / len(xs), sum(ys) / len(ys))
    ax.scatter(xs, ys, c=COLORS[collection], label=collection, s=140,
               alpha=0.85, edgecolors="black", linewidths=0.6)
    for r, xi, yi in zip(rows, x, y):
        if r["collection"] == collection:
            name = Path(r["source_file"]).stem
            ax.annotate(name, (xi, yi), fontsize=6, alpha=0.8,
                        xytext=(4, 4), textcoords="offset points")
for collection, (cx, cy) in centroids.items():
    ax.scatter([cx], [cy], marker="X", s=420, c=COLORS[collection],
               edgecolors="black", linewidths=2.0, zorder=5,
               label=f"{collection} centroid")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA of qhchina stylometry vectors (100 features, z-scored)")
ax.legend(title="Collection", loc="best")
ax.grid(True, linestyle=":", alpha=0.4)
fig.tight_layout()
png_path = OUT / "pca_stylometry.png"
fig.savefig(png_path)
plt.close(fig)

# --- Interactive HTML ---
hover = [f"{r['document_id']}<br>{r['author']} · {r['collection']}<br>"
         f"{Path(r['source_file']).name}<br>tokens: {r['token_count']}" for r in rows]
fig = go.Figure()
for collection in COLLECTION_ORDER:
    idx = [i for i, c in enumerate(cols) if c == collection]
    fig.add_trace(go.Scatter(
        x=[x[i] for i in idx], y=[y[i] for i in idx],
        mode="markers", name=collection,
        marker=dict(color=COLORS[collection], size=16, opacity=0.85,
                    line=dict(width=1, color="black")),
        customdata=[hover[i].split("<br>") for i in idx],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}"
                      "<br>%{customdata[3]}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        text=[Path(rows[i]["source_file"]).stem for i in idx],
    ))
    cx = sum(x[i] for i in idx) / len(idx)
    cy = sum(y[i] for i in idx) / len(idx)
    fig.add_trace(go.Scatter(
        x=[cx], y=[cy], mode="markers", name=f"{collection} centroid",
        marker=dict(symbol="cross-thin-open", size=26, color=COLORS[collection],
                    line=dict(width=4)),
        hovertemplate=f"<b>{collection} centroid</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>",
    ))
fig.update_layout(
    title="PCA of qhchina stylometry vectors (100 features, z-scored)",
    xaxis_title="PC1", yaxis_title="PC2",
    legend=dict(title="Collection", itemclick="toggle", itemdoubleclick="toggleothers"),
    hovermode="closest",
    template="plotly_white",
)
html_path = OUT / "pca_stylometry.html"
fig.write_html(html_path, include_plotlyjs=True, full_html=True)

print(f"wrote {png_path} and {html_path}")
