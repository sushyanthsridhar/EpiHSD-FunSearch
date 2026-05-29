"""
visualize_latent_animation.py
────────────────────────────────────────────────────────────────
Creates an interactive animated 3D plot of 10 DR provinces moving
through latent space year by year (2015-2023).

Opens in any browser. Auto-plays. You can rotate while it runs.
Each province is a unique color with a trajectory line.

Reads:   latent_all_years_latent3.csv  (from train_autoencoder_eval_latent3.py)
Saves:   latent_animation.html

Install once:
  pip install plotly

Run:
  python3 visualize_latent_animation.py
────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Run:  pip install plotly  then try again.")

HERE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "latent_all_years_latent3.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        "latent_all_years.csv not found.\n"
        "Run train_autoencoder_eval_latent3.py first."
    )

# ── 10 provinces spanning the full latent space ────────────────
# Chosen to cover urban, coastal, interior, dry, wet, large, small
SELECTED = [
    "32 Santo Domingo",          # largest city, extreme +z1
    "25 Santiago",               # second city
    "01 Distrito Nacional",      # capital core
    "11 La Altagracia",          # east coast tourism, extreme -z1
    "15 Monte Cristi",           # remote northwest, extreme -z1
    "16 Pedernales",             # driest desert, extreme -z3
    "14 María Trinidad Sánchez", # Cibao valley, extreme +z3
    "06 Duarte",                 # interior agricultural
    "21 San Cristóbal",          # suburban south
    "04 Barahona",               # southern coast
]

COLORS = [
    "#e63946",   # red            Santo Domingo
    "#2a9d8f",   # teal           Santiago
    "#f4a261",   # orange         Distrito Nacional
    "#457b9d",   # steel blue     La Altagracia
    "#a786c9",   # purple         Monte Cristi
    "#e9c46a",   # gold           Pedernales
    "#06d6a0",   # mint green     María Trinidad Sánchez
    "#ff6b9d",   # pink           Duarte
    "#90e0ef",   # light blue     San Cristóbal
    "#fb8500",   # deep orange    Barahona
]

SHORT_NAMES = [p.split(" ", 1)[1] for p in SELECTED]

# ── Load data ──────────────────────────────────────────────────
df    = pd.read_csv(CSV_PATH)
df    = df[df["province"].isin(SELECTED)].copy()
years = sorted(df["year"].unique())
print(f"Years: {years}")
print(f"Provinces: {df['province'].nunique()} found")

# Build trajectory dict
traj = {}
for prov in SELECTED:
    sub = df[df["province"] == prov].sort_values("year")
    traj[prov] = {
        "years": sub["year"].tolist(),
        "z1":    sub["z1"].tolist(),
        "z2":    sub["z2"].tolist(),
        "z3":    sub["z3"].tolist(),
        "split": sub["split"].tolist(),
    }

# ── Build Plotly frames ────────────────────────────────────────
# Each frame shows trajectories up to and including that year.
frames = []

for frame_idx, year in enumerate(years):

    frame_traces = []

    for prov, color, short in zip(SELECTED, COLORS, SHORT_NAMES):
        t = traj[prov]
        # Only include years up to current frame
        mask   = [y <= year for y in t["years"]]
        z1_his = [v for v, m in zip(t["z1"], mask) if m]
        z2_his = [v for v, m in zip(t["z2"], mask) if m]
        z3_his = [v for v, m in zip(t["z3"], mask) if m]
        yr_his = [v for v, m in zip(t["years"], mask) if m]
        sp_his = [v for v, m in zip(t["split"], mask) if m]

        if not z1_his:
            continue

        # Trajectory line
        frame_traces.append(go.Scatter3d(
            x=z1_his, y=z2_his, z=z3_his,
            mode="lines",
            line=dict(color=color, width=4),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Historical dots (smaller)
        frame_traces.append(go.Scatter3d(
            x=z1_his[:-1], y=z2_his[:-1], z=z3_his[:-1],
            mode="markers",
            marker=dict(size=4, color=color, opacity=0.45),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Current year marker (large, bright)
        hover_txt = (
            f"<b>{short}</b><br>"
            f"Year: {yr_his[-1]}<br>"
            f"Split: {sp_his[-1]}<br>"
            f"z1: {z1_his[-1]:.3f}<br>"
            f"z2: {z2_his[-1]:.3f}<br>"
            f"z3: {z3_his[-1]:.3f}"
        )
        frame_traces.append(go.Scatter3d(
            x=[z1_his[-1]], y=[z2_his[-1]], z=[z3_his[-1]],
            mode="markers+text",
            marker=dict(size=10, color=color, opacity=1.0,
                        line=dict(color="white", width=1.5)),
            text=[short[:6]],
            textposition="top center",
            textfont=dict(size=9, color=color),
            name=short,
            showlegend=(frame_idx == 0),
            hovertemplate=hover_txt + "<extra></extra>",
        ))

    frames.append(go.Frame(data=frame_traces, name=str(year)))

# ── Initial frame (year 0) ─────────────────────────────────────
initial_traces = frames[0].data

# ── Layout ─────────────────────────────────────────────────────
def split_color(year):
    if year <= 2019: return "#4fc3f7"
    if year <= 2021: return "#ffb74d"
    return "#ef5350"

slider_steps = []
for year in years:
    color = split_color(year)
    label = "TRAIN" if year <= 2019 else ("VAL" if year <= 2021 else "TEST")
    slider_steps.append(dict(
        args=[[str(year)],
              {"frame":     {"duration": 600, "redraw": True},
               "mode":      "immediate",
               "transition": {"duration": 400}}],
        label=f"{year} ({label})",
        method="animate",
    ))

layout = go.Layout(
    title=dict(
        text="Dengue Outbreak Shape Space<br>"
             "<sup>10 DR Provinces Drifting Through 3D Latent Space (2015-2023)</sup>",
        font=dict(size=16, color="white"),
        x=0.5,
    ),
    paper_bgcolor="#0f0f1a",
    plot_bgcolor="#0f0f1a",
    scene=dict(
        xaxis=dict(title="z1  (urban scale and connectivity,  pop r=+0.71)", color="#aaaacc",
                   gridcolor="#222244", backgroundcolor="#0f0f1a"),
        yaxis=dict(title="z2  (interior vs coastal,  intra-year climate curve shape)", color="#aaaacc",
                   gridcolor="#222244", backgroundcolor="#0f0f1a"),
        zaxis=dict(title="z3  (epidemic intensity and regional geography,  incidence r=+0.53)", color="#aaaacc",
                   gridcolor="#222244", backgroundcolor="#0f0f1a"),
        bgcolor="#0f0f1a",
    ),
    legend=dict(
        font=dict(color="white", size=10),
        bgcolor="rgba(20,20,40,0.8)",
        bordercolor="#444466",
        borderwidth=1,
        x=0.01, y=0.99,
    ),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        y=0.08,
        x=0.5,
        xanchor="center",
        buttons=[
            dict(label="▶  Play",
                 method="animate",
                 args=[None, {"frame":     {"duration": 700, "redraw": True},
                              "fromcurrent": True,
                              "transition": {"duration": 400},
                              "mode":      "immediate"}]),
            dict(label="⏸  Pause",
                 method="animate",
                 args=[[None], {"frame":     {"duration": 0, "redraw": False},
                                "mode":      "immediate",
                                "transition": {"duration": 0}}]),
        ],
        font=dict(color="white", size=13),
        bgcolor="#1a1a2e",
        bordercolor="#444488",
    )],
    sliders=[dict(
        active=0,
        steps=slider_steps,
        currentvalue=dict(
            prefix="Year: ",
            font=dict(size=14, color="white"),
            visible=True,
            xanchor="center",
        ),
        pad=dict(t=50, b=10),
        len=0.85,
        x=0.075,
        font=dict(color="#aaaacc", size=10),
        bgcolor="#1a1a2e",
        bordercolor="#444488",
        tickcolor="#aaaacc",
    )],
    height=750,
    margin=dict(l=0, r=0, t=80, b=80),
    annotations=[
        dict(text="<b>TRAIN</b> 2015-2019",
             x=0.01, y=0.04, xref="paper", yref="paper",
             font=dict(color="#4fc3f7", size=11), showarrow=False),
        dict(text="<b>VAL</b> 2020-2021",
             x=0.15, y=0.04, xref="paper", yref="paper",
             font=dict(color="#ffb74d", size=11), showarrow=False),
        dict(text="<b>TEST</b> 2022-2023",
             x=0.28, y=0.04, xref="paper", yref="paper",
             font=dict(color="#ef5350", size=11), showarrow=False),
    ],
)

# ── Build and save figure ──────────────────────────────────────
fig = go.Figure(data=initial_traces, layout=layout, frames=frames)

out_path = os.path.join(HERE, "latent_animation.html")
fig.write_html(
    out_path,
    auto_play=True,
    include_plotlyjs=True,
    full_html=True,
)
print(f"Saved {out_path}")
print("Open latent_animation.html in your browser. It will auto-play.")
print("You can rotate the 3D view while it plays.")
print("Use the slider to jump to any year.")
print("Hover over any point to see province name, year, and coordinates.")
