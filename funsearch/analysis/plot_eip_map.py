"""
plot_eip_map.py
────────────────────────────────────────────────────────────────────────────
Map of Dominican Republic provinces coloured by FunSearch island assignment.

  Island A (Coastal, 13 provinces) — EIP thermal-forcing  → BLUE STARS
  Island B (Inland,  19 provinces) — Auto-regressive      → GREEN CIRCLES

Install dependencies (one-time):
────────────────────────────────
    pip install folium cartopy matplotlib numpy pyproj

Note on cartopy: the first run downloads Natural Earth shapefiles (~10 MB)
and caches them permanently in ~/cartopy_data/.  Subsequent runs are offline.

Produces two files in funsearch/analysis/:
    eip_map_interactive.html   — open in Chrome/Firefox; full OSM tile layers,
                                 switchable between Voyager / OSM / Satellite
    eip_map_static.png         — publication figure with coastlines, ocean,
                                 country & province borders (Natural Earth)

Run:
────
    cd funsearch/analysis/
    python plot_eip_map.py
────────────────────────────────────────────────────────────────────────────
"""

import os
import folium
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Province centroids  (lat, lon) ────────────────────────────────────────────

ISLAND_A = {           # Coastal — EIP terms present in evolved equation
    "Santo Domingo"       : (18.485, -69.931),
    "Distrito Nacional"   : (18.479, -69.890),
    "La Vega"             : (19.221, -70.529),
    "Barahona"            : (18.211, -71.101),
    "Duarte"              : (19.199, -70.033),
    "Sánchez Ramírez"     : (19.052, -70.153),
    "La Altagracia"       : (18.616, -68.712),
    "San Juan"            : (18.806, -71.228),
    "Monseñor Nouel"      : (18.922, -70.415),
    "San José de Ocoa"    : (18.544, -70.503),
    "Hato Mayor"          : (18.764, -69.255),
    "Samaná"              : (19.206, -69.336),
    "Baoruco"             : (18.486, -71.418),
}

ISLAND_B = {           # Inland — purely auto-regressive, no EIP
    "Santiago"            : (19.451, -70.697),
    "San Cristóbal"       : (18.418, -70.106),
    "Puerto Plata"        : (19.795, -70.685),
    "Monte Cristi"        : (19.866, -71.648),
    "Monte Plata"         : (18.806, -69.784),
    "Pedernales"          : (17.929, -71.444),
    "Elías Piña"          : (18.875, -71.706),
    "Azua"                : (18.452, -70.735),
    "Dajabón"             : (19.549, -71.707),
    "El Seibo"            : (18.765, -69.038),
    "Espaillat"           : (19.623, -70.275),
    "Independencia"       : (18.407, -71.845),
    "La Romana"           : (18.427, -68.972),
    "Mª Trinidad Sánchez" : (19.374, -69.853),
    "Peravia"             : (18.280, -70.336),
    "Hermanas Mirabal"    : (19.375, -70.307),
    "San Pedro de Macorís": (18.451, -69.301),
    "Santiago Rodríguez"  : (19.481, -71.336),
    "Valverde"            : (19.583, -71.072),
}

EIP_TERMS = [
    "+0.943  EIP × rolling_incidence_4wk",
    "−0.768  EIP × incidence_lag2",
    "+0.022  EIP × week_sin",
]

COL_A = "#1565C0"
COL_B = "#2E7D32"


# ══════════════════════════════════════════════════════════════════════════════
# 1. INTERACTIVE HTML MAP  (folium — opens in browser with full OSM tiles)
# ══════════════════════════════════════════════════════════════════════════════

def make_interactive():
    m = folium.Map(
        location=[18.9, -70.5],
        zoom_start=8,
        tiles=None,           # we add tiles manually so we can label them
    )

    # Good-looking tile layers the user can switch between
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
             '&copy; <a href="https://carto.com/">CARTO</a>',
        name="CARTO Voyager (default)",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Satellite",
        control=True,
    ).add_to(m)

    # ── Island A — blue pins ───────────────────────────────────────────────
    fg_a = folium.FeatureGroup(name="★  Island A — EIP-driven (13 coastal)")
    for name, (lat, lon) in ISLAND_A.items():
        popup_html = f"""
        <div style='font-family:Arial,sans-serif; font-size:13px; width:270px'>
          <div style='background:#1565C0; color:white; padding:8px 12px;
                      border-radius:6px 6px 0 0; margin:-1px -1px 8px -1px'>
            <b>★ {name}</b>
          </div>
          <b>Island A — Coastal / EIP-driven</b><br>
          <hr style='margin:6px 0; border-color:#ddd'>
          <b>EIP terms in evolved equation:</b><br>
          {'<br>'.join(f"&nbsp;&nbsp;&nbsp;<code>{t}</code>" for t in EIP_TERMS)}
          <hr style='margin:6px 0; border-color:#ddd'>
          <span style='font-size:11px; color:#555'>
            EIP = exp(0.347·T − 9.17) &nbsp;(Brady et al. 2013)<br>
            Higher temp → shorter incubation → faster transmission
          </span>
        </div>"""

        # Custom blue star icon
        icon = folium.DivIcon(
            html=f"""<div style='
                font-size:22px; color:{COL_A};
                text-shadow: 0 0 3px white, 0 0 5px white;
                line-height:1'>★</div>""",
            icon_size=(28, 28),
            icon_anchor=(14, 14),
        )
        folium.Marker(
            location=[lat, lon],
            icon=icon,
            popup=folium.Popup(popup_html, max_width=290),
            tooltip=f"★ {name}  [EIP-driven]",
        ).add_to(fg_a)
    fg_a.add_to(m)

    # ── Island B — green circles ───────────────────────────────────────────
    fg_b = folium.FeatureGroup(name="●  Island B — Auto-regressive (19 inland)")
    for name, (lat, lon) in ISLAND_B.items():
        popup_html = f"""
        <div style='font-family:Arial,sans-serif; font-size:13px; width:270px'>
          <div style='background:#2E7D32; color:white; padding:8px 12px;
                      border-radius:6px 6px 0 0; margin:-1px -1px 8px -1px'>
            <b>● {name}</b>
          </div>
          <b>Island B — Inland / Auto-regressive</b><br>
          <hr style='margin:6px 0; border-color:#ddd'>
          <b>Evolved equation terms:</b><br>
          &nbsp;&nbsp;&nbsp;<code>−1.412 &nbsp;incidence_lag1</code><br>
          &nbsp;&nbsp;&nbsp;<code>+1.364 &nbsp;rolling_incidence_4wk</code><br>
          &nbsp;&nbsp;&nbsp;<code>−0.605 &nbsp;incidence_lag2</code><br>
          &nbsp;&nbsp;&nbsp;<code>−0.068 &nbsp;incidence_lag4</code><br>
          &nbsp;&nbsp;&nbsp;<code>+0.027 &nbsp;temp_lag4 × rolling_incidence</code><br>
          <hr style='margin:6px 0; border-color:#ddd'>
          <span style='font-size:11px; color:#555'>
            No EIP — dynamics driven by own epidemic history.<br>
            Weak thermal interaction at 4-week lag only.
          </span>
        </div>"""

        icon = folium.DivIcon(
            html=f"""<div style='
                width:16px; height:16px; border-radius:50%;
                background:{COL_B}; border:2.5px solid white;
                box-shadow:0 0 4px rgba(0,0,0,0.4)'></div>""",
            icon_size=(16, 16),
            icon_anchor=(8, 8),
        )
        folium.Marker(
            location=[lat, lon],
            icon=icon,
            popup=folium.Popup(popup_html, max_width=290),
            tooltip=f"● {name}  [Auto-regressive]",
        ).add_to(fg_b)
    fg_b.add_to(m)

    # ── Layer control ──────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # ── Title overlay ──────────────────────────────────────────────────────
    title_html = """
    <div style='position:fixed; top:14px; left:50%; transform:translateX(-50%);
                background:rgba(255,255,255,0.95); padding:10px 22px 8px;
                border-radius:8px; border:1px solid #ccc;
                box-shadow:2px 2px 8px rgba(0,0,0,0.18);
                font-family:Arial,sans-serif; z-index:9999; text-align:center;
                max-width:520px'>
      <div style='font-size:15px; font-weight:bold; color:#222; margin-bottom:3px'>
        FunSearch-SINDy &mdash; DR Province Island Assignment
      </div>
      <div style='font-size:12px; color:#555'>
        <span style='color:#1565C0; font-weight:bold'>★ Island A (13)</span>
        &nbsp; EIP thermal-forcing
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span style='color:#2E7D32; font-weight:bold'>● Island B (19)</span>
        &nbsp; Auto-regressive
        &nbsp;&nbsp;&mdash;&nbsp;&nbsp; click any pin for equation
      </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    out = os.path.join(HERE, "eip_map_interactive.html")
    m.save(out)
    print(f"Interactive  →  {out}")
    print("             Open this file in any browser (Chrome / Firefox / Safari).")


# ══════════════════════════════════════════════════════════════════════════════
# 2. STATIC PNG  (cartopy — Natural Earth coastlines, ocean, borders)
# ══════════════════════════════════════════════════════════════════════════════

def make_static():
    fig = plt.figure(figsize=(14, 9), facecolor="white")

    proj = ccrs.PlateCarree()
    ax   = fig.add_subplot(1, 1, 1, projection=proj)

    # ── Geographic extent — DR + a little buffer ───────────────────────────
    ax.set_extent([-72.2, -68.1, 17.4, 20.2], crs=proj)

    # ── Natural Earth features — ocean, land, borders, coastlines ─────────
    ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                   facecolor="#AED6F1", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"),
                   facecolor="#F0EAD6", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                   linewidth=0.7, edgecolor="#555555", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                   linewidth=1.2, edgecolor="#888888",
                   linestyle="--", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("10m"),
                   linewidth=0.4, edgecolor="#AAAAAA",
                   linestyle=":", zorder=2)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"),
                   linewidth=0.5, edgecolor="#7FB3D3",
                   facecolor="none", zorder=2)
    ax.add_feature(cfeature.LAKES.with_scale("10m"),
                   facecolor="#AED6F1", edgecolor="#7FB3D3",
                   linewidth=0.4, zorder=2)

    # ── Gridlines ─────────────────────────────────────────────────────────
    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color="#CCCCCC", alpha=0.7, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": "#666666"}
    gl.ylabel_style = {"size": 8, "color": "#666666"}

    # ── Plot Island B first (lower z-order so A sits on top) ──────────────
    for name, (lat, lon) in ISLAND_B.items():
        ax.plot(lon, lat, transform=proj,
                marker="o", markersize=9, color=COL_B,
                markeredgecolor="white", markeredgewidth=1.5,
                zorder=5)
        txt = ax.text(lon, lat + 0.09, name,
                      transform=proj, fontsize=6.2, ha="center", va="bottom",
                      color=COL_B, fontweight="bold", zorder=6)
        txt.set_path_effects(
            [pe.withStroke(linewidth=2.2, foreground="white")])

    # ── Plot Island A (blue stars) ─────────────────────────────────────────
    for name, (lat, lon) in ISLAND_A.items():
        ax.plot(lon, lat, transform=proj,
                marker="*", markersize=16, color=COL_A,
                markeredgecolor="white", markeredgewidth=1.2,
                zorder=7)
        txt = ax.text(lon, lat + 0.09, name,
                      transform=proj, fontsize=6.2, ha="center", va="bottom",
                      color=COL_A, fontweight="bold", zorder=8)
        txt.set_path_effects(
            [pe.withStroke(linewidth=2.5, foreground="white")])

    # ── Legend ─────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=COL_A,
               markersize=14, markeredgecolor="white",
               label=f"Island A — EIP-driven  (n=13 coastal provinces)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_B,
               markersize=9, markeredgecolor="white",
               label=f"Island B — Auto-regressive  (n=19 inland provinces)"),
    ]
    leg = ax.legend(handles=legend_elements, loc="lower right",
                    framealpha=0.93, fontsize=9,
                    title="FunSearch Island Assignment",
                    title_fontsize=9.5, edgecolor="#CCCCCC")

    # ── EIP terms annotation ───────────────────────────────────────────────
    eip_str = "EIP terms in Island A:\n" + "\n".join(f"  {t}" for t in EIP_TERMS)
    ax.text(0.01, 0.03, eip_str,
            transform=ax.transAxes, fontsize=7.5,
            color=COL_A, va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="#E3F2FD",
                      ec=COL_A, alpha=0.92))

    # ── Title ─────────────────────────────────────────────────────────────
    ax.set_title(
        "FunSearch-SINDy — Province Island Assignment\n"
        "Dominican Republic Dengue · EIP thermal-forcing vs. auto-regressive provinces",
        fontsize=12, fontweight="bold", pad=10,
    )

    out = os.path.join(HERE, "eip_map_static.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Static PNG   →  {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating EIP province maps …\n")
    make_interactive()
    make_static()
    print("\nAll done.")
