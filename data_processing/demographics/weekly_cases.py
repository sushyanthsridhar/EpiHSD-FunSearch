import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no windows will open
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

INPUT_FILE  = "list_data.csv"
OUTPUT_FILE = "cases_by_province_week.csv"
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Loaded {len(df):,} rows")

# ── 2. Drop rows with no symptom date ────────────────────────────────────────
df = df.dropna(subset=["date_begin_symptoms"])
print(f"Rows with date_begin_symptoms: {len(df):,}")

# ── 3. Parse date ────────────────────────────────────────────────────────────
df["date_begin_symptoms"] = pd.to_datetime(df["date_begin_symptoms"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date_begin_symptoms"])
print(f"Rows with valid parsed date: {len(df):,}")

# ── 4. Extract time fields ───────────────────────────────────────────────────
df["year"]  = df["year_begin_symptoms"].fillna(df["date_begin_symptoms"].dt.year).astype(int)
df["month"] = df["month_begin_symptoms"].fillna(df["date_begin_symptoms"].dt.month).astype(int)
df["week"]  = df["week_begin_symptoms"].fillna(df["date_begin_symptoms"].dt.isocalendar().week.astype(int)).astype(int)

# ── 5. Aggregate ─────────────────────────────────────────────────────────────
agg = (
    df.groupby(["province", "year", "month", "week"])
    .size()
    .reset_index(name="total_cases")
    .sort_values(["province", "year", "week"])
    .reset_index(drop=True)
)

# ── 6. Save CSV ──────────────────────────────────────────────────────────────
agg.to_csv(OUTPUT_FILE, index=False)
print(f"\nDone! {len(agg):,} rows written to: {OUTPUT_FILE}")
print(agg.head(10).to_string(index=False))

# ── 7. Helper: build a sortable time label (year + fractional week) ──────────
def add_time_index(df):
    df = df.copy()
    df["time"] = df["year"] + (df["week"] - 1) / 53
    return df.sort_values("time")

# ── 8. Plot helper ────────────────────────────────────────────────────────────
def plot_series(x, y, title, filename):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y, linewidth=1.2, color="#2196F3")
    ax.fill_between(x, y, alpha=0.15, color="#2196F3")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Cases per week", fontsize=10)

    # x-axis: show only whole years
    year_min = int(x.min())
    year_max = int(x.max()) + 1
    year_ticks = list(range(year_min, year_max + 1))
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_ticks, rotation=45, ha="right", fontsize=8)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

# ── 9. One plot per province ──────────────────────────────────────────────────
provinces = agg["province"].unique()
print(f"\nGenerating {len(provinces)} province plots ...")

for prov in provinces:
    sub = add_time_index(agg[agg["province"] == prov])
    safe_name = prov.replace("/", "-").replace(" ", "_")
    fname = os.path.join(PLOTS_DIR, f"{safe_name}.png")
    plot_series(sub["time"], sub["total_cases"], f"Weekly Cases — {prov}", fname)
    print(f"  Saved: {fname}")

# ── 10. Total across all provinces ───────────────────────────────────────────
print("\nGenerating total (all provinces) plot ...")
total = (
    agg.groupby(["year", "week"])["total_cases"]
    .sum()
    .reset_index()
)
total = add_time_index(total)
plot_series(
    total["time"],
    total["total_cases"],
    "Weekly Cases — All Provinces Combined",
    os.path.join(PLOTS_DIR, "00_ALL_PROVINCES.png")
)

print(f"\nAll plots saved to: ./{PLOTS_DIR}/")