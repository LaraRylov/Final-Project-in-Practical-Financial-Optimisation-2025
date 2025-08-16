

# ========== Step 7 (CVaR) ==========
# ===================================


"""
Visualization for CVaR backtesting outputs (with external benchmark overlay).

Reads from GAMS (CVaR exports) where the file name is 'Step7.gms':
  - weights_long.csv  or  holdings_long.csv
  - wealth.csv
  - risk.csv
  - indices.csv      (IndexActual, IndexMean, IndexWorst, IndexBest, IndexBM if you already stored it)
  - scenarios_monthly.csv

Also reads external benchmark from a parquet file (same as DR):
  - FILE_PATH (parquet with weekly returns), column BENCH_ISIN
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

# ========= SETTINGS =========
BASE = r"/Users/sarasvenstrup/Desktop/Kanditaten/PFO 2025/Day 4 - Scenario Optimization"
SAVE_PNG = True

# External benchmark source (Parquet)
FILE_PATH  = "/Users/sarasvenstrup/Desktop/PFOData.gzip"
BENCH_ISIN = "DK0060259430"
TRAIN_END  = "2019-08-08"   # last in-sample date in GAMS

# IMPORTANT: fix the lag so benchmark is identical across models
LAG_BENCH = -1

# ========= UTILITIES =========
def need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return path

def load_csv(name):
    df = pd.read_csv(need(os.path.join(BASE, name)))
    if "ShiftNum" not in df.columns and "Shift" in df.columns:
        df["ShiftNum"] = df["Shift"].astype(str).str.extract(r"(\d+)").astype(int)
    return df

# ========= LOAD WEIGHTS → WEIGHT MATRIX W =========
weights_name = "weights_long.csv" if os.path.exists(os.path.join(BASE, "weights_long.csv")) else "holdings_long.csv"
weights_raw  = load_csv(weights_name)

def build_weight_matrix(weights_df: pd.DataFrame) -> pd.DataFrame:
    asset_col = next((c for c in ["Asset","ISIN","Ticker","Symbol","Security"] if c in weights_df.columns), None)
    if asset_col is None:
        raise ValueError(f"{weights_name} must include an asset column (Asset/ISIN/Ticker/Symbol/Security).")
    if "ShiftNum" not in weights_df.columns:
        if "Shift" in weights_df.columns:
            weights_df["ShiftNum"] = weights_df["Shift"].astype(str).str.extract(r"(\d+)").astype(int)
        else:
            raise ValueError(f"{weights_name} must include Shift or ShiftNum.")

    if "Weight" in weights_df.columns:
        wcol = "Weight"
        df = weights_df[["ShiftNum", asset_col, wcol]].copy()
        df[wcol] = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0)
        W = df.pivot(index="ShiftNum", columns=asset_col, values=wcol).fillna(0.0)
    elif "Holdings" in weights_df.columns:
        hcol = "Holdings"
        df = weights_df[["ShiftNum", asset_col, hcol]].copy()
        df[hcol] = pd.to_numeric(df[hcol], errors="coerce").fillna(0.0)
        df["_sum"] = df.groupby("ShiftNum")[hcol].transform("sum").replace(0, np.nan)
        df["Weight"] = (df[hcol] / df["_sum"]).fillna(0.0)
        W = df.pivot(index="ShiftNum", columns=asset_col, values="Weight").fillna(0.0)
    else:
        raise ValueError(f"{weights_name} must contain 'Weight' or 'Holdings'.")

    # row-normalize (defensive)
    W = (W.T / W.sum(axis=1).replace(0, np.nan)).T.fillna(0.0)
    return W

W = build_weight_matrix(weights_raw)

# ========= LOAD OTHER INPUTS =========
wealth   = load_csv("wealth.csv").sort_values("ShiftNum").reset_index(drop=True)
risk     = load_csv("risk.csv").sort_values("ShiftNum").reset_index(drop=True)
indices  = load_csv("indices.csv").sort_values("ShiftNum").reset_index(drop=True)

# ========= EXTERNAL BENCHMARK (parquet → weekly → index → resampled to #shifts) =========
oos_start = (pd.to_datetime(TRAIN_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
df_all    = pd.read_parquet(FILE_PATH)
if not isinstance(df_all.index, pd.DatetimeIndex):
    raise ValueError("Parquet index must be a DatetimeIndex.")
oos_end   = df_all.index.max().strftime("%Y-%m-%d")

bench_ret_weekly = df_all.loc[oos_start:oos_end, BENCH_ISIN].astype(float).dropna()
if isinstance(bench_ret_weekly, pd.DataFrame):
    bench_ret_weekly = bench_ret_weekly.iloc[:, 0]
if bench_ret_weekly.empty:
    raise ValueError(f"No benchmark data for {BENCH_ISIN} between {oos_start} and {oos_end}.")

bench_ix = (1.0 + bench_ret_weekly).cumprod()
bench_ix = 100.0 * bench_ix / float(bench_ix.iloc[0])

# Downsample to number of shifts in indices
n_shifts = len(indices)
vals = bench_ix.to_numpy(dtype="float64").ravel()
if np.isnan(vals).any():
    vals = pd.Series(vals).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype="float64")
xp = np.arange(vals.size, dtype="float64")
x_target = np.linspace(0.0, float(vals.size - 1), n_shifts)
indices["IndexBM"] = np.interp(x_target, xp, vals)

# ========= SCENARIOS (flexible reader) =========
def load_scenarios_flexible(filename="scenarios_monthly.csv") -> pd.DataFrame:
    df = pd.read_csv(need(os.path.join(BASE, filename)))

    # Two-column combined keys case
    two_col_combined = (
        df.shape[1] == 2 and
        df.columns[0].lower().startswith("shift") and
        (df.iloc[:,0].astype(str).str.contains(",").any())
    )
    if two_col_combined:
        parts = df.iloc[:, 0].astype(str).str.split(",", expand=True)
        if parts.shape[1] < 3:
            raise ValueError("Couldn't split combined key into Shift, Scenario, Asset.")
        tidy = pd.DataFrame({
            "Shift":    parts[0].str.strip(),
            "Scenario": parts[1].str.strip(),
            "Asset":    parts[2].str.strip(),
            "Return":   pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0.0)
        })
        tidy["ShiftNum"] = tidy["Shift"].astype(str).str.extract(r"(\d+)").astype(int)
        tidy.drop(columns=["Shift"], inplace=True)
        return tidy

    # Otherwise detect tidy columns
    def pick(cands):
        for c in cands:
            if c in df.columns: return c
        return None

    col_shift  = pick(["ShiftNum","Shift"])
    col_scen   = pick(["Scenario","Scen","ScenarioID","Path","Shock"])
    col_asset  = pick(["Asset","ISIN","Ticker","Symbol","Security"])
    col_ret    = pick(["Return","Ret","r","ExRet","ScenarioReturn","MonthlyReturn","MthlyReturn","nthlyReturn"])

    if not all([col_scen, col_asset, col_ret]) or (col_shift is None and "Shift" not in df.columns):
        raise ValueError("scenarios_monthly.csv: couldn't detect Shift/Scenario/Asset/Return columns.")

    tidy = df[[col_shift if col_shift else "Shift", col_scen, col_asset, col_ret]].copy()
    tidy.columns = ["ShiftLike","Scenario","Asset","Return"]
    if col_shift == "ShiftNum":
        tidy["ShiftNum"] = tidy["ShiftLike"].astype(int)
    else:
        tidy["ShiftNum"] = tidy["ShiftLike"].astype(str).str.extract(r"(\d+)").astype(int)
    tidy.drop(columns=["ShiftLike"], inplace=True)
    tidy["Return"] = pd.to_numeric(tidy["Return"], errors="coerce").fillna(0.0)
    return tidy

scen = load_scenarios_flexible()

# ========= PER-SHIFT SCENARIO EVALUATION =========
W_long = (
    W.reset_index()
     .melt(id_vars="ShiftNum", var_name="Asset", value_name="Weight")
     .fillna(0.0)
)

shifts = sorted(set(scen["ShiftNum"]).intersection(set(W_long["ShiftNum"])))
rows = []
for sh in shifts:
    w_sh  = W_long[W_long["ShiftNum"] == sh][["Asset","Weight"]]
    sc_sh = scen[scen["ShiftNum"] == sh][["Scenario","Asset","Return"]]
    merged = sc_sh.merge(w_sh, on="Asset", how="left").fillna({"Weight": 0.0, "Return": 0.0})
    port_by_scen = (merged["Weight"] * merged["Return"]).groupby(merged["Scenario"], sort=False).sum()
    rows.append({"ShiftNum": sh,
                 "Worst": float(port_by_scen.min()),
                 "Mean":  float(port_by_scen.mean()),
                 "Best":  float(port_by_scen.max())})
per_shift = pd.DataFrame(rows).sort_values("ShiftNum").reset_index(drop=True)

# ========= RETURNS PER SHIFT (Realized + Benchmark with FIXED lag) =========
indices = indices.sort_values("ShiftNum").reset_index(drop=True)
realized_ret = indices.set_index("ShiftNum")["IndexActual"].pct_change()
bench_ret    = indices.set_index("ShiftNum")["IndexBM"].pct_change()

best_lag = int(LAG_BENCH)  # fixed (0 by default)
bench_ret_aligned = bench_ret.shift(best_lag)

# Assemble per-shift DataFrame and drop NaNs
plot_df = (
    pd.DataFrame({
        "ShiftNum": realized_ret.index,
        "Realized": realized_ret.values,
        "Worst":    per_shift.set_index("ShiftNum")["Worst"].reindex(realized_ret.index).values,
        "Mean":     per_shift.set_index("ShiftNum")["Mean"].reindex(realized_ret.index).values,
        "Best":     per_shift.set_index("ShiftNum")["Best"].reindex(realized_ret.index).values,
        "Benchmark": bench_ret_aligned.reindex(realized_ret.index).values,
    })
    .dropna()
    .reset_index(drop=True)
)

print(f"[Info] Benchmark lag applied: {best_lag} shift(s).")

# ========= PLOT 1: Weights (100% stacked) =========
W100 = (W.T / W.sum(axis=1).replace(0, np.nan)).T.fillna(0)
K = 25  # match DR plotting
avg_w = W100.mean(axis=0).sort_values(ascending=False)
keep = avg_w.index[:K]
other = W100.columns.difference(keep)

# Create plotting DataFrame
Wplot = W100[keep].copy()
if len(other) > 0:
    Wplot["Other"] = W100[other].sum(axis=1)

# Define colors: palette for top assets, grey for "Other"
palette = sns.color_palette("tab20", n_colors=len(keep))
if "Other" in Wplot.columns:
    palette = palette + [(0.6, 0.6, 0.6)]  # grey for Other

# Plot
fig1, ax1 = plt.subplots(figsize=(14, 6))
Wplot.plot(kind="bar", stacked=True, ax=ax1, width=0.95, color=palette)

# Formatting
ax1.set_title("Portfolio Composition (100% stacked)")
ax1.set_xlabel("Months")
ax1.set_ylabel("Composition")
ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
ax1.set_ylim(0, 1)

# X-axis tick spacing
step = max(1, len(Wplot.index) // 20)
ax1.set_xticks(range(0, len(Wplot.index), step))
ax1.set_xticklabels(Wplot.index[::step], rotation=0)

# Legend
ncol = 6 if Wplot.shape[1] >= 12 else max(3, Wplot.shape[1] // 2)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
           ncol=ncol, title="Asset")

plt.tight_layout()
if SAVE_PNG:
    fig1.savefig(os.path.join(BASE, "plot_7_6_1_weights.png"),
                 dpi=200, bbox_inches="tight")

# ========= PLOT 2: Per-shift returns (Realized vs scenario-based vs Benchmark) =========
figR, axR = plt.subplots(figsize=(12, 6))
x = plot_df["ShiftNum"].to_numpy()
axR.plot(x, plot_df["Realized"],  linewidth=1.2, label="Realized (ex-post)", color="cornflowerblue")
axR.plot(x, plot_df["Mean"],      linewidth=1.2, linestyle="--", label="Scenario Mean (ex-ante)", color="gold")
axR.plot(x, plot_df["Worst"],     linewidth=1.2, label="Scenario Worst (ex-ante)", color="crimson")
axR.plot(x, plot_df["Best"],      linewidth=1.2, label="Scenario Best (ex-ante)",  color="seagreen")
axR.plot(x, plot_df["Benchmark"], linewidth=1.2, linestyle="--", color="silver", label="Benchmark")
axR.set_title("Monthly returns: realized vs scenario-based (min/avg/max) vs benchmark")
axR.set_xlabel("Months")
axR.set_ylabel("Return")
axR.yaxis.set_major_formatter(PercentFormatter(1.0))
axR.grid(True, linestyle=":", linewidth=0.5)
axR.legend(loc="best")
plt.tight_layout()
if SAVE_PNG:
    figR.savefig(os.path.join(BASE, "plot_7_6_3_returns_per_shift.png"), dpi=200, bbox_inches="tight")

# ========= PLOT 3: Cumulative index (Actual + ex-ante + Benchmark) =========
def to_index_from_returns(shifts: np.ndarray, r: pd.Series) -> pd.DataFrame:
    order = np.argsort(shifts)
    srt_sn = shifts[order]
    srt_r  = np.asarray(r)[order]
    ix = 100.0 * np.cumprod(1.0 + srt_r)
    return pd.DataFrame({"ShiftNum": srt_sn, "Index": ix})

idx_mean  = to_index_from_returns(plot_df["ShiftNum"].values, plot_df["Mean"])
idx_worst = to_index_from_returns(plot_df["ShiftNum"].values, plot_df["Worst"])
idx_best  = to_index_from_returns(plot_df["ShiftNum"].values, plot_df["Best"])
idx_bench = to_index_from_returns(plot_df["ShiftNum"].values, plot_df["Benchmark"])

indices_aligned = indices.merge(plot_df[["ShiftNum"]], on="ShiftNum", how="inner")

fig2, ax2 = plt.subplots(figsize=(12, 6))
x_all = indices_aligned["ShiftNum"].to_numpy()
ax2.plot(x_all, indices_aligned["IndexActual"],  label="Actual (ex-post)", linewidth=1.5, color="cornflowerblue")
ax2.plot(idx_mean["ShiftNum"],  idx_mean["Index"],  label="Ex-ante Mean (Python)",  linewidth=1.5, linestyle="--", color="gold")
ax2.plot(idx_worst["ShiftNum"], idx_worst["Index"], label="Ex-ante Worst (Python)", linewidth=1.5, color="crimson")
ax2.plot(idx_best["ShiftNum"],  idx_best["Index"],  label="Ex-ante Best (Python)",  linewidth=1.5, color="seagreen")
ax2.plot(idx_bench["ShiftNum"], idx_bench["Index"], label=f"Benchmark", linewidth=1.8, linestyle="--", color="silver")
ax2.set_title("Portfolio Value Index: Actual vs Ex-ante (Python) vs Benchmark (start = 100)")
ax2.set_xlabel("Shift")
ax2.set_ylabel("Index Level")
ax2.legend(loc="best")
ax2.grid(True, linestyle=":", linewidth=0.5)
plt.tight_layout()
if SAVE_PNG:
    fig2.savefig(os.path.join(BASE, "plot_7_6_2_indices.png"), dpi=200, bbox_inches="tight")

# ========= PLOT 2A: One-step-ahead value per shift =========
# Pick W0
wealth_col_candidates = [c for c in wealth.columns if "wealth" in c.lower() or c.lower() in ("nav","value","portfolio","aum")]
wcol = wealth_col_candidates[0] if wealth_col_candidates else None
first_shift = int(plot_df["ShiftNum"].iloc[0])
if wcol and "ShiftNum" in wealth.columns:
    s0 = wealth.loc[wealth["ShiftNum"] == first_shift, wcol].dropna()
    W0 = float(s0.iloc[0]) if not s0.empty else float(wealth[wcol].dropna().iloc[0])
elif wcol:
    W0 = float(wealth[wcol].dropna().iloc[0])
else:
    W0 = 1_000_000.0  # fallback

# Realized wealth path (base each month)
r_real = plot_df.set_index("ShiftNum")["Realized"]
W_real = W0 * (1.0 + r_real).cumprod()

# Ex-ante one-step-ahead values (strategy-dependent)
W_base = W_real.shift(1).fillna(W0)
r_mean = plot_df.set_index("ShiftNum")["Mean"]
r_wrst = plot_df.set_index("ShiftNum")["Worst"]
r_best = plot_df.set_index("ShiftNum")["Best"]
V_mean  = W_base * (1.0 + r_mean)
V_worst = W_base * (1.0 + r_wrst)
V_best  = W_base * (1.0 + r_best)

# IMPORTANT: Benchmark value path independent of strategy (identical across models)
r_bench = plot_df.set_index("ShiftNum")["Benchmark"]
W_bench = W0 * (1.0 + r_bench).cumprod()

x = plot_df["ShiftNum"].to_numpy()
figV, axV = plt.subplots(figsize=(12, 6))
axV.plot(x, W_real.values,  linewidth=1.2,                label="Realized (ex-post)",       color="cornflowerblue")
axV.plot(x, V_mean.values,  linewidth=1.2, linestyle="--", label="Scenario Mean (ex-ante)", color="gold")
axV.plot(x, V_worst.values, linewidth=1.2,                label="Scenario Worst (ex-ante)", color="crimson")
axV.plot(x, V_best.values,  linewidth=1.2,                label="Scenario Best (ex-ante)",  color="seagreen")
axV.plot(x, W_bench.values, linewidth=1.2, linestyle="--", label="Benchmark",                color="silver")

axV.set_title("Portfolio value: realized vs scenario-based (min/avg/max) vs benchmark")
axV.set_xlabel("Months")
axV.set_ylabel("Portfolio Value")
axV.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
axV.grid(True, linestyle=":", linewidth=0.5)
axV.legend(loc="best")
plt.tight_layout()
if SAVE_PNG:
    figV.savefig(os.path.join(BASE, "plot_7_6_3_values_per_shift.png"), dpi=200, bbox_inches="tight")

# ========= PLOT 4: Risk path =========
fig3, ax3 = plt.subplots(figsize=(12, 6))
xr = risk["ShiftNum"].to_numpy()
ax3.plot(xr, risk["VaR"],  label="VaR")
ax3.plot(xr, risk["CVaR"], label="CVaR")
ax3.set_title("Risk Path per Shift")
ax3.set_xlabel("Shift")
ax3.set_ylabel("Currency units")
ax3.legend(loc="best")
ax3.grid(True, linestyle=":", linewidth=0.5)
plt.tight_layout()
if SAVE_PNG:
    fig3.savefig(os.path.join(BASE, "plot_7_6_4_risk.png"), dpi=200, bbox_inches="tight")

plt.show()

# ========= LOG =========
print("\n[CVaR] Weights source:", weights_name)
print("[CVaR] External benchmark:", BENCH_ISIN, "from", FILE_PATH, "| OOS:", oos_start, "→", oos_end)
print("[CVaR] Benchmark lag applied:", best_lag, "shift(s)")
print("[CVaR] Saved PNGs to:", BASE if SAVE_PNG else "(not saved)")
