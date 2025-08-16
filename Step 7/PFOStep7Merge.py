
# ========== Step 7 (Comparison) ==========
# =========================================


# the following is code to visualize the performance of both strategies CVaR and DR.

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

# === SETTINGS ===
BASE = r"/Users/sarasvenstrup/Desktop/Kanditaten/PFO 2025/Day 4 - Scenario Optimization"
FILE_PATH  = "/Users/sarasvenstrup/Desktop/PFOData.gzip"   # weekly returns parquet
BENCH_ISIN = "DK0060259430"
TRAIN_END  = "2019-08-08"  # last training date
SAVE_PNG   = False
BENCH_LAG  = -2  # force benchmark lag

# === Helper: Load realized path from wealth + indices ===
def load_realized_path(wealth_file, indices_file):
    wealth   = pd.read_csv(os.path.join(BASE, wealth_file))
    indices  = pd.read_csv(os.path.join(BASE, indices_file))

    # Make sure ShiftNum exists
    if "ShiftNum" not in wealth.columns and "Shift" in wealth.columns:
        wealth["ShiftNum"] = wealth["Shift"].astype(str).str.extract(r"(\d+)").astype(int)
    if "ShiftNum" not in indices.columns and "Shift" in indices.columns:
        indices["ShiftNum"] = indices["Shift"].astype(str).str.extract(r"(\d+)").astype(int)

    wealth  = wealth.sort_values("ShiftNum").reset_index(drop=True)
    indices = indices.sort_values("ShiftNum").reset_index(drop=True)

    # Starting value
    wealth_col_candidates = [c for c in wealth.columns if "wealth" in c.lower() or c.lower() in ("nav", "value", "portfolio", "aum")]
    wcol = wealth_col_candidates[0] if wealth_col_candidates else None

    first_shift = int(indices["ShiftNum"].iloc[0])
    if wcol and "ShiftNum" in wealth.columns:
        s0 = wealth.loc[wealth["ShiftNum"] == first_shift, wcol].dropna()
        W0 = float(s0.iloc[0]) if not s0.empty else float(wealth[wcol].dropna().iloc[0])
    elif wcol:
        W0 = float(wealth[wcol].dropna().iloc[0])
    else:
        W0 = 1_000_000.0

    # Realized value path
    realized_ret = indices.set_index("ShiftNum")["IndexActual"].pct_change().dropna()
    W_real = W0 * (1.0 + realized_ret).cumprod()

    return realized_ret.index.to_numpy(), W_real.to_numpy()

# === Load CVaR and DR realized paths ===
x_cvar, W_cvar = load_realized_path("wealth.csv",   "indices.csv")
x_dr,   W_dr   = load_realized_path("wealthDR.csv", "indicesDR.csv")

# === Build Benchmark index ===
oos_start = (pd.to_datetime(TRAIN_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
df_all    = pd.read_parquet(FILE_PATH)
if not isinstance(df_all.index, pd.DatetimeIndex):
    raise ValueError("Parquet index must be a DatetimeIndex.")

oos_end = df_all.index.max().strftime("%Y-%m-%d")

bench_ret_weekly = df_all.loc[oos_start:oos_end, BENCH_ISIN].astype(float).dropna()
if isinstance(bench_ret_weekly, pd.DataFrame):
    bench_ret_weekly = bench_ret_weekly.iloc[:, 0]
if bench_ret_weekly.empty:
    raise ValueError(f"No benchmark data for {BENCH_ISIN} between {oos_start} and {oos_end}.")

bench_ix = (1.0 + bench_ret_weekly).cumprod()
bench_ix = 100.0 * bench_ix / float(bench_ix.iloc[0])

# Downsample benchmark to DR shift count (assuming DR has longest OOS window)
n_shifts = max(len(x_cvar), len(x_dr))
vals     = bench_ix.to_numpy(dtype="float64")
if np.isnan(vals).any():
    vals = pd.Series(vals).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype="float64")
xp       = np.arange(vals.size, dtype="float64")
x_target = np.linspace(0.0, float(vals.size - 1), n_shifts)
bench_ds = np.interp(x_target, xp, vals)

# === Apply forced lag ===
bench_ds = np.roll(bench_ds, BENCH_LAG)
if BENCH_LAG > 0:
    bench_ds[:BENCH_LAG] = np.nan
elif BENCH_LAG < 0:
    bench_ds[BENCH_LAG:] = np.nan

# === Normalize all to start at same value ===
start_val = W_cvar[0]
W_cvar_n  = W_cvar / W_cvar[0] * start_val
W_dr_n    = W_dr   / W_dr[0]   * start_val
bench_n   = bench_ds / bench_ds[~np.isnan(bench_ds)][0] * start_val

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_cvar, W_cvar_n, color="royalblue", label="CVaR (Min CVaR)")
ax.plot(x_dr,   W_dr_n,   color="midnightblue",     label="Downside Regret (Max Exp. Return)")
ax.plot(x_dr,   bench_n[:len(x_dr)], color="silver", linestyle="--", label=f"Benchmark")

ax.set_title("Comparison of Realized Portfolio Value: CVaR vs Downside Regret vs Benchmark")
ax.set_xlabel("Months")
ax.set_ylabel("Portfolio Value")
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.grid(True, linestyle=":", linewidth=0.5)
ax.legend()

plt.tight_layout()
if SAVE_PNG:
    fig.savefig(os.path.join(BASE, "comparison_CVaR_DR_Benchmark.png"), dpi=200, bbox_inches="tight")
plt.show()
