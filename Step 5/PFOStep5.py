
# ========== Step 5 ==========
# ============================

import numpy as np
import pandas as pd

# -------- 5.1 Calculation of expected downside regret BM for GAMS -------------

# This following code calculates the expected downside regret benchmark for
# four weeks rolling forward for 77 periods.

# Load data first
file_path   = "/Users/sarasvenstrup/Desktop/PFOData.gzip"
BENCH_ISIN  = "DK0060259430"   # Stabil portfolio
g           = 0.001651         # threshold per month

# initial training window (inclusive)
train_start = pd.Timestamp("2013-01-09")
train_end   = pd.Timestamp("2019-08-08")

# step size for rolling forward the training end
step_weeks  = 4

# ---------- 0) Load and normalize datetime index ----------
df = pd.read_parquet(file_path)

if "date" in df.columns:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df = df.set_index("date").sort_index()
else:
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    df = df.sort_index()

# make index tz-naive if needed
try:
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
except Exception:
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass

if BENCH_ISIN not in df.columns:
    raise ValueError(f"Column '{BENCH_ISIN}' not found in the dataframe.")

# ---------- 1) Get full weekly benchmark series from train_start to the end of data ----------
bench_full = (
    df.loc[train_start:, BENCH_ISIN]
      .astype(float)
      .dropna()
      .sort_index()
)
if bench_full.empty:
    raise ValueError(f"No data for {BENCH_ISIN} on/after {train_start.date()}.")

# Ensure strictly increasing weekly index (drop duplicates if any)
bench_full = bench_full[~bench_full.index.duplicated(keep="first")]

# ---------- 2) Locate initial training slice positions ----------
idx = bench_full.index
start_pos = idx.searchsorted(train_start, side="left")
end_pos_0 = idx.searchsorted(train_end,   side="right") - 1
start_pos = max(0, min(start_pos, len(idx)-1))
end_pos_0 = max(0, min(end_pos_0, len(idx)-1))
if end_pos_0 < start_pos:
    raise ValueError("Training end is before training start in the benchmark index.")

# ---------- 3) Helper to compute avg downside regret from a weekly slice ----------
def avg_regret_from_weekly_slice(weekly: pd.Series, g_month_4w: float) -> float:
    """weekly: simple weekly returns; g_month_4w: monthly (4w) decimal threshold"""
    vals = weekly.values.astype(float).reshape(-1)
    n_blocks = vals.size // 4
    if n_blocks == 0:
        return np.nan  # not enough data for one 4-week block
    vals = vals[:n_blocks * 4]
    blocks = vals.reshape(n_blocks, 4)
    r_4w = (1.0 + blocks).prod(axis=1) - 1.0  # 4-week compounded simple returns
    y = np.maximum(0.0, g_month_4w - r_4w)
    return float(y.mean())

# ---------- 4) Roll the training end forward in 4-week steps ----------
avg_regs = []   # expanding-window average regrets (decimal)
ends_used = []  # corresponding end dates
k = 0
while True:
    end_pos = end_pos_0 + k * step_weeks
    if end_pos >= len(idx):
        break
    window_weekly = bench_full.iloc[start_pos:end_pos+1]
    avg_r = avg_regret_from_weekly_slice(window_weekly, g)
    if np.isnan(avg_r):
        # if at some early step we somehow don't have one full 4w block, skip; otherwise stop
        if k == 0:
            # try extending one more step to accumulate enough weeks
            k += 1
            continue
        else:
            break
    avg_regs.append(avg_r)
    ends_used.append(idx[end_pos])
    k += 1

s = len(avg_regs)
if s == 0:
    raise ValueError("Could not form any 4-week blocks in the rolling training windows.")

avg_regs = np.asarray(avg_regs, dtype=float)

# ---------- 5) Print GAMS-ready vector (decimal) ----------

per_line = 12
print(f"* Expanding-window average downside regret (decimal), training end rolls +{step_weeks} weeks each step")
print(f"* Start fixed at {train_start.date()}, initial end at {train_end.date()}, total steps: {s}")
print("parameter avgRegret(m) /")
for i in range(0, s, per_line):
    chunk = " ".join(f"{v:.8f}" for v in avg_regs[i:i+per_line])
    print("  " + chunk)
print("/;")


