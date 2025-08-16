
# ========== Step 6 (extra) ==========
# ====================================

# This code takes in given weights, visualizes performance, computes summary stats,
# and adds a SINGLE annualized "Downside Regret" for 2019-08-09 → 2025-07-25 (no rolling).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ USER INPUTS ============
FILE_PATH  = "/Users/sarasvenstrup/Desktop/PFOData.gzip"
BENCH_ISIN = "DK0060259430"

start_date = "2013-01-09"
end_date   = "2019-08-08"
oos_cap    = pd.Timestamp("2025-07-25")

# ----- Step 5 weights  -----
wA_given = {
    "LU0376447578": 0.003081, "LU0376446257": 0.016278, "LU1893597564": 0.018179, "DK0060189041": 0.013571,
    "DK0010264456": 0.004307, "DK0060051282": 0.067207, "DK0016023229": 0.173975, "DK0061553245": 0.121975,
    "DK0016261910": 0.002880, "DK0016262728": 0.021398, "DK0016262058": 0.016036, "DK0010106111": 0.070756,
    "DK0060240687": 0.014926, "DK0060005254": 0.009093, "DK0010301324": 0.013154, "DK0060300929": 0.004600,
    "DK0060158160": 0.022246, "LU1028171921": 0.008685, "LU0230817339": 0.002867, "DK0060815090": 0.029126,
    "DK0061067220": 0.003327, "DK0060498509": 0.033982, "DK0060498269": 0.009621, "DK0061134194": 0.011148,
    "DK0016300403": 0.030650, "DK0061150984": 0.170751, "IE00B02KXK85": 0.002829, "DE000A0Q4R36": 0.011363,
    "IE00B27YCF74": 0.004856, "IE00B1XNHC34": 0.002814, "DE000A0H08H3": 0.011918, "DE000A0Q4R28": 0.003026,
    "IE00B0M63516": 0.003230, "IE00B0M63623": 0.007913, "DE000A0H08S0": 0.010081, "IE00B42Z5J44": 0.003344,
    "DE000A0H08Q4": 0.008403, "IE00B5377D42": 0.001291, "IE00B0M63391": 0.002372, "IE00B2QWCY14": 0.008656,
    "DK0061544178": 0.007710, "DK0061544418": 0.016373,
}
wB_given = {
    "LU0376446257": 0.279913, "LU1893597564": 0.063349, "DK0010264456": 0.192849, "DK0010301324": 0.010709,
    "LU1028171921": 0.041028, "DK0060498269": 0.011071, "DK0061134194": 0.049378, "DE000A0Q4R36": 0.009240,
    "IE00B0M63516": 0.000131, "DE000A0H08S0": 0.103524, "DE000A0H08Q4": 0.030468, "IE00B2QWCY14": 0.208341,
}

# Stats config
alpha = 0.05                   # for calc_stats
g_4w  = 0.001651               # 4-week (monthly) downside regret threshold (decimal)
ANNUAL_FACTOR_4W = 365.25 / 28.0  # ≈ 13.0446 blocks/year → annualize 4-wk metric

# If calc_stats is in a local package:
import sys
sys.path.append("/.venv")  # adjust if needed
from PFOStep3 import calc_stats

# ---------- Helpers ----------
def ensure_index_tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Force DatetimeIndex and drop timezone info to avoid tz-aware/naive slicing errors."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    try:
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
    except Exception:
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
    return df.sort_index()

def validate_and_align_weights(raw_weights: dict | pd.Series, available_isins: list[str]) -> pd.Series:
    if isinstance(raw_weights, dict):
        w = pd.Series(raw_weights, dtype=float)
    elif isinstance(raw_weights, pd.Series):
        w = raw_weights.astype(float)
    else:
        raise TypeError("Provide weights as dict or pandas Series indexed by ISINs.")
    w = w.reindex(available_isins).fillna(0.0)
    if (w < -1e-12).any():
        raise ValueError("Weights must be non-negative for long-only portfolios.")
    if w.sum() <= 0:
        raise ValueError("Weights sum to zero; provide at least one positive weight.")
    return w / w.sum()

def avg_downside_regret_4w_entire_series(weekly_returns: pd.Series, g_month_4w: float) -> float:
    """
    SINGLE expected downside regret over ENTIRE series (no rolling), via non-overlapping 4-week blocks.
    Returns a unitless decimal for a 4-week period.
    """
    vals = np.asarray(weekly_returns.dropna(), dtype="float64")
    n_blocks = vals.size // 4
    if n_blocks <= 0:
        return np.nan
    vals = vals[:n_blocks * 4]
    r_4w = (1.0 + vals.reshape(n_blocks, 4)).prod(axis=1) - 1.0
    y = np.maximum(0.0, g_month_4w - r_4w)
    return float(y.mean())

def main():
    # ---------- Load ----------
    df = pd.read_parquet(FILE_PATH)
    # If there's a 'date' column instead of index, use it
    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df["date"], errors="coerce")).drop(columns=["date"])
    # Normalize column labels
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    else:
        df.columns = pd.Index([c[0] if isinstance(c, tuple) else c for c in df.columns])
    df.columns = df.columns.astype(str).str.strip()
    df = ensure_index_tz_naive(df)

    # ---------- OOS window ----------
    oos_start = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # 2019-08-09

    # ---------- Investable set ----------
    candidates = sorted(set(list(wA_given.keys()) + list(wB_given.keys())))
    names = [c for c in candidates if c in df.columns]
    if not names:
        raise ValueError("None of the ISINs in your given weights are present in the dataset.")

    # ---------- OOS returns panel & benchmark ----------
    R_oos = (
        df.loc[oos_start:oos_cap, names]
          .apply(pd.to_numeric, errors="coerce")
          .dropna(how="any")
    )
    if BENCH_ISIN not in df.columns:
        raise ValueError(f"Benchmark ISIN {BENCH_ISIN} not found in dataset.")
    bench = df.loc[oos_start:oos_cap, BENCH_ISIN].astype(float).dropna()

    # Align dates
    idx = R_oos.index.intersection(bench.index)
    R_oos = R_oos.loc[idx]
    bench = bench.loc[idx]
    if R_oos.empty:
        raise ValueError("No overlapping OOS data between assets and benchmark.")

    # ---------- Align weights ----------
    wA = validate_and_align_weights(wA_given, list(R_oos.columns))
    wB = validate_and_align_weights(wB_given, list(R_oos.columns))

    # ---------- Buy-and-hold wealth paths ----------
    growth = (1.0 + R_oos).cumprod()  # per-asset growth of 1
    wealth_A = (growth * wA).sum(axis=1)
    wealth_B = (growth * wB).sum(axis=1)
    wealth_M = (1.0 + bench).cumprod()

    # Normalize to start at 1
    wealth_A /= wealth_A.iloc[0]
    wealth_B /= wealth_B.iloc[0]
    wealth_M /= wealth_M.iloc[0]

    # Weekly returns of the 3 wealth series
    retA = wealth_A.pct_change().dropna()
    retB = wealth_B.pct_change().dropna()
    retM = wealth_M.pct_change().dropna()

    # ---------- Plot growth of 1 ----------
    plt.figure(figsize=(8, 5))
    plt.plot(wealth_A, label="Strategy 1 (buy-and-hold)", color="lightgreen")
    plt.plot(wealth_B, label="Strategy 2 (buy-and-hold)", color="pink")
    plt.plot(wealth_M, label="Benchmark", color="silver", linestyle="--")
    plt.ylabel("Wealth (growth of 1 DKK)")
    plt.title("Out-of-sample Wealth (2019–2025)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- Summary statistics ----------
    rets = pd.concat([
        pd.Series(retA, name="Strategy 1"),
        pd.Series(retB, name="Strategy 2"),
        pd.Series(retM, name="Benchmark"),
    ], axis=1).dropna()

    stats_df = calc_stats(rets, alpha=alpha, cvar_method="compound")

    # ---------- SINGLE Expected Downside Regret over whole OOS (no rolling), then ANNUALIZE ----------
    dsr_4w_A = avg_downside_regret_4w_entire_series(retA, g_4w)
    dsr_4w_B = avg_downside_regret_4w_entire_series(retB, g_4w)
    dsr_4w_M = avg_downside_regret_4w_entire_series(retM, g_4w)

    # Annualized expected downside regret (unitless decimals per year)
    dsr_ann_A = dsr_4w_A * ANNUAL_FACTOR_4W
    dsr_ann_B = dsr_4w_B * ANNUAL_FACTOR_4W
    dsr_ann_M = dsr_4w_M * ANNUAL_FACTOR_4W

    dsr_series = pd.Series(
        {"Strategy 1": dsr_ann_A, "Strategy 2": dsr_ann_B, "Benchmark": dsr_ann_M},
        name="Downside Regret (annual)"
    )
    stats_df = stats_df.join(dsr_series)

    # Formatting: show annual return/std dev/drawdown/CVaR as percents; regret stays unitless (decimal)
    pct_cols = ["Annual Return", "Annual Std Dev", f"Annual CVaR (α={int(alpha*100)}%)", "Max Drawdown (loss)"]
    for col in pct_cols:
        if col in stats_df.columns:
            stats_df[col] = (stats_df[col] * 100).round(2)
    if "TUW share (%)" in stats_df.columns:
        stats_df["TUW share (%)"] = stats_df["TUW share (%)"].round(1)
    # If you prefer the annual downside regret in percent, uncomment:
    # stats_df["Downside Regret (annual)"] = (stats_df["Downside Regret (annual)"] * 100).round(2)

    print("\n=== OOS Summary Statistics ===")
    print(stats_df)

    # Optional: save outputs
    stats_df.to_csv("oos_stats_step4_3_with_annual_dsr.csv", index=True)
    pd.DataFrame({
        "Wealth_A": wealth_A,
        "Wealth_B": wealth_B,
        "Wealth_Benchmark": wealth_M
    }).to_csv("oos_wealth_paths_step4_3.csv")

if __name__ == "__main__":
    main()
