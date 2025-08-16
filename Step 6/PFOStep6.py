

# ========== Step 6 ==========
# ============================

# This code takes in given weights from GAMS, and then visualizes the perfomance together with
# calculating the summary statistics. This file is used for both step 4.3, 5.5 and 6.5.


# ============ USER INPUTS ============
# Data
FILE_PATH = "/Users/sarasvenstrup/Desktop/PFOData.gzip"
BENCH_ISIN = "DK0060259430"  # benchmark ISIN: Stabil

# In-sample window used previously
start_date = "2013-01-09"
end_date   = "2019-08-08"



# Given weights for Step 4:

#wA_given = {
#        "DK0060158160": 0.307625,  # Nykredit Alpha Mira
#       "DK0016300403": 0.242113,  # Sampension Invest GEM II
#       "LU0376446257": 0.162484,  # BGF Swiss Small & MidCap Opps A2
#        "DK0060498509": 0.151670,  # Formuepleje Fokus
#       "LU1893597564": 0.082481,  # BSF European Unconstrained Eq I2 PF EUR
#       "DK0060005254": 0.028485,  # Maj Invest Vækstaktier
#       "DK0016262728": 0.020618,  # Jyske Invest High Yield Corp Bond CL
#        "DK0010264456": 0.004523   # Danske Invest Bioteknologi Indeks KL
#}

#wB_given = {
#        "DK0016023229": 0.516185,  # Danske Invest Teknologi Indeks KL
#        "DK0016300403": 0.125467,  # Sampension Invest GEM II
#        "DK0060158160": 0.100208,  # Nykredit Alpha Mira
#        "DK0061553245": 0.055998,  # Wealth Invest HP Engros Korte Danske Obl
#        "DK0061150984": 0.055675,  # Wealth Invest HP Invest Korte Danske Obl
#        "LU0376446257": 0.050503,  # BGF Swiss Small & MidCap Opps A2
#        "DK0060498509": 0.030342,  # Formuepleje Fokus
#        "LU1893597564": 0.027576,  # BSF European Unconstrained Eq I2 PF EUR
#        "DK0060005254": 0.019672,  # Maj Invest Vækstaktier
#        "DK0016262728": 0.017601   # Jyske Invest High Yield Corp Bond CL
#}


# Given weights for Step 5:

#wA_given = {
#    "LU0376447578": 0.003081,
#    "LU0376446257": 0.016278,
#    "LU1893597564": 0.018179,
#    "DK0060189041": 0.013571,
#    "DK0010264456": 0.004307,
#    "DK0060051282": 0.067207,
#    "DK0016023229": 0.173975,
#    "DK0061553245": 0.121975,
#    "DK0016261910": 0.002880,
#    "DK0016262728": 0.021398,
#    "DK0016262058": 0.016036,
#    "DK0010106111": 0.070756,
#    "DK0060240687": 0.014926,
#    "DK0060005254": 0.009093,
#    "DK0010301324": 0.013154,
#    "DK0060300929": 0.004600,
#    "DK0060158160": 0.022246,
#    "LU1028171921": 0.008685,
#    "LU0230817339": 0.002867,
#    "DK0060815090": 0.029126,
#    "DK0061067220": 0.003327,
#    "DK0060498509": 0.033982,
#    "DK0060498269": 0.009621,
#    "DK0061134194": 0.011148,
#    "DK0016300403": 0.030650,
#    "DK0061150984": 0.170751,
#    "IE00B02KXK85": 0.002829,
#    "DE000A0Q4R36": 0.011363,
#    "IE00B27YCF74": 0.004856,
#    "IE00B1XNHC34": 0.002814,
#    "DE000A0H08H3": 0.011918,
#    "DE000A0Q4R28": 0.003026,
#    "IE00B0M63516": 0.003230,
#    "IE00B0M63623": 0.007913,
#    "DE000A0H08S0": 0.010081,
#    "IE00B42Z5J44": 0.003344,
#    "DE000A0H08Q4": 0.008403,
#    "IE00B5377D42": 0.001291,
#    "IE00B0M63391": 0.002372,
#    "IE00B2QWCY14": 0.008656,
#    "DK0061544178": 0.007710,
#    "DK0061544418": 0.016373,
#}

#wB_given = {
#    "LU0376446257": 0.279913,
#    "LU1893597564": 0.063349,
#    "DK0010264456": 0.192849,
#    "DK0010301324": 0.010709,
#    "LU1028171921": 0.041028,
#    "DK0060498269": 0.011071,
#    "DK0061134194": 0.049378,
#    "DE000A0Q4R36": 0.009240,
#    "IE00B0M63516": 0.000131,
#    "DE000A0H08S0": 0.103524,
#    "DE000A0H08Q4": 0.030468,
#    "IE00B2QWCY14": 0.208341,
#}


# Given weights for Step 6:

wA_given = {
    "DK0016023229": 0.21098020,
    "DK0061150984": 0.21275860,
    "DK0060051282": 0.10199120,
    "DK0061553245": 0.10406790,
    "DK0016300403": 0.04761077,
    "DK0060498509": 0.04412155,
    "DK0010106111": 0.03482587,
    "DK0060240687": 0.02360547,
    "LU1893597564": 0.01370798,
    "DK0060189041": 0.01306429,
    "DE000A0H08H3": 0.01276461,
    "DK0060158160": 0.01182111,
    "IE00B2QWCY14": 0.01111960,
    "LU0332084994": 0.01022756,
    "DK0060498269": 0.00952369,
    "DK0010301324": 0.00950356,
    "DK0061134194": 0.00834192,
    "LU0376446257": 0.00837154,
    "DE000A0Q4R36": 0.00712502,
    "DK0016262728": 0.00742604,
    "DE000A0H08Q4": 0.00695904,
    "DK0061067220": 0.00496876,
    "IE00B0M63623": 0.00499406,
    "IE00B0M63516": 0.00407941,
    "LU0230817339": 0.00443718,
    "DK0061544418": 0.00585447,
    "DK0060005254": 0.00531830,
    "IE00B27YCF74": 0.00525675,
    "LU0376447578": 0.00324606,
    "LU1028171921": 0.00312517,
    "IE00B42Z5J44": 0.00348412,
    "DK0061544178": 0.00296034,
    "DE000A0Q4R28": 0.00236521,
    "LU0471299072": 0.00223527,
    "IE00B02KXK85": 0.00211385,
    "DK0010264456": 0.00511019,
    "DK0060300929": 0.00482549,
    "IE00B1XNHC34": 0.00030301,
    "DE000A0H08S0": 0.00923733,
    "IE00B0M63391": 0.00190349,
}

wB_given = {
    "DK0061150984": 0.2197992,
    "DK0061553245": 0.1361602,
    "DK0016023229": 0.0907104,
    "DK0060498509": 0.0506160,
    "DK0016300403": 0.0505841,
    "LU1893597564": 0.0201185,
    "DK0060189041": 0.0220644,
    "DK0060815090": 0.0191006,
    "DK0060240687": 0.0418776,
    "LU0332084994": 0.0223675,
    "DK0060158160": 0.0142765,
    "DK0010301324": 0.0135558,
    "DK0060051282": 0.0753869,
    "LU0376446257": 0.0146464,
    "IE00B2QWCY14": 0.01455099,
    "DE000A0Q4R36": 0.01492325,
    "DE000A0H08S0": 0.01406945,
    "DK0060498269": 0.01191538,
    "DK0016262728": 0.01156740,
    "DK0060300929": 0.00967306,
    "DE000A0H08Q4": 0.00974434,
    "DK0010264456": 0.00667493,
    "IE00B27YCF74": 0.00617613,
    "DE000A0H08H3": 0.0187035,
    "LU0376447578": 0.00224479,
    "LU1028171921": 0.00440175,
    "LU0230817339": 0.00689712,
    "IE00B02KXK85": 0.00354378,
    "IE00B1XNHC34": 0.00129048,
    "IE00B0M63516": 0.00386703,
    "IE00B0M63623": 0.01043621,
    "IE00B42Z5J44": 0.00398164,
    "DK0061544418": 0.01534435,
    "DK0061544178": 0.00309572,
    "LU0471299072": 0.00412110,
    "DK0060005254": 0.00049701
}



# --- after you define wA_given and wB_given ---
wA_given = {str(k).strip(): float(v) for k, v in wA_given.items()}
wB_given = {str(k).strip(): float(v) for k, v in wB_given.items()}

# Stats config
alpha = 0.05  # for CVaR and VaR

# Import the function created earlier :
import sys
sys.path.append("/PyCharmMiscProject")  # adjust if needed
from PFOStep3 import calc_stats
# =====================================

def validate_and_align_weights(raw_weights: dict | pd.Series, available_isins: list[str]) -> pd.Series:
    """
    Aligns provided weights to available ISINs, fills missing as 0, checks nonnegativity,
    and renormalizes to sum to 1.
    """
    if isinstance(raw_weights, dict):
        w = pd.Series(raw_weights, dtype=float)
    elif isinstance(raw_weights, pd.Series):
        w = raw_weights.astype(float)
    else:
        raise TypeError("Provide weights as dict or pandas Series indexed by ISINs.")

    # Keep only assets that are in the data, fill missing with 0
    w = w.reindex(available_isins).fillna(0.0)

    if (w < -1e-12).any():
        raise ValueError("Weights must be non-negative for long-only portfolios.")

    if w.sum() <= 0:
        raise ValueError("Weights sum to zero; provide at least one positive weight.")

    return w / w.sum()

def hist_var(x: pd.Series, a: float) -> float:
    """
    Historical VaR at level a.
    Returns a positive loss number (so a 5% VaR of 2% is returned as 0.02).
    """
    x = pd.Series(x).dropna()
    if x.empty:
        return np.nan
    return float(-(np.nanquantile(x.values, a)))

def main():
    # ---------- Load full panel ----------
    df = pd.read_parquet(FILE_PATH)

    # ---- normalize column names to plain ISIN strings ----
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    else:
        df.columns = pd.Index([c[0] if isinstance(c, tuple) else c for c in df.columns])

    df.columns = df.columns.astype(str).str.strip()

    # ---------- Define OOS window ----------
    oos_start = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    oos_end   = df.index.max().strftime("%Y-%m-%d")

    # ---------- Determine investable set from given weights ----------
    # Use the ISINs present in the data and referenced by the weights
    candidates = sorted(set(list(wA_given.keys()) + list(wB_given.keys())))
    names = [c for c in candidates if c in df.columns]
    if len(names) == 0:
        raise ValueError("None of the ISINs in your given weights are present in the dataset.")

    # ---------- Build OOS panel & benchmark (weekly returns) ----------
    R_oos = (
        df.loc[oos_start:oos_end, names]
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="any")
    )

    if BENCH_ISIN not in df.columns:
        raise ValueError(f"Benchmark ISIN {BENCH_ISIN} not found in dataset.")

    bench = df.loc[R_oos.index, BENCH_ISIN].astype(float).dropna()

    # Align dates
    idx = R_oos.index.intersection(bench.index)
    R_oos = R_oos.loc[idx]
    bench = bench.loc[idx]

    if R_oos.empty:
        raise ValueError("No overlapping OOS data between assets and benchmark.")

    # ---------- Align and renormalize given weights to the OOS columns ----------
    wA = validate_and_align_weights(wA_given, list(R_oos.columns))
    wB = validate_and_align_weights(wB_given, list(R_oos.columns))

    # ---------- Buy-and-hold wealth paths ----------
    # growth = cumulative product of (1 + weekly returns) per asset
    growth = (1.0 + R_oos).cumprod()

    wealth_A = (growth * wA).sum(axis=1)
    wealth_B = (growth * wB).sum(axis=1)
    wealth_M = (1.0 + bench).cumprod()

    # Normalize so all series start at 1
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
    plt.title(f"Out-of-sample Wealth (2019 – 2025)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- Summary statistics (using your calc_stats) ----------
    sA = pd.Series(retA.squeeze(), index=retA.index, name="Strategy 1")
    sB = pd.Series(retB.squeeze(), index=retB.index, name="Strategy 2")
    sM = pd.Series(retM.squeeze(), index=retM.index, name="Benchmark")

    rets = pd.concat([sA, sB, sM], axis=1).dropna()

    stats_df = calc_stats(rets, alpha=alpha, cvar_method="compound")

    # ---------- Add historical VaR (weekly + compounded annual) ----------
    # Weekly VaR from weekly returns
    weekly_var = rets.apply(lambda s: hist_var(s, alpha))

    # Compounded annual returns (rolling 52 weeks), then VaR on those
    ann_comp = (1.0 + rets).rolling(52).apply(lambda w: np.prod(w) - 1.0, raw=False).dropna(how="all")
    annual_var_compound = ann_comp.apply(lambda s: hist_var(s, alpha))

    # Append as new columns (match index order)
    stats_df[f"Weekly VaR (α={int(alpha*100)}%)"] = weekly_var.reindex(stats_df.index)
    stats_df[f"Annual VaR (α={int(alpha*100)}%) (compound)"] = annual_var_compound.reindex(stats_df.index)

    # ---------- Pretty formatting (percentages) ----------
    pct_cols = [
        "Annual Return",
        "Annual Std Dev",
        f"Annual CVaR (α={int(alpha*100)}%)",
        f"Weekly VaR (α={int(alpha*100)}%)",
        f"Annual VaR (α={int(alpha*100)}%) (compound)",
        "Max Drawdown (loss)",
    ]
    for col in pct_cols:
        if col in stats_df.columns:
            stats_df[col] = (stats_df[col] * 100).round(2)
    if "TUW share (%)" in stats_df.columns:
        stats_df["TUW share (%)"] = stats_df["TUW share (%)"].round(1)

    print("\n=== OOS Summary Statistics ===")
    print(stats_df)

    # Optional: save outputs
    stats_df.to_csv("oos_stats_step4_3.csv", index=True)
    pd.DataFrame({
        "Wealth_A": wealth_A,
        "Wealth_B": wealth_B,
        "Wealth_Benchmark": wealth_M
    }).to_csv("oos_wealth_paths_step4_3.csv")

if __name__ == "__main__":
    main()

