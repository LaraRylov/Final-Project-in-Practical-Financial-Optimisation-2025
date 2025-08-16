

# ========== Step 3 ==========
# ============================

import numpy as np
import pandas as pd

# -------- 3.1 Prepare Data -------------
file_path = "/Users/sarasvenstrup/Desktop/PFOData.gzip"
df = pd.read_parquet(file_path)

# Next we filter our dataset by ISIN codes, to only look at the 46 assets
# chose by MST feature selection.

isin_list = [
    "LU0589470672", "LU0376447578", "LU0376446257", "LU0471299072", "LU1893597564",
    "DK0060189041", "DK0010264456", "DK0060051282", "LU0332084994", "DK0016023229",
    "DK0061553245", "DK0016261910", "DK0016262728", "DK0016262058", "DK0010106111",
    "DK0060240687", "DK0060005254", "DK0010301324", "DK0060300929", "DK0060158160",
    "LU1028171921", "LU0230817339", "DK0060815090", "DK0061067220", "DK0060498509",
    "DK0060498269", "DK0061134194", "DK0016300403", "DK0061150984", "IE00B02KXK85",
    "DE000A0Q4R36", "IE00B27YCF74", "IE00B1XNHC34", "DE000A0H08H3", "DE000A0Q4R28",
    "IE00B0M63516", "IE00B0M63623", "IE00B5WHFQ43", "DE000A0H08S0", "IE00B42Z5J44",
    "DE000A0H08Q4", "IE00B5377D42", "IE00B0M63391", "IE00B2QWCY14", "DK0061544178",
    "DK0061544418",
]

# Keep only ISINs that exist in df (just to check if they all exist)
isin_list = [isin for isin in isin_list if isin in df.columns]
print(f"Found {len(isin_list)} ISINs in dataset.")

df_filtered = df[isin_list].copy()

# ---------- Date filter ----------
start_date = "2013-01-09"
end_date = "2019-08-08"
df_filtered = df_filtered.loc[start_date:end_date]

# ---------- Params ----------
weeks_per_year = 52
alpha = 0.05
rf_annual = 0.00

# ---------- Metrics ----------

def annualized_return(weekly_returns: pd.Series) -> float:
    total_return = (1 + weekly_returns).prod() - 1
    years = (weekly_returns.index[-1] - weekly_returns.index[0]).days / 365.25
    return (1 + total_return) ** (1 / years) - 1

def annualized_std(weekly_returns: pd.Series) -> float:
    return weekly_returns.std(ddof=1) * np.sqrt(weeks_per_year)

def cvar_empirical(series: pd.Series, alpha: float = 0.05) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().values
    if x.size == 0:
        return np.nan
    var_alpha = np.quantile(x, alpha)
    tail = x[x <= var_alpha]
    if tail.size == 0:
        return np.nan
    return tail.mean()

def annualized_cvar(series: pd.Series, alpha: float = 0.05, method: str = "compound") -> float:
    ann = (1 + series).rolling(weeks_per_year).apply(np.prod, raw=True) - 1
    cvar_ann = cvar_empirical(ann, alpha=alpha)
    return -cvar_ann

def sharpe_ratio(weekly: pd.Series, rf_annual: float = 0.0) -> float:
    r = annualized_return(weekly)
    s = annualized_std(weekly)
    return (r - rf_annual) / s if s > 0 else np.nan

def sortino_ratio(weekly: pd.Series, rf_annual: float = 0.0) -> float:
    mar_w = rf_annual / weeks_per_year
    downside = np.minimum(weekly - mar_w, 0.0)
    downside_std_ann = downside.std(ddof=1) * np.sqrt(weeks_per_year)
    r = annualized_return(weekly)
    return (r - rf_annual) / downside_std_ann if downside_std_ann > 0 else np.nan

def max_drawdown_and_tuw(weekly: pd.Series):
    wealth = (1 + weekly).cumprod()
    run_max = wealth.cummax()
    dd = wealth / run_max - 1.0
    max_dd = dd.min()
    under = (dd < 0).astype(int)
    longest, cur = 0, 0
    for v in under:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    share_under = under.mean()
    return -max_dd, int(longest), float(share_under)

# ---------- Stats calc ----------
def calc_stats(df_in: pd.DataFrame, alpha: float = 0.05, cvar_method="compound"):
    stats = pd.DataFrame(index=df_in.columns)
    stats["Annual Return"] = df_in.apply(annualized_return)
    stats["Annual Std Dev"] = df_in.apply(annualized_std)
    stats[f"Annual CVaR (α={int(alpha*100)}%)"] = df_in.apply(
        annualized_cvar, alpha=alpha, method=cvar_method
    )
    stats["Sharpe"] = df_in.apply(sharpe_ratio, rf_annual=rf_annual)
    stats["Sortino"] = df_in.apply(sortino_ratio, rf_annual=rf_annual)

    md_list, tuw_weeks_list, tuw_share_list = [], [], []
    for col in df_in.columns:
        md, tuw_weeks, tuw_share = max_drawdown_and_tuw(df_in[col])
        md_list.append(md)
        tuw_weeks_list.append(tuw_weeks)
        tuw_share_list.append(tuw_share)

    stats["Max Drawdown (loss)"] = md_list
    stats["Max TUW (weeks)"] = tuw_weeks_list
    stats["TUW share (%)"] = np.array(tuw_share_list) * 100.0

    return stats

# ---------- Output ----------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

stats_df = calc_stats(df_filtered, alpha=alpha, cvar_method="compound")

# Convert some metrics to %
for col in ["Annual Return", "Annual Std Dev", f"Annual CVaR (α={int(alpha*100)}%)", "Max Drawdown (loss)"]:
    stats_df[col] = (stats_df[col] * 100).round(2)
stats_df["TUW share (%)"] = stats_df["TUW share (%)"].round(1)

print(stats_df)



# Lastly, we must compute the Variance Covariance matrix for the chosen assets:


def covariance_matrix(df_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute variance-covariance matrix using:
    Q_{i,j} = (1 / (T-2)) * sum_{t=2}^T (r_{i,t} - mu_i)*(r_{j,t} - mu_j)

    Parameters:
    df_returns : DataFrame with assets as columns and weekly returns as rows

    Returns:
    cov_matrix : DataFrame with covariance values
    """
    # Means for each asset
    means = df_returns.mean()

    # Number of observations
    T = df_returns.shape[0]

    # Center returns (subtract means)
    centered = df_returns - means

    # Compute covariance manually
    cov_matrix = centered.iloc[1:].T @ centered.iloc[1:] / (T - 2)

    # Ensure DataFrame structure
    cov_matrix = pd.DataFrame(cov_matrix, index=df_returns.columns, columns=df_returns.columns)

    return cov_matrix


cov_weekly = covariance_matrix(df_filtered)

# Which we want to annualize:
weeks_per_year = 52
cov_annual = cov_weekly * weeks_per_year

# Save to Desktop
cov_annual.to_csv(
    "/Users/sarasvenstrup/Desktop/covariance_matrix_annual.csv",
    float_format="%.6f"
)







