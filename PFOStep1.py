

# ========== Step 1 ==========
# ============================

import numpy as np
import pandas as pd


# -------- 1.1 Prepare Data -------------

# To complete step 1, we first import the given data:
file_path = "/Users/sarasvenstrup/Desktop/PFOData.gzip"
df = pd.read_parquet(file_path)


# Next we filter our dataset by ISIN codes, to only look at the four portfolios.
isin_list = [
    "DK0060259513",  # Vækst
    "DK0060259786",  # Balanceret
    "DK0060259430",  # Stabil
    "DK0060259356"   # Dæmpet
]

# Filter DataFrame to only these columns
df_filtered = df[isin_list]

# -------- 1.2 Prep summary statistics functions -------------

# We now create the summary statistic functions that we use
# throughout the whole project.

weeks_per_year = 52
alpha = 0.05  # 5%


def annualized_return(weekly_returns):
    mean_weekly = weekly_returns.mean()
    return (1 + mean_weekly) ** weeks_per_year - 1

def annualized_std(weekly_returns):
    return weekly_returns.std(ddof=1) * np.sqrt(weeks_per_year)


def cvar_empirical(series, alpha=0.05):
    x = pd.to_numeric(series, errors="coerce").dropna().values
    if x.size == 0:
        return np.nan
    var_alpha = np.quantile(x, alpha)
    tail = x[x <= var_alpha]
    if tail.size == 0:
        return np.nan
    return tail.mean()  # negative

def annualized_cvar(series, alpha=0.05, method="compound"):
    ann = (1 + series).rolling(weeks_per_year).apply(np.prod, raw=True) - 1
    cvar_ann = cvar_empirical(ann, alpha=alpha)   # negative annual tail return
    return -cvar_ann  # return positive annual loss magnitude

def calc_stats(df, periods=[3, 5, 12], alpha=0.05, cvar_method="compound"):
    results = {}
    for years in periods:
        n_weeks = years * weeks_per_year
        df_period = df.tail(n_weeks)

        stats = pd.DataFrame(index=df_period.columns)
        stats["Annual Return"] = df_period.apply(annualized_return)
        stats["Annual Std Dev"] = df_period.apply(annualized_std)
        stats[f"Annual CaVaR (α={int(alpha*100)}%)"] = df_period.apply(
            annualized_cvar, alpha=alpha, method=cvar_method
        )
        results[f"{years}y"] = stats
    return results

# -------- 1.3 Prepare Output -------------

# Table output for Step 1
stats_dict = calc_stats(df_filtered, periods=[3,5,12], alpha=0.05, cvar_method="compound")
for k, v in stats_dict.items():
    print(f"\n--- {k} ---\n", (v*100).round(2))  # convert to %













