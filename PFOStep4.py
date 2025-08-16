

# ========== Step 4 ==========
# ============================

import numpy as np
import pandas as pd
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

# -------- 4.1 Prepare Data and needed Functions -------------
file_path = "/Users/sarasvenstrup/Desktop/PFOData.gzip"
df = pd.read_parquet(file_path)

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

isin_list = [isin for isin in isin_list if isin in df.columns]
print(f"Found {len(isin_list)} ISINs in dataset.")
df_filtered = df[isin_list].copy()

# ---------- Date filter ----------
start_date = "2013-01-09"
end_date = "2019-08-08"
df_filtered = df_filtered.loc[start_date:end_date]


def covariance_matrix(df_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute variance-covariance matrix using:
    Q_{i,j} = (1 / (T-2)) * sum_{t=2}^T (r_{i,t} - mu_i)*(r_{j,t} - mu_j)

    Parameters:
    df_returns : DataFrame with assets as columns and weekly returns as rows

    Returns:
    cov_matrix : DataFrame with covariance values
    """
    means = df_returns.mean()

    T = df_returns.shape[0]

    centered = df_returns - means

    cov_matrix = centered.iloc[1:].T @ centered.iloc[1:] / (T - 2)

    cov_matrix = pd.DataFrame(cov_matrix, index=df_returns.columns, columns=df_returns.columns)

    return cov_matrix


cov_weekly = covariance_matrix(df_filtered)
weeks_per_year = 52
cov_annual = cov_weekly * weeks_per_year





# -------- Step 4 -------------------------------

# We choose Jyske Portefølje Stabil Akk KL, so we log its corresponding ISIN code.
BENCH_ISIN = "DK0060259430" #stabil




# --- 1) Asset weekly returns
R_w = df_filtered.sort_index().copy()


if BENCH_ISIN in R_w.columns:
    R_w = R_w.drop(columns=[BENCH_ISIN])

# --- 2) Benchmark weekly returns pulled from the ORIGINAL df ---
if BENCH_ISIN not in df.columns:
    raise ValueError(f"Benchmark ISIN {BENCH_ISIN} not found in original dataset.")

bench_w = df.loc[R_w.index.min():R_w.index.max(), BENCH_ISIN].reindex(R_w.index)

bench_w = bench_w.astype(float).dropna()
# Align dates to the intersection
common_idx = R_w.index.intersection(bench_w.index)
R_w = R_w.loc[common_idx]
bench_w = bench_w.loc[common_idx]

# --- 3) Annualized parameters ---
mu = ((1 + R_w.mean())**weeks_per_year - 1).values              # geometric annualization
Sigma = (R_w.cov(ddof=1) * weeks_per_year).values
names = R_w.columns.tolist()
n = len(names)

# Annualized benchmark stats as scalars
r_b = float((1 + bench_w.mean())**weeks_per_year - 1)
sigma_b = float(bench_w.std(ddof=1) * np.sqrt(weeks_per_year))






# --- 4) Two Markowitz runs ---
def run_markowitz(mu, Sigma, sigma_b, r_b, names):
    n = len(mu)

    # Run A: maximize return subject to risk <= benchmark
    xA = cp.Variable(n, nonneg=True)
    probA = cp.Problem(cp.Maximize(mu @ xA),
                       [cp.sum(xA) == 1, cp.quad_form(xA, Sigma) <= sigma_b**2])
    probA.solve(solver=cp.ECOS, verbose=False)

    # Run B: minimize risk subject to return >= benchmark
    xB = cp.Variable(n, nonneg=True)
    probB = cp.Problem(cp.Minimize(cp.quad_form(xB, Sigma)),
                       [cp.sum(xB) == 1, mu @ xB >= r_b])
    probB.solve(solver=cp.ECOS, verbose=False)

    def pack(x):
        w = np.array(x.value).ravel()
        ret = float(mu @ w)
        std = float(np.sqrt(w @ Sigma @ w))
        return pd.Series(w, index=names), ret, std

    wA, retA, stdA = pack(xA)
    wB, retB, stdB = pack(xB)

    summary = pd.DataFrame({
        "Metric": ["Exp. Return (annual)", "Std. Dev. (annual)"],
        "Run A (max ER | risk≤bench)": [retA, stdA],
        "Run B (min risk | ER≥bench)": [retB, stdB],
        "Benchmark": [r_b, sigma_b]
    })

    return wA, wB, summary, probA.status, probB.status

wA, wB, summary, statusA, statusB = run_markowitz(mu, Sigma, sigma_b, r_b, names)

print("Solver status — Run A:", statusA, "| Run B:", statusB)
pd.set_option("display.max_columns", None)
print(summary.round(4))

# Save weights
wA.rename("Weight_RunA").to_csv("weights_runA.csv")
wB.rename("Weight_RunB").to_csv("weights_runB.csv")

# Show the largest positions
print("\nTop holdings — Run A:")
print(wA[wA > 0.001].sort_values(ascending=False).head(10))
print("\nTop holdings — Run B:")
print(wB[wB > 0.001].sort_values(ascending=False).head(10))

# So now we have completed the two runs so this is step 4.1 completed.


# Now we move on to complete step 4.2

# ---------- helpers ----------
def gmv_portfolio(Sigma):
    """Global minimum-variance (long-only, fully invested). Returns (weights, std)."""
    n = Sigma.shape[0]
    w = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)),
                      [cp.sum(w) == 1])
    prob.solve(solver=cp.ECOS, verbose=False)
    w_gmv = np.array(w.value).ravel()
    sigma_gmv = float(np.sqrt(w_gmv @ Sigma @ w_gmv))
    return w_gmv, sigma_gmv

def max_return_port(mu):
    """Max return under long-only, fully invested -> puts all weight in best-return asset."""
    n = len(mu)
    w = np.zeros(n)
    w[np.argmax(mu)] = 1.0
    return w

# ---------- frontier grid (equally spaced in variance) ----------
w_gmv, sigma_min = gmv_portfolio(Sigma)
w_maxR = max_return_port(mu)
sigma_max = float(np.sqrt(w_maxR @ Sigma @ w_maxR))  # variance upper bound from best-return corner
var_grid = np.linspace(sigma_min**2, sigma_max**2, 70)  # 10 points equidistant in variance

frontier_std = []
frontier_ret = []
frontier_wts = []

n = len(mu)
for v in var_grid:
    w = cp.Variable(n, nonneg=True)
    # maximize return subject to variance cap v
    prob = cp.Problem(cp.Maximize(mu @ w),
                      [cp.sum(w) == 1,
                       cp.quad_form(w, Sigma) <= float(v)])
    prob.solve(solver=cp.ECOS, verbose=False)
    w_opt = np.array(w.value).ravel()
    er = float(mu @ w_opt)
    sd = float(np.sqrt(w_opt @ Sigma @ w_opt))
    frontier_ret.append(er)
    frontier_std.append(sd)
    frontier_wts.append(w_opt)

frontier_std = np.array(frontier_std)
frontier_ret = np.array(frontier_ret)

# ---------- points for A, B, benchmark ----------
er_A = float(mu @ wA); sd_A = float(np.sqrt(wA @ Sigma @ wA))
er_B = float(mu @ wB); sd_B = float(np.sqrt(wB @ Sigma @ wB))
er_bmk = float(r_b);   sd_bmk = float(sigma_b)

# ---------- plot ----------

plt.figure(figsize=(7,5))
plt.plot(frontier_std**2, frontier_ret, marker='o', linewidth=1.5, label='Efficient Frontier', color = 'royalblue')

var_A   = float(wA @ Sigma @ wA)
var_B   = float(wB @ Sigma @ wB)
var_bmk = float(sigma_b**2)

er_A = float(mu @ wA); er_B = float(mu @ wB); er_bmk = float(r_b)

plt.scatter(var_A, er_A, marker='o', s=120, label='Strategy 1', color = 'lightgreen')
plt.scatter(var_B, er_B, marker='o', s=120, label='Strategy 2', color = 'pink')
plt.scatter(var_bmk, er_bmk, marker='o', s=120, label='Benchmark', color = 'silver')

plt.xlabel('Portfolio Variance')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()






# Now we are ready to complete step 4.3.
# ===== Step 4.3 — Buy-and-hold out-of-sample (2019–2025) =====

# First, we must define the out-of-sample [oos] time period, which is one day after the in-sample date.

oos_start = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
oos_end   = df.index.max().strftime("%Y-%m-%d")


# 1) Build OOS panel and benchmark (weekly returns)
# So here we take the returns from the exactly those assets in the optimal portfolio for both
# strategies on the training data. And likewise with the benchmark.
R_oos = df.loc[oos_start:oos_end, names].apply(pd.to_numeric, errors="coerce").dropna(how="any")
bench  = df.loc[R_oos.index, BENCH_ISIN].astype(float).dropna()
idx = R_oos.index.intersection(bench.index)
R_oos, bench = R_oos.loc[idx], bench.loc[idx]

# 2) Align/renormalize weights
wA_oos = wA.reindex(R_oos.columns).fillna(0.0); wA_oos /= wA_oos.sum()
wB_oos = wB.reindex(R_oos.columns).fillna(0.0); wB_oos /= wB_oos.sum()

# 3) Buy-and-hold wealth paths and weekly returns
growth = (1 + R_oos).cumprod()
wealth_A = (growth * wA_oos).sum(axis=1)
wealth_B = (growth * wB_oos).sum(axis=1)
wealth_M = (1 + bench).cumprod()

wealth_A /= wealth_A.iloc[0]
wealth_B /= wealth_B.iloc[0]
wealth_M /= wealth_M.iloc[0]

retA = wealth_A.pct_change().dropna()
retB = wealth_B.pct_change().dropna()
retM = wealth_M.pct_change().dropna()


# 4) Plot growth of 1
plt.figure(figsize=(8,5))
plt.plot(wealth_A, label="Strategy 1 (buy-and-hold)", color="lightgreen")
plt.plot(wealth_B, label="Strategy 2 (buy-and-hold)", color="pink")
plt.plot(wealth_M, label="Benchmark", color="silver", linestyle="--")
plt.ylabel("Wealth (growth of 1DKK)")
plt.title("Out-of-sample Wealth (2019–2025)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# And now to the summary statistics:

alpha = 0.05

import sys
sys.path.append("/.venv")
from PFOStep3 import calc_stats

# Ensure each is a Series and give it a name
sA = pd.Series(retA.squeeze(), index=retA.index, name="Strategy 1")
sB = pd.Series(retB.squeeze(), index=retB.index, name="Strategy 2")
sM = pd.Series(retM.squeeze(), index=retM.index, name="Benchmark")

# Build the 3-column DataFrame
rets = pd.concat([sA, sB, sM], axis=1).dropna()

stats_df = calc_stats(rets, alpha=alpha, cvar_method="compound")

# Format %
for col in ["Annual Return", "Annual Std Dev", f"Annual CVaR (α={int(alpha*100)}%)", "Max Drawdown (loss)"]:
    stats_df[col] = (stats_df[col] * 100).round(2)
stats_df["TUW share (%)"] = stats_df["TUW share (%)"].round(1)

print(stats_df)





#Extra:

# Calcualtion of CVaR in-sample for benchmark:

alpha = 0.05

# Convert to Series and drop NaNs
bench_insample = bench_w.squeeze().dropna()
bench_insample.name = "Benchmark"

# Compute stats
bench_stats = calc_stats(bench_insample.to_frame(), alpha=alpha, cvar_method="compound")

# Extract CVaR in %
cvar_bench_insample = bench_stats.loc["Benchmark", f"Annual CVaR (α={int(alpha*100)}%)"] * 100
print(f"In-sample Benchmark CVaR (α={alpha*100:.0f}%): {cvar_bench_insample:.2f}%")

