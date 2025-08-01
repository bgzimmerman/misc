import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

# --- Assume these already exist:
# truth_da : xarray.DataArray, dims = ("valid_time",), values 0/1 int
# fcst_da  : xarray.DataArray, dims = ("init_time","lead_time"),
#            values = forecast probability [0..1]

# 1. Compute event climatology
climatology = float(truth_da.mean().item())  # e.g. 0.25
# seasonal climatology by month
monthly_clim = (truth_da
                .groupby("valid_time.month")
                .mean()
                .rename("monthly_climatology"))

# 2. Align forecast & truth via valid_time
#    Assume lead_time is integer days:
fcst_aligned = fcst_da.assign_coords(
    valid_time=fcst_da.init_time + xr.to_timedelta(fcst_da.lead_time, "D")
)
# select only times present in truth
common = np.intersect1d(fcst_aligned.valid_time, truth_da.valid_time)
fcst_sel = fcst_aligned.sel(valid_time=common)
truth_sel = truth_da.sel(valid_time=common)

# 3. Compute summary metrics for each lead_time
records = []
for lead in fcst_sel.lead_time.values:
    probs = fcst_sel.sel(lead_time=lead).values.ravel()  # shape (n_init,)
    # corresponding observed flags
    vt = fcst_sel.sel(lead_time=lead).valid_time.values
    obs  = truth_da.sel(valid_time=vt).values.astype(int)

    # contingency at threshold=0.5
    fc_bin = probs >= 0.5
    hits  = np.logical_and( fc_bin,  obs).sum()
    misses= np.logical_and(~fc_bin,  obs).sum()
    fas   = np.logical_and( fc_bin, ~obs).sum()
    cns   = np.logical_and(~fc_bin, ~obs).sum()

    POD = hits / (hits + misses) if hits+misses>0 else np.nan
    FAR = fas  / (fas  + cns)    if fas + cns   >0 else np.nan
    Bias = (hits+fas) / (hits+misses) if hits+misses>0 else np.nan

    # Brier score and skill
    BS = np.mean((probs - obs)**2)
    BS_clim = np.mean((climatology - obs)**2)
    BSS = 1 - BS/BS_clim if BS_clim>0 else np.nan

    # ROC AUC
    try:
        AUC = roc_auc_score(obs, probs)
    except ValueError:
        AUC = np.nan

    records.append({
        "lead_time": lead,
        "POD": POD,
        "FAR": FAR,
        "Bias": Bias,
        "BS": BS,
        "BSS": BSS,
        "AUC": AUC
    })

lead_scores = pd.DataFrame(records).set_index("lead_time")

# 4. Reliability diagram aggregated across all leads & inits
#    Flatten forecast & obs
flat_probs = fcst_sel.values.ravel()
flat_obs   = truth_da.sel(valid_time=fcst_sel.valid_time).values.ravel().astype(int)

# define Wilson CI helper
def wilson_ci(k, n, alpha=0.05):
    z = norm.ppf(1 - alpha/2)
    phat = k / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = (z * np.sqrt(phat*(1-phat)/n + z*z/(4*n*n))) / denom
    return center - half, center + half

n_bins = 10
bins = np.linspace(0, 1, n_bins+1)
bin_ids = np.digitize(flat_probs, bins) - 1  # 0..n_bins-1

rel_stats = []
for k in range(n_bins):
    mask = bin_ids == k
    N = mask.sum()
    if N > 0:
        p_mean = flat_probs[mask].mean()
        o_mean = flat_obs[mask].mean()
        k_obs  = flat_obs[mask].sum()
        lo, hi = wilson_ci(k_obs, N)
    else:
        p_mean = np.nan; o_mean = np.nan; lo = np.nan; hi = np.nan
    rel_stats.append({
        "bin_mid": (bins[k] + bins[k+1])/2,
        "count": N,
        "forecast_mean": p_mean,
        "observed_freq": o_mean,
        "lower_CI": lo,
        "upper_CI": hi
    })
reliability_df = pd.DataFrame(rel_stats)

# --- OUTPUT ---

print("=== Event Climatology ===")
print(f"Overall: {climatology:.2%}")
print(monthly_clim)

print("\n=== Lead-Time Scores ===")
print(lead_scores)

print("\n=== Reliability Diagram Table ===")
print(reliability_df)
