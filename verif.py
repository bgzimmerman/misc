import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

# — assume these DataArrays already exist —
# truth_da: dims = ("valid_time",), values ∈ {0,1}, dtype=int
# fcst_da: dims = ("init_time","lead_time"), values ∈ [0,1], dtype=float
# where lead_time is dtype timedelta64[ns] (or [s]).

# 1) Add valid_time coordinate to forecast
fcst_da = fcst_da.assign_coords(
    valid_time = fcst_da.init_time + fcst_da.lead_time
)

# 2) Stack to 1D along a new "point" dimension
fcst_flat = fcst_da.stack(point=("init_time","lead_time"))

# 3) Filter to only those valid_times present in truth_da
#    (cast to numpy datetime64 to be sure)
fcst_times = fcst_flat.valid_time.values.astype("datetime64[ns]")
truth_times = truth_da.valid_time.values.astype("datetime64[ns]")
common = np.intersect1d(fcst_times, truth_times)
mask = np.isin(fcst_times, common)
fcst_flat = fcst_flat.isel(point=np.where(mask)[0])

# 4) Align observed truth to the stacked forecast
#    Selecting truth at each corresponding valid_time
truth_flat = truth_da.sel(valid_time=fcst_flat.valid_time.values)

# 5) Compute overall climatology
climatology = float(truth_da.mean().item())
monthly_clim = (
    truth_da
      .groupby("valid_time.month")
      .mean()
      .rename("monthly_climatology")
)

# 6) Compute lead-time–specific metrics
records = []
# convert lead_time to days (float)
lead_days = (fcst_flat.lead_time.values.astype("timedelta64[s]") /
             np.timedelta64(1, "D"))
unique_leads = np.unique(lead_days)

for lead in unique_leads:
    # select indices for this lead
    idx = np.where(lead_days == lead)[0]
    probs = fcst_flat.values[idx]
    obs   = truth_flat.values[idx].astype(int)

    # contingency at 0.5 threshold
    fc_bin = probs >= 0.5
    hits   = np.logical_and(fc_bin,  obs).sum()
    misses = np.logical_and(~fc_bin, obs).sum()
    fas    = np.logical_and(fc_bin, ~obs).sum()
    cns    = np.logical_and(~fc_bin, ~obs).sum()

    POD = hits / (hits + misses) if (hits + misses) else np.nan
    FAR = fas  / (fas  + cns)    if (fas  + cns)    else np.nan
    Bias = (hits + fas) / (hits + misses) if (hits + misses) else np.nan

    # Brier Score & Skill
    BS      = np.mean((probs - obs) ** 2)
    BS_clim = np.mean((climatology - obs) ** 2)
    BSS     = 1 - BS / BS_clim if BS_clim else np.nan

    # ROC AUC
    try:
        AUC = roc_auc_score(obs, probs)
    except ValueError:
        AUC = np.nan

    records.append({
        "lead_days": lead,
        "POD": POD,
        "FAR": FAR,
        "Bias": Bias,
        "BS": BS,
        "BSS": BSS,
        "AUC": AUC
    })

lead_scores = pd.DataFrame(records).set_index("lead_days")

# 7) Build reliability diagram table across all forecasts
flat_probs = fcst_flat.values
flat_obs   = truth_flat.values.astype(int)

def wilson_ci(k, n, alpha=0.05):
    z = norm.ppf(1 - alpha/2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = z * np.sqrt(phat*(1-phat)/n + z**2/(4*n**2)) / denom
    return center - half, center + half

n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
bin_idx = np.digitize(flat_probs, bins) - 1

rel_stats = []
for k in range(n_bins):
    sel = bin_idx == k
    N = sel.sum()
    if N:
        p_mean = flat_probs[sel].mean()
        o_mean = flat_obs[sel].mean()
        k_obs  = flat_obs[sel].sum()
        lo, hi = wilson_ci(k_obs, N)
    else:
        p_mean = o_mean = lo = hi = np.nan
    rel_stats.append({
        "bin_mid": (bins[k] + bins[k+1]) / 2,
        "count": N,
        "forecast_mean": p_mean,
        "observed_freq": o_mean,
        "lower_CI": lo,
        "upper_CI": hi
    })

reliability_df = pd.DataFrame(rel_stats)

# --- OUTPUT EXAMPLE ---
print("=== Climatology ===")
print(f"Overall event frequency: {climatology:.2%}")
print(monthly_clim)

print("\n=== Lead-Time Scores ===")
print(lead_scores)

print("\n=== Reliability Diagram Table ===")
print(reliability_df)
