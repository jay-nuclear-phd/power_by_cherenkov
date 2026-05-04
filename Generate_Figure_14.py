import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =====================================================
# Configuration
# =====================================================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"

LIN_CSV  = "aligned_linear_fit_LinPower.csv"
POLY_CSV = "aligned_polynomial_fits_LinPower.csv"

OUT_DIR = "Figures"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG = os.path.join(OUT_DIR, "Figure 14.png")

# Save Table 3 in the current working directory, not in Figures
OUT_TABLE = "Table3.csv"

# Intervals to analyze (seconds)
intervals = [
    (60, 500),
    (850, 1160),
    (1167, 1470),
    (1490, 1777),
    (1790, 2078),
    (2095, 2389),
    (2418, 2723),
    (2747, 3034),
    (3051, 3331),
    (3355, 3627),
    (3658, 3929),
    (3954, 4232),
]

# =====================================================
# 1) Load data
# =====================================================
df_lin = pd.read_csv(LIN_CSV)
df_poly3 = pd.read_csv(POLY_CSV)

# Defensive numeric conversion
t = pd.to_numeric(df_lin["t_video_s"], errors="coerce").to_numpy()
nm = pd.to_numeric(df_lin["LinPower_true"], errors="coerce").to_numpy()
poly3 = pd.to_numeric(df_poly3["LinPower_fit_poly3"], errors="coerce").to_numpy()

# Basic consistency check
n = min(len(t), len(nm), len(poly3))
t, nm, poly3 = t[:n], nm[:n], poly3[:n]

# Remove invalid rows
valid = np.isfinite(t) & np.isfinite(nm) & np.isfinite(poly3)
t, nm, poly3 = t[valid], nm[valid], poly3[valid]

# =====================================================
# 2) Compute interval statistics
# =====================================================
results = []

for (t0, t1) in intervals:
    mask = (t >= t0) & (t <= t1)

    nm_seg = nm[mask]
    p3_seg = poly3[mask]

    # Skip empty intervals safely
    if nm_seg.size == 0 or p3_seg.size == 0:
        results.append({
            "interval": f"{t0}-{t1}",
            "NM_mean": np.nan,
            "NM_std": np.nan,
            "NM_relative_noise_percent": np.nan,
            "poly3_mean": np.nan,
            "poly3_std": np.nan,
            "poly3_relative_noise_percent": np.nan,
            "N_samples": 0
        })
        continue

    nm_mean = float(np.mean(nm_seg))
    nm_std = float(np.std(nm_seg))

    p3_mean = float(np.mean(p3_seg))
    p3_std = float(np.std(p3_seg))

    # Relative noise = STD / mean * 100
    # If the mean is too close to zero, relative noise is not meaningful.
    eps = 1e-12

    if abs(nm_mean) > eps:
        nm_rel = 100.0 * nm_std / abs(nm_mean)
    else:
        nm_rel = np.nan

    if abs(p3_mean) > eps:
        p3_rel = 100.0 * p3_std / abs(p3_mean)
    else:
        p3_rel = np.nan

    results.append({
        "interval": f"{t0}-{t1}",
        "NM_mean": nm_mean,
        "NM_std": nm_std,
        "NM_relative_noise_percent": nm_rel,
        "poly3_mean": p3_mean,
        "poly3_std": p3_std,
        "poly3_relative_noise_percent": p3_rel,
        "N_samples": int(min(nm_seg.size, p3_seg.size))
    })

df_results = pd.DataFrame(results)

# =====================================================
# 3) Convert units and save revised Table 3
# =====================================================
dfp = df_results.dropna(subset=[
    "NM_mean",
    "NM_std",
    "poly3_mean",
    "poly3_std"
]).copy()

# Original data are assumed to be in W, so convert to kW.
dfp["NM_mean_kW"] = dfp["NM_mean"] / 1000.0
dfp["NM_std_kW"] = dfp["NM_std"] / 1000.0
dfp["poly3_mean_kW"] = dfp["poly3_mean"] / 1000.0
dfp["poly3_std_kW"] = dfp["poly3_std"] / 1000.0

# Reorder table columns for manuscript use
table3 = dfp[[
    "interval",
    "NM_mean_kW",
    "NM_std_kW",
    "NM_relative_noise_percent",
    "poly3_mean_kW",
    "poly3_std_kW",
    "poly3_relative_noise_percent",
    "N_samples"
]].copy()

# Optional rounding for readability
table3_rounded = table3.copy()
for col in table3_rounded.columns:
    if col not in ["interval", "N_samples"]:
        table3_rounded[col] = table3_rounded[col].round(4)

# Sort by mean power before saving
table3_rounded = table3_rounded.sort_values(
    by="NM_mean_kW",
    ascending=True
).reset_index(drop=True)

table3_rounded.to_csv(OUT_TABLE, index=False)

print(f"Saved revised noise table: {OUT_TABLE}")
print(table3_rounded)

# =====================================================
# 4) Plot: Absolute noise and relative noise
# =====================================================
NM_mean = dfp["NM_mean_kW"].to_numpy()
NM_std = dfp["NM_std_kW"].to_numpy()
P3_mean = dfp["poly3_mean_kW"].to_numpy()
P3_std = dfp["poly3_std_kW"].to_numpy()

NM_rel = dfp["NM_relative_noise_percent"].to_numpy()
P3_rel = dfp["poly3_relative_noise_percent"].to_numpy()

# Exclude near-zero-power interval from relative noise plot.
# Relative noise becomes ill-conditioned when mean power approaches zero.
valid_rel = (
    np.isfinite(NM_rel)
    & np.isfinite(P3_rel)
    & (NM_mean > 1.0)
    & (P3_mean > 1.0)
)

# Use the same x-axis limits for both subplots.
# This makes panel (b) use the same mean-power scale as panel (a).
all_x = np.concatenate([NM_mean[np.isfinite(NM_mean)], P3_mean[np.isfinite(P3_mean)]])
x_min = -50
x_max = np.nanmax(all_x) * 1.05

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# -----------------------------------------------------
# (a) Absolute noise
# -----------------------------------------------------
axes[0].scatter(
    NM_mean,
    NM_std,
    c="blue",
    label="NM Power",
    s=25
)

axes[0].scatter(
    P3_mean,
    P3_std,
    c="red",
    label="Cherenkov-derived power",
    s=25
)

axes[0].set_xlabel("Mean Power (kW)", fontsize=12)
axes[0].set_ylabel("Standard Deviation (kW)", fontsize=12)
axes[0].set_xlim(x_min, x_max)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Panel label below the graph
axes[0].text(
    0.5,
    -0.15,
    "(a) Absolute fluctuation",
    transform=axes[0].transAxes,
    ha="center",
    va="top",
    fontsize=12
)

# -----------------------------------------------------
# (b) Relative noise
# -----------------------------------------------------
axes[1].scatter(
    NM_mean[valid_rel],
    NM_rel[valid_rel],
    c="blue",
    label="NM Power",
    s=25
)

axes[1].scatter(
    P3_mean[valid_rel],
    P3_rel[valid_rel],
    c="red",
    label="Cherenkov-derived power",
    s=25
)

axes[1].set_xlabel("Mean Power (kW)", fontsize=12)
axes[1].set_ylabel("Relative Noise (%)", fontsize=12)
axes[1].set_xlim(x_min, x_max)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

# Panel label below the graph
axes[1].text(
    0.5,
    -0.15,
    "(b) Relative fluctuation",
    transform=axes[1].transAxes,
    ha="center",
    va="top",
    fontsize=12
)

# Leave enough bottom margin for panel labels
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.savefig(OUT_FIG, dpi=300)
# plt.show()

print(f"Saved figure: {OUT_FIG}")