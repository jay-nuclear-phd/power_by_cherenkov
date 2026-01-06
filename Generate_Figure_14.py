import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =====================================================
# Configuration
# =====================================================
mpl.rcParams["font.family"] = "Times New Roman"

LIN_CSV  = "aligned_linear_fit_LinPower.csv"
POLY_CSV = "aligned_polynomial_fits_LinPower.csv"   # or use your combined poly CSV if available
OUT_FIG  = "Figures/Figure 14.png"

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
            "poly3_mean": np.nan,
            "poly3_std": np.nan,
            "N_samples": 0
        })
        continue

    results.append({
        "interval": f"{t0}-{t1}",
        "NM_mean": float(np.mean(nm_seg)),
        "NM_std": float(np.std(nm_seg)),
        "poly3_mean": float(np.mean(p3_seg)),
        "poly3_std": float(np.std(p3_seg)),
        "N_samples": int(nm_seg.size)
    })

df_results = pd.DataFrame(results)


# =====================================================
# 3) Plot: Mean vs Std (kW)
# =====================================================
# Drop intervals with NaN (if any)
dfp = df_results.dropna(subset=["NM_mean", "NM_std", "poly3_mean", "poly3_std"]).copy()

NM_mean = dfp["NM_mean"].to_numpy() / 1000.0
NM_std  = dfp["NM_std"].to_numpy() / 1000.0
P3_mean = dfp["poly3_mean"].to_numpy() / 1000.0
P3_std  = dfp["poly3_std"].to_numpy() / 1000.0

plt.figure(figsize=(7, 5))

plt.scatter(NM_mean, NM_std, c="blue", label="NM Power", s=20)
plt.scatter(P3_mean, P3_std, c="red",  label="Synchronized blue intensity (poly3)", s=20)

plt.xlabel("Mean Power (kW)", fontsize=12)
plt.ylabel("Standard Deviation (kW)", fontsize=12)

plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.show()

print(f"Saved figure: {OUT_FIG}")
