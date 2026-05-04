import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import welch, detrend

# =====================================================
# Configuration
# =====================================================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"

LIN_CSV  = "aligned_linear_fit_LinPower.csv"
POLY_CSV = "aligned_polynomial_fits_LinPower.csv"

OUT_DIR = "Figures"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG_ABS_REL = os.path.join(OUT_DIR, "Figure_14_absolute_relative_noise.png")
OUT_FIG_PSD_ABS = os.path.join(OUT_DIR, "Figure_15_absolute_PSD_noise_analysis.png")
OUT_FIG_PSD_NORM = os.path.join(OUT_DIR, "Figure_15_normalized_PSD_noise_analysis.png")

OUT_TABLE_NOISE = os.path.join(OUT_DIR, "Table_3_revised_noise_statistics.csv")
OUT_TABLE_DETRENDED = os.path.join(OUT_DIR, "Table_3_detrended_noise_statistics.csv")

# Sampling frequency after synchronization/downsampling
# Manuscript: video and reactor data are synchronized at 10 Hz
FS = 10.0  # Hz

# Intervals to analyze, in seconds
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

# Representative intervals for PSD analysis
# Adjust these if your actual plateau order is different.
psd_intervals = {
    "Low power": (3954, 4232),   # approximately 100 kW
    "Mid power": (2418, 2723),   # approximately 575 kW
    "High power": (850, 1160),   # approximately 900 kW
}

# =====================================================
# Helper functions
# =====================================================
def safe_std(x):
    """
    Population standard deviation, same as np.std(x).
    Use ddof=1 if you prefer sample standard deviation.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.std(x))


def safe_mean(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(x))


def safe_cv_percent(mean_value, std_value, eps=1e-12):
    """
    Coefficient of variation in percent.
    Returns NaN when mean is too close to zero.
    """
    if not np.isfinite(mean_value) or not np.isfinite(std_value):
        return np.nan
    if abs(mean_value) < eps:
        return np.nan
    return 100.0 * std_value / abs(mean_value)


def get_segment(t, y, t0, t1):
    """
    Extract time and signal segment between t0 and t1.
    Removes NaNs and infs.
    """
    mask = (t >= t0) & (t <= t1)
    t_seg = t[mask]
    y_seg = y[mask]

    valid = np.isfinite(t_seg) & np.isfinite(y_seg)
    return t_seg[valid], y_seg[valid]


def compute_detrended_residual(y):
    """
    Linear detrending for steady-state plateau.
    Returns residual signal.
    """
    y = np.asarray(y)
    y = y[np.isfinite(y)]

    if y.size < 2:
        return np.full_like(y, np.nan)

    return detrend(y, type="linear")


def compute_welch_psd(y, fs, nperseg_max=256):
    """
    Compute Welch PSD.
    Assumes y is already detrended or processed as desired.
    """
    y = np.asarray(y)
    y = y[np.isfinite(y)]

    if y.size < 32:
        return None, None

    nperseg = min(nperseg_max, len(y))

    f, pxx = welch(
        y,
        fs=fs,
        nperseg=nperseg,
        detrend=False,
        scaling="density"
    )

    return f, pxx


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
t = t[:n]
nm = nm[:n]
poly3 = poly3[:n]

# Remove rows where time is invalid
valid_time = np.isfinite(t)
t = t[valid_time]
nm = nm[valid_time]
poly3 = poly3[valid_time]

# =====================================================
# 2) Compute interval statistics
# =====================================================
results = []

for (t0, t1) in intervals:
    _, nm_seg = get_segment(t, nm, t0, t1)
    _, p3_seg = get_segment(t, poly3, t0, t1)

    if nm_seg.size == 0 or p3_seg.size == 0:
        results.append({
            "interval_s": f"{t0}-{t1}",
            "t0_s": t0,
            "t1_s": t1,
            "NM_mean_W": np.nan,
            "NM_std_W": np.nan,
            "NM_relative_noise_percent": np.nan,
            "poly3_mean_W": np.nan,
            "poly3_std_W": np.nan,
            "poly3_relative_noise_percent": np.nan,
            "N_samples": 0
        })
        continue

    nm_mean = safe_mean(nm_seg)
    nm_std = safe_std(nm_seg)

    p3_mean = safe_mean(p3_seg)
    p3_std = safe_std(p3_seg)

    results.append({
        "interval_s": f"{t0}-{t1}",
        "t0_s": t0,
        "t1_s": t1,
        "NM_mean_W": nm_mean,
        "NM_std_W": nm_std,
        "NM_relative_noise_percent": safe_cv_percent(nm_mean, nm_std),
        "poly3_mean_W": p3_mean,
        "poly3_std_W": p3_std,
        "poly3_relative_noise_percent": safe_cv_percent(p3_mean, p3_std),
        "N_samples": int(min(nm_seg.size, p3_seg.size))
    })

df_results = pd.DataFrame(results)

# Convert to kW for plotting and manuscript table
df_results["NM_mean_kW"] = df_results["NM_mean_W"] / 1000.0
df_results["NM_std_kW"] = df_results["NM_std_W"] / 1000.0
df_results["poly3_mean_kW"] = df_results["poly3_mean_W"] / 1000.0
df_results["poly3_std_kW"] = df_results["poly3_std_W"] / 1000.0

# Save revised Table 3
table3 = df_results[[
    "interval_s",
    "NM_mean_kW",
    "NM_std_kW",
    "NM_relative_noise_percent",
    "poly3_mean_kW",
    "poly3_std_kW",
    "poly3_relative_noise_percent",
    "N_samples"
]].copy()

table3_rounded = table3.copy()
for col in table3_rounded.columns:
    if col not in ["interval_s", "N_samples"]:
        table3_rounded[col] = table3_rounded[col].round(4)

table3_rounded.to_csv(OUT_TABLE_NOISE, index=False)

print(f"Saved revised noise table: {OUT_TABLE_NOISE}")
print(table3_rounded)

# =====================================================
# 3) Compute detrended residual noise statistics
# =====================================================
detrended_results = []

for (t0, t1) in intervals:
    _, nm_seg = get_segment(t, nm, t0, t1)
    _, p3_seg = get_segment(t, poly3, t0, t1)

    if nm_seg.size < 2 or p3_seg.size < 2:
        detrended_results.append({
            "interval_s": f"{t0}-{t1}",
            "t0_s": t0,
            "t1_s": t1,
            "NM_mean_kW": np.nan,
            "NM_residual_std_kW": np.nan,
            "NM_residual_rms_kW": np.nan,
            "NM_residual_relative_std_percent": np.nan,
            "poly3_mean_kW": np.nan,
            "poly3_residual_std_kW": np.nan,
            "poly3_residual_rms_kW": np.nan,
            "poly3_residual_relative_std_percent": np.nan,
            "N_samples": 0
        })
        continue

    # Convert to kW before detrending
    nm_seg_kW = nm_seg / 1000.0
    p3_seg_kW = p3_seg / 1000.0

    nm_mean_kW = safe_mean(nm_seg_kW)
    p3_mean_kW = safe_mean(p3_seg_kW)

    nm_res = compute_detrended_residual(nm_seg_kW)
    p3_res = compute_detrended_residual(p3_seg_kW)

    nm_res_std = safe_std(nm_res)
    p3_res_std = safe_std(p3_res)

    nm_res_rms = float(np.sqrt(np.mean(nm_res[np.isfinite(nm_res)] ** 2)))
    p3_res_rms = float(np.sqrt(np.mean(p3_res[np.isfinite(p3_res)] ** 2)))

    detrended_results.append({
        "interval_s": f"{t0}-{t1}",
        "t0_s": t0,
        "t1_s": t1,
        "NM_mean_kW": nm_mean_kW,
        "NM_residual_std_kW": nm_res_std,
        "NM_residual_rms_kW": nm_res_rms,
        "NM_residual_relative_std_percent": safe_cv_percent(nm_mean_kW, nm_res_std),
        "poly3_mean_kW": p3_mean_kW,
        "poly3_residual_std_kW": p3_res_std,
        "poly3_residual_rms_kW": p3_res_rms,
        "poly3_residual_relative_std_percent": safe_cv_percent(p3_mean_kW, p3_res_std),
        "N_samples": int(min(nm_seg.size, p3_seg.size))
    })

df_detrended = pd.DataFrame(detrended_results)

df_detrended_rounded = df_detrended.copy()
for col in df_detrended_rounded.columns:
    if col not in ["interval_s", "N_samples"]:
        df_detrended_rounded[col] = df_detrended_rounded[col].round(4)

df_detrended_rounded.to_csv(OUT_TABLE_DETRENDED, index=False)

print(f"Saved detrended noise table: {OUT_TABLE_DETRENDED}")
print(df_detrended_rounded)

# =====================================================
# 4) Figure 14: Absolute and relative noise
# =====================================================
dfp = df_results.dropna(subset=[
    "NM_mean_kW",
    "NM_std_kW",
    "poly3_mean_kW",
    "poly3_std_kW"
]).copy()

NM_mean = dfp["NM_mean_kW"].to_numpy()
NM_std = dfp["NM_std_kW"].to_numpy()
P3_mean = dfp["poly3_mean_kW"].to_numpy()
P3_std = dfp["poly3_std_kW"].to_numpy()

NM_rel_noise = dfp["NM_relative_noise_percent"].to_numpy()
P3_rel_noise = dfp["poly3_relative_noise_percent"].to_numpy()

# Exclude near-zero mean values for relative noise plot
# CV is ill-conditioned near zero power.
valid_rel = (
    np.isfinite(NM_rel_noise) &
    np.isfinite(P3_rel_noise) &
    (NM_mean > 1.0) &
    (P3_mean > 1.0)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# -----------------------------
# Figure 14(a): Absolute noise
# -----------------------------
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
axes[0].set_title("(a) Absolute fluctuation", fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# -----------------------------
# Figure 14(b): Relative noise
# -----------------------------
axes[1].scatter(
    NM_mean[valid_rel],
    NM_rel_noise[valid_rel],
    c="blue",
    label="NM Power",
    s=25
)

axes[1].scatter(
    P3_mean[valid_rel],
    P3_rel_noise[valid_rel],
    c="red",
    label="Cherenkov-derived power",
    s=25
)

axes[1].set_xlabel("Mean Power (kW)", fontsize=12)
axes[1].set_ylabel("Relative Noise (%)", fontsize=12)
axes[1].set_title("(b) Relative fluctuation", fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT_FIG_ABS_REL, dpi=300)
plt.show()

print(f"Saved figure: {OUT_FIG_ABS_REL}")

# =====================================================
# 5) Figure 15: Absolute PSD analysis
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (label, (t0, t1)) in zip(axes, psd_intervals.items()):
    _, nm_seg = get_segment(t, nm, t0, t1)
    _, p3_seg = get_segment(t, poly3, t0, t1)

    # Convert to kW
    nm_seg = nm_seg / 1000.0
    p3_seg = p3_seg / 1000.0

    if len(nm_seg) < 32 or len(p3_seg) < 32:
        ax.set_title(f"{label}\nNot enough samples", fontsize=12)
        ax.axis("off")
        continue

    # Detrend to focus on fluctuation rather than slow drift
    nm_res = compute_detrended_residual(nm_seg)
    p3_res = compute_detrended_residual(p3_seg)

    f_nm, psd_nm = compute_welch_psd(nm_res, FS)
    f_p3, psd_p3 = compute_welch_psd(p3_res, FS)

    if f_nm is None or f_p3 is None:
        ax.set_title(f"{label}\nPSD failed", fontsize=12)
        ax.axis("off")
        continue

    ax.semilogy(
        f_nm,
        psd_nm,
        c="blue",
        label="NM Power"
    )

    ax.semilogy(
        f_p3,
        psd_p3,
        c="red",
        label="Cherenkov-derived power"
    )

    nm_mean_label = np.mean(nm_seg)
    p3_mean_label = np.mean(p3_seg)

    ax.set_title(
        f"{label}\nNM: {nm_mean_label:.0f} kW, Cherenkov: {p3_mean_label:.0f} kW",
        fontsize=11
    )
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("PSD (kW$^2$/Hz)", fontsize=12)
axes[0].legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT_FIG_PSD_ABS, dpi=300)
plt.show()

print(f"Saved absolute PSD figure: {OUT_FIG_PSD_ABS}")

# =====================================================
# 6) Figure 15 alternative: Normalized PSD analysis
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (label, (t0, t1)) in zip(axes, psd_intervals.items()):
    _, nm_seg = get_segment(t, nm, t0, t1)
    _, p3_seg = get_segment(t, poly3, t0, t1)

    # Convert to kW
    nm_seg = nm_seg / 1000.0
    p3_seg = p3_seg / 1000.0

    if len(nm_seg) < 32 or len(p3_seg) < 32:
        ax.set_title(f"{label}\nNot enough samples", fontsize=12)
        ax.axis("off")
        continue

    nm_mean_seg = np.mean(nm_seg)
    p3_mean_seg = np.mean(p3_seg)

    if abs(nm_mean_seg) < 1e-12 or abs(p3_mean_seg) < 1e-12:
        ax.set_title(f"{label}\nNear-zero mean", fontsize=12)
        ax.axis("off")
        continue

    # Detrended residual
    nm_res = compute_detrended_residual(nm_seg)
    p3_res = compute_detrended_residual(p3_seg)

    # Normalize residual by mean power
    nm_norm_res = nm_res / nm_mean_seg
    p3_norm_res = p3_res / p3_mean_seg

    f_nm, psd_nm = compute_welch_psd(nm_norm_res, FS)
    f_p3, psd_p3 = compute_welch_psd(p3_norm_res, FS)

    if f_nm is None or f_p3 is None:
        ax.set_title(f"{label}\nPSD failed", fontsize=12)
        ax.axis("off")
        continue

    ax.semilogy(
        f_nm,
        psd_nm,
        c="blue",
        label="NM Power"
    )

    ax.semilogy(
        f_p3,
        psd_p3,
        c="red",
        label="Cherenkov-derived power"
    )

    ax.set_title(
        f"{label}\nNM: {nm_mean_seg:.0f} kW, Cherenkov: {p3_mean_seg:.0f} kW",
        fontsize=11
    )
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Normalized PSD (1/Hz)", fontsize=12)
axes[0].legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT_FIG_PSD_NORM, dpi=300)
plt.show()

print(f"Saved normalized PSD figure: {OUT_FIG_PSD_NORM}")

# =====================================================
# 7) Quick summary printed to console
# =====================================================
print("\n=====================================================")
print("Analysis complete.")
print("Generated files:")
print(f"1) {OUT_FIG_ABS_REL}")
print(f"2) {OUT_FIG_PSD_ABS}")
print(f"3) {OUT_FIG_PSD_NORM}")
print(f"4) {OUT_TABLE_NOISE}")
print(f"5) {OUT_TABLE_DETRENDED}")
print("=====================================================")