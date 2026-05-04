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

LIN_CSV = "aligned_linear_fit_LinPower.csv"
POLY_CSV = "aligned_polynomial_fits_LinPower.csv"

OUT_DIR = "Figures"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG_ABS = os.path.join(OUT_DIR, "Figure_15_absolute_PSD.png")
OUT_TABLE = "PSD_band_integrated_RMS.csv"

# Sampling frequency after synchronization/downsampling.
# The synchronized detector and video-derived signals are compared at 10 Hz.
FS = 10.0  # Hz

# Nyquist frequency is 5 Hz for 10 Hz sampled data.
# Plotting up to 4.5 Hz avoids over-interpreting behavior near the Nyquist limit.
FMAX_PLOT = 4.5

# Representative steady-state intervals for PSD analysis.
psd_intervals = {
    "Low power": (3954, 4232),   # approximately 100 kW
    "Mid power": (2418, 2723),   # approximately 575 kW
    "High power": (850, 1160),   # approximately 900 kW
}

panel_labels = ["(a)", "(b)", "(c)"]

# Frequency bands for band-integrated RMS analysis.
frequency_bands = {
    "0.01-0.1 Hz": (0.01, 0.1),
    "0.1-1 Hz": (0.1, 1.0),
    "1-4.5 Hz": (1.0, 4.5),
    "0.01-4.5 Hz": (0.01, 4.5),
}


# =====================================================
# Helper functions
# =====================================================
def get_segment(t, y, t0, t1):
    """
    Extract a signal segment between t0 and t1.
    NaN and inf values are removed.
    """
    mask = (t >= t0) & (t <= t1)
    t_seg = t[mask]
    y_seg = y[mask]

    valid = np.isfinite(t_seg) & np.isfinite(y_seg)
    return t_seg[valid], y_seg[valid]


def detrend_linear(y):
    """
    Remove a first-order linear trend from a steady-state segment.
    """
    y = np.asarray(y)
    y = y[np.isfinite(y)]

    if len(y) < 2:
        return np.array([])

    return detrend(y, type="linear")


def compute_welch_psd(y, fs, nperseg_max=256):
    """
    Compute Welch's power spectral density estimate.
    """
    y = np.asarray(y)
    y = y[np.isfinite(y)]

    if len(y) < 32:
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


def band_integrated_rms(f, psd, f_low, f_high):
    """
    Compute RMS fluctuation from PSD over a frequency band.

    RMS_band = sqrt(integral PSD(f) df)

    For absolute PSD, the result has units of kW.
    """
    f = np.asarray(f)
    psd = np.asarray(psd)

    mask = (
        np.isfinite(f)
        & np.isfinite(psd)
        & (f >= f_low)
        & (f <= f_high)
    )

    if np.sum(mask) < 2:
        return np.nan

    variance = np.trapz(psd[mask], f[mask])

    if variance < 0:
        return np.nan

    return float(np.sqrt(variance))


# =====================================================
# 1) Load data
# =====================================================
df_lin = pd.read_csv(LIN_CSV)
df_poly3 = pd.read_csv(POLY_CSV)

t = pd.to_numeric(df_lin["t_video_s"], errors="coerce").to_numpy()
nm = pd.to_numeric(df_lin["LinPower_true"], errors="coerce").to_numpy()
poly3 = pd.to_numeric(df_poly3["LinPower_fit_poly3"], errors="coerce").to_numpy()

# Basic consistency check
n = min(len(t), len(nm), len(poly3))
t = t[:n]
nm = nm[:n]
poly3 = poly3[:n]

# Remove invalid rows
valid = np.isfinite(t) & np.isfinite(nm) & np.isfinite(poly3)
t = t[valid]
nm = nm[valid]
poly3 = poly3[valid]

# Convert W to kW for PSD analysis
nm_kW = nm / 1000.0
poly3_kW = poly3 / 1000.0


# =====================================================
# 2) Compute absolute PSDs for selected intervals
# =====================================================
psd_data = {}
band_rows = []

for label, (t0, t1) in psd_intervals.items():
    _, nm_seg = get_segment(t, nm_kW, t0, t1)
    _, p3_seg = get_segment(t, poly3_kW, t0, t1)

    if len(nm_seg) < 32 or len(p3_seg) < 32:
        print(f"Skipping {label}: not enough samples")
        continue

    nm_mean = float(np.mean(nm_seg))
    p3_mean = float(np.mean(p3_seg))

    # Absolute residuals, units: kW
    nm_abs_res = detrend_linear(nm_seg)
    p3_abs_res = detrend_linear(p3_seg)

    f_nm_abs, psd_nm_abs = compute_welch_psd(nm_abs_res, FS)
    f_p3_abs, psd_p3_abs = compute_welch_psd(p3_abs_res, FS)

    if f_nm_abs is None or f_p3_abs is None:
        print(f"Skipping {label}: PSD computation failed")
        continue

    psd_data[label] = {
        "interval": (t0, t1),
        "NM_mean_kW": nm_mean,
        "P3_mean_kW": p3_mean,
        "f_nm_abs": f_nm_abs,
        "psd_nm_abs": psd_nm_abs,
        "f_p3_abs": f_p3_abs,
        "psd_p3_abs": psd_p3_abs,
    }

    # Band-integrated RMS table
    for band_name, (f_low, f_high) in frequency_bands.items():
        nm_abs_rms = band_integrated_rms(f_nm_abs, psd_nm_abs, f_low, f_high)
        p3_abs_rms = band_integrated_rms(f_p3_abs, psd_p3_abs, f_low, f_high)

        band_rows.append({
            "power_regime": label,
            "interval_s": f"{t0}-{t1}",
            "frequency_band": band_name,
            "NM_mean_kW": nm_mean,
            "Cherenkov_mean_kW": p3_mean,
            "NM_absolute_RMS_kW": nm_abs_rms,
            "Cherenkov_absolute_RMS_kW": p3_abs_rms,
            "Cherenkov_to_NM_absolute_RMS_ratio": (
                p3_abs_rms / nm_abs_rms
                if np.isfinite(nm_abs_rms) and abs(nm_abs_rms) > 1e-12
                else np.nan
            ),
        })


# =====================================================
# 3) Plot absolute PSD
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, label, panel_label in zip(axes, psd_intervals.keys(), panel_labels):
    if label not in psd_data:
        ax.set_title(f"{label}\nNo data", fontsize=12)
        ax.axis("off")
        continue

    d = psd_data[label]

    ax.semilogy(
        d["f_nm_abs"],
        d["psd_nm_abs"],
        c="blue",
        label="NM Power"
    )

    ax.semilogy(
        d["f_p3_abs"],
        d["psd_p3_abs"],
        c="red",
        label="Cherenkov-derived power"
    )

    ax.set_xlim(0, FMAX_PLOT)

    # Title remains above each subplot
    ax.set_title("")

    ax.set_xlabel("Frequency (Hz)", fontsize=12)

    # Put full panel label below the x-axis label
    ax.text(
        0.5,
        -0.12,
        f"{panel_label} {label}\nNM: {d['NM_mean_kW']:.0f} kW, Cherenkov: {d['P3_mean_kW']:.0f} kW",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12
    )

    ax.grid(True, alpha=0.3, which="both")

axes[0].set_ylabel("PSD (kW$^2$/Hz)", fontsize=12)
axes[0].legend(fontsize=10)

# Leave enough bottom margin for panel labels below x-axis
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.savefig(OUT_FIG_ABS, dpi=300)
plt.close(fig)

print(f"Saved absolute PSD figure: {OUT_FIG_ABS}")


# =====================================================
# 4) Save band-integrated RMS table
# =====================================================
df_band = pd.DataFrame(band_rows)

regime_order = {
    "Low power": 0,
    "Mid power": 1,
    "High power": 2,
}

band_order = {
    "0.01-0.1 Hz": 0,
    "0.1-1 Hz": 1,
    "1-4.5 Hz": 2,
    "0.01-4.5 Hz": 3,
}

df_band["regime_order"] = df_band["power_regime"].map(regime_order)
df_band["band_order"] = df_band["frequency_band"].map(band_order)

df_band = df_band.sort_values(
    by=["regime_order", "band_order"],
    ascending=True
).drop(columns=["regime_order", "band_order"])

# Round for readability
df_band_rounded = df_band.copy()
for col in df_band_rounded.columns:
    if col not in ["power_regime", "interval_s", "frequency_band"]:
        df_band_rounded[col] = df_band_rounded[col].round(6)

df_band_rounded.to_csv(OUT_TABLE, index=False)

print(f"Saved band-integrated RMS table: {OUT_TABLE}")
print(df_band_rounded)

print("\nAnalysis complete.")