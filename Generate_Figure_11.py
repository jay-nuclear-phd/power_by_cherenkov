import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl

# -----------------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------------
mpl.rcParams["font.family"] = "Times New Roman"

# =====================================================
# Configuration
# =====================================================
FIT_CSV    = "aligned_linear_fit_LinPower.csv"
PIXELS_CSV = "avg_saturated_pixels_by_interval.csv"

# Total pixels in ROI (used to convert counts -> %)
TOTAL_PIXELS = 1416312

OUT_FIG = "Figures/Figure 11.png"

# =====================================================
# Helpers
# =====================================================
def parse_interval_to_seconds(interval_str: str) -> tuple[float, float]:
    """
    Parse an interval string into (start_s, end_s).

    Supported examples:
      "(60,500)" , "(60, 500)" , "(60.0, 500.0)"
      "60-500"   , "60–500"    , "60~500"
    """
    s = str(interval_str).strip()

    # Case 1) Tuple-like: "(60,500)"
    if s.startswith("(") and s.endswith(")"):
        try:
            a, b = ast.literal_eval(s)  # safe parsing for tuples
            return float(a), float(b)
        except Exception:
            pass

    # Case 2) Range-like: "60-500", "60–500", "60~500"
    m = re.search(r"([0-9]*\.?[0-9]+)\s*[-–~]\s*([0-9]*\.?[0-9]+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))

    raise ValueError(f"Unrecognized interval format: {interval_str!r}")


# =====================================================
# 1) Load CSVs
# =====================================================
df_fit = pd.read_csv(FIT_CSV)
df_pixels = pd.read_csv(PIXELS_CSV)

# Defensive checks
required_fit_cols = {"t_video_s", "LinPower_true"}
required_pix_cols = {"Interval", "avg_pixels_gt250"}

missing_fit = required_fit_cols - set(df_fit.columns)
missing_pix = required_pix_cols - set(df_pixels.columns)

if missing_fit:
    raise KeyError(f"Missing columns in {FIT_CSV}: {missing_fit}")
if missing_pix:
    raise KeyError(f"Missing columns in {PIXELS_CSV}: {missing_pix}")

# Ensure numeric
df_fit["t_video_s"] = pd.to_numeric(df_fit["t_video_s"], errors="coerce")
df_fit["LinPower_true"] = pd.to_numeric(df_fit["LinPower_true"], errors="coerce")
df_fit = df_fit.dropna(subset=["t_video_s", "LinPower_true"])

df_pixels["avg_pixels_gt250"] = pd.to_numeric(df_pixels["avg_pixels_gt250"], errors="coerce")

# =====================================================
# 2) Compute NM_mean per interval from aligned_linear_fit_LinPower.csv
# =====================================================
nm_means = []

for interval_str in df_pixels["Interval"]:
    t0, t1 = parse_interval_to_seconds(interval_str)

    # Select the time window (inclusive)
    m = (df_fit["t_video_s"] >= t0) & (df_fit["t_video_s"] <= t1)
    seg = df_fit.loc[m, "LinPower_true"]

    if len(seg) == 0:
        nm_means.append(np.nan)
    else:
        nm_means.append(float(seg.mean()))

df_pixels["NM_mean"] = nm_means

# Optional: drop rows where NM_mean is missing (e.g., interval outside available data)
df_pixels = df_pixels.dropna(subset=["NM_mean", "avg_pixels_gt250"]).copy()

# =====================================================
# 3) Sort by NM_mean (ascending)
# =====================================================
df_sorted = df_pixels.sort_values(by="NM_mean", ascending=True).reset_index(drop=True)

# =====================================================
# 4) Plot (Figure 11)
# =====================================================
plt.figure(figsize=(8, 5))

plt.plot(
    df_sorted["NM_mean"] / 1000,                              # W -> kW
    df_sorted["avg_pixels_gt250"] / TOTAL_PIXELS * 100,       # count -> %
    marker="o",
    linewidth=1.5
)

plt.xlabel("Reactor Power (kW)")
plt.ylabel("Percentage of pixels > 250 (%)")

plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MultipleLocator(100))

plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure: {OUT_FIG}")
