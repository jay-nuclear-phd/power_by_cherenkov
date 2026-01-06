import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------
csv_path = "2025_12_15.csv"
out_path = "Figures/Figure 3.png"

# Ensure the output directory exists (e.g., "Figures/")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# -----------------------------------------------------------------------------
# Global plot style (fonts and sizes)
# -----------------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

# -----------------------------------------------------------------------------
# Column definitions
# -----------------------------------------------------------------------------
# Control rod position columns (units typically in steps or similar)
cols_rods = ["Tran", "Shim1", "Shim2", "Reg"]

# Neutron detector channels
cols_detectors = ["NM", "NPP", "NP"]

# -----------------------------------------------------------------------------
# Data loading and basic cleaning
# -----------------------------------------------------------------------------
# Read CSV with the first column as datetime index
df = pd.read_csv(csv_path, index_col=0, parse_dates=True, low_memory=False)

# Convert rod and detector columns to numeric where possible
# (invalid entries become NaN)
for c in cols_rods + cols_detectors:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------------------------------------------------------
# Time window and x-axis formatting
# -----------------------------------------------------------------------------
start = pd.to_datetime("2025-12-15 10:06:43")
end   = pd.to_datetime("2025-12-15 10:44:30")

# Major ticks every 5 minutes, formatted as HH:MM
locator = mdates.MinuteLocator(interval=5)
fmt = mdates.DateFormatter("%H:%M")

# -----------------------------------------------------------------------------
# Create the figure (1 row, 2 columns)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5),
    constrained_layout=True,
    sharex=True
)

# -----------------------------------------------------------------------------
# (a) Control Rods
# -----------------------------------------------------------------------------
ax = axes[0]

# Plot each control rod time series if the column exists
for c in cols_rods:
    if c in df.columns:
        ax.plot(df.index, df[c], label=c, lw=1.2)

ax.set_ylabel("Units")
ax.grid(True, ls="--", alpha=0.6)
ax.legend(loc="upper right")

# Panel label placed below the axes area
ax.text(
    0.5, -0.1, "(a) Control Rods",
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=11
)

# -----------------------------------------------------------------------------
# (b) Neutron Detectors (scale factor applied)
# -----------------------------------------------------------------------------
ax = axes[1]

# Apply a scale factor (×10) to convert detector readings to kW
# (Adjust this factor if your calibration changes.)
scale_factor = 10

for c in cols_detectors:
    if c in df.columns:
        ax.plot(df.index, df[c] * scale_factor, label=c, lw=1.2)

ax.set_ylabel("Power (kW)")
ax.grid(True, ls="--", alpha=0.6)
ax.legend()

# Set only the upper bound (Matplotlib chooses the lower bound automatically)
# Note: 850 kW here corresponds to an original unscaled upper ~85 if scale_factor=10.
ax.set_ylim(None, 850)

ax.text(
    0.5, -0.1, "(b) Neutron Detectors",
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=11
)

# -----------------------------------------------------------------------------
# Shared x-axis formatting for both subplots
# -----------------------------------------------------------------------------
for ax in axes:
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

# -----------------------------------------------------------------------------
# Save and show
# -----------------------------------------------------------------------------
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {out_path}")
