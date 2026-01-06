import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# -----------------------------------------------------------------------------
# User configuration: input/output paths
# -----------------------------------------------------------------------------
csv_path = "2025_12_12.csv"
out_path = "Figures/Figure 4.png"

# Create the output directory if it does not exist
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# -----------------------------------------------------------------------------
# Global plotting style (font and sizes)
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
# Column definitions: adjust as needed for your CSV schema
# -----------------------------------------------------------------------------
cols_power     = ["LinPower"]                           # Reactor power channel
cols_rods      = ["Tran", "Shim1", "Shim2", "Reg"]      # Control rod positions
cols_detectors = ["NM", "NPP", "NP"]                    # Neutron detector channels
cols_temps     = ["FuelTemp1", "FuelTemp2", "WaterTemp"]# Fuel & water temperatures

# List of columns we will try to coerce to numeric (if present in the file)
needed = cols_power + cols_rods + cols_detectors + cols_temps

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
# Read CSV using the first column as a datetime index
df = pd.read_csv(csv_path, index_col=0, parse_dates=True, low_memory=False)

# Convert relevant columns to numeric; invalid values become NaN
for c in [c for c in needed if c in df.columns]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------------------------------------------------------
# Time window and x-axis formatting
# -----------------------------------------------------------------------------
# Define the plotting time span (note: this window crosses midnight)
start = pd.to_datetime("2025-12-11 23:00:00")
end   = pd.to_datetime("2025-12-13 01:00:00")

# Major tick marks every 3 hours; labels as HH:MM
locator = mdates.HourLocator(byhour=range(0, 24, 3))
fmt = mdates.DateFormatter("%H:%M")

# -----------------------------------------------------------------------------
# Create a 2x2 figure layout; share x-axis for consistent time alignment
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 7.5),
    constrained_layout=True,
    sharex=True
)

# -----------------------------------------------------------------------------
# (a) Power (log scale)
# -----------------------------------------------------------------------------
ax = axes[0, 0]

# Use log scale for power; mask non-positive values to avoid log errors
if "LinPower" in df.columns:
    y = df["LinPower"].mask(df["LinPower"] <= 0)
    ax.semilogy(df.index, y, label="Power", color="royalblue")

ax.set_ylabel("Power (W)")
ax.grid(True, which="both", ls="--", alpha=0.6)
ax.legend()

# Panel label placed slightly below the subplot
ax.text(
    0.5, -0.02, "(a) Power (log)",
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=11,
    fontfamily="Times New Roman"
)

# -----------------------------------------------------------------------------
# (b) Control Rods
# -----------------------------------------------------------------------------
ax = axes[0, 1]

# Plot each rod position time series if present
for c in cols_rods:
    if c in df.columns:
        ax.plot(df.index, df[c], label=c, lw=1.2)

ax.set_ylabel("Units")
ax.grid(True, ls="--", alpha=0.6)
ax.legend()

ax.text(
    0.5, -0.02, "(b) Control Rods",
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=11,
    fontfamily="Times New Roman"
)

# -----------------------------------------------------------------------------
# (c) Neutron Detectors
# -----------------------------------------------------------------------------
ax = axes[1, 0]

# Plot detector channels (note: y-label is currently "Power (%)" as in your code;
# change it if your detector signals are in different units)
for c in cols_detectors:
    if c in df.columns:
        ax.plot(df.index, df[c], label=c, lw=1.2)

ax.set_ylabel("Power (%)")
ax.grid(True, ls="--", alpha=0.6)
ax.legend()

# More spacing below because lower row often needs extra room for x tick labels
ax.text(
    0.5, -0.12, "(c) Neutron Detectors",
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=11,
    fontfamily="Times New Roman"
)

# -----------------------------------------------------------------------------
# (d) Fuel/Water Temperature (WaterTemp on the right y-axis)
# -----------------------------------------------------------------------------
ax = axes[1, 1]

# Create a secondary y-axis for WaterTemp so fuel and water scales can differ
twin = ax.twinx()

# Fuel temperatures on left axis
if "FuelTemp1" in df.columns:
    ax.plot(df.index, df["FuelTemp1"], label="Fuel 1", color="firebrick", lw=1.2)
if "FuelTemp2" in df.columns:
    ax.plot(df.index, df["FuelTemp2"], label="Fuel 2", color="orange", lw=1.2)

# Water temperature on right axis
if "WaterTemp" in df.columns:
    twin.plot(df.index, df["WaterTemp"], label="Water", color="royalblue", lw=1.2)

ax.set_ylabel("Fuel Temp (°C)")
twin.set_ylabel("Water Temp (°C)")

# Grid applied to the primary axis only (typically sufficient)
ax.grid(True, ls="--", alpha=0.6)

# Combine legends from both axes into a single legend box
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = twin.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=True)

ax.text(
    0.5, -0.12, "(d) Fuel/Water Temperature",
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=11,
    fontfamily="Times New Roman"
)

# -----------------------------------------------------------------------------
# Shared x-axis settings: apply identical time limits and tick formatting
# -----------------------------------------------------------------------------
for ax in axes.ravel():
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

# -----------------------------------------------------------------------------
# Save and display
# -----------------------------------------------------------------------------
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

plt.show()
