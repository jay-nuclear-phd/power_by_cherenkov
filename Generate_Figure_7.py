import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os

# -----------------------------------------------------------------------------
# Global style: use Times New Roman for all text in the figure
# -----------------------------------------------------------------------------
mpl.rcParams["font.family"] = "Times New Roman"

# -----------------------------------------------------------------------------
# Input CSV file paths (relative to the current working directory)
# - If the CSV files are in the same folder where you run this script,
#   these simple filenames are sufficient.
# -----------------------------------------------------------------------------
csv_path_r = "mean_red.csv"
csv_path_g = "mean_green.csv"
csv_path_b = "mean_blue.csv"

# -----------------------------------------------------------------------------
# Load CSV files
# Assumption: each CSV has at least two columns:
#   col0 = time in seconds, col1 = mean intensity
# -----------------------------------------------------------------------------
df_r = pd.read_csv(csv_path_r)
df_g = pd.read_csv(csv_path_g)
df_b = pd.read_csv(csv_path_b)

# -----------------------------------------------------------------------------
# Define the common x-axis (time in seconds)
# We assume all three files share the same time column structure.
# -----------------------------------------------------------------------------
x = df_r.iloc[:, 0]

# -----------------------------------------------------------------------------
# Select a time window (seconds) to plot
# This mask trims the plotted data to a specific segment of the video.
# -----------------------------------------------------------------------------
t_start = 60
t_end = 4460
mask = (x >= t_start) & (x <= t_end)

# -----------------------------------------------------------------------------
# Formatter: convert seconds -> "MM:SS" for readability on the x-axis
# -----------------------------------------------------------------------------
def sec_to_mmss(val, pos):
    """
    Convert seconds (val) into MM:SS format for x-axis tick labels.
    Matplotlib passes 'val' as a float; we cast to int for clean formatting.
    """
    val = int(val)
    if val < 0:
        return ""
    m = val // 60
    s = val % 60
    return f"{m:02d}:{s:02d}"

# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 4))
ax = plt.gca()

# Plot each color channel (mean intensity over time)
ax.plot(x[mask], df_r.iloc[:, 1][mask], color="red",   label="Red")
ax.plot(x[mask], df_g.iloc[:, 1][mask], color="green", label="Green")
ax.plot(x[mask], df_b.iloc[:, 1][mask], color="blue",  label="Blue")

# Axis labels and limits
ax.set_xlabel("Video time (MM:SS)")
ax.set_ylabel("Mean Intensity")

# Keep your original x-limits (note: left bound is negative to add margin)
ax.set_xlim([-50, 4450])

# Optional y-limits (uncomment if you want fixed scaling)
# ax.set_ylim([45, 60])

# Improve readability
ax.legend()
ax.grid(True)

# -----------------------------------------------------------------------------
# X-axis ticks: format as MM:SS and show ticks every 600 seconds (10 minutes)
# -----------------------------------------------------------------------------
ax.xaxis.set_major_formatter(FuncFormatter(sec_to_mmss))
ax.xaxis.set_major_locator(MultipleLocator(600))

# Use tight_layout to reduce overlap and improve spacing
plt.tight_layout()

# -----------------------------------------------------------------------------
# Save the figure (300 dpi) into the Figures/ directory
# -----------------------------------------------------------------------------
out_path = os.path.join("Figures", "Figure 7.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

plt.savefig(out_path, dpi=300)
plt.show()

print(f"Saved: {out_path}")
