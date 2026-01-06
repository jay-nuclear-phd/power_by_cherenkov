import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

# -----------------------------------------------------------------------------
# Global style: use Times New Roman for all text in the figure
# -----------------------------------------------------------------------------
mpl.rcParams["font.family"] = "Times New Roman"

# -----------------------------------------------------------------------------
# Input CSV paths
# Assumption: each CSV has at least two columns:
#   col0 = time in seconds, col1 = mean intensity
# -----------------------------------------------------------------------------
csv_org = r"mean_blue.csv"
csv_dn  = r"mean_blue_denoised.csv"

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df_org = pd.read_csv(csv_org)
df_dn  = pd.read_csv(csv_dn)

# Extract time columns (seconds) for each dataset
x_org = df_org.iloc[:, 0]
x_dn  = df_dn.iloc[:, 0]

# -----------------------------------------------------------------------------
# Define the plotting time window (same as your Figure 5 intent)
# Here: from 60 seconds to 75 minutes (75*60 seconds)
# -----------------------------------------------------------------------------
t_start = 60
t_end = 75 * 60
mask_org = (x_org >= t_start) & (x_org <= t_end)
mask_dn  = (x_dn  >= t_start) & (x_dn  <= t_end)

# -----------------------------------------------------------------------------
# Formatter: convert seconds -> "MM:SS" for x-axis tick labels
# -----------------------------------------------------------------------------
def sec_to_mmss(val, pos):
    """
    Convert seconds (val) into MM:SS format.
    Matplotlib provides tick values as float; cast to int for clean formatting.
    """
    val = int(val)
    m = val // 60
    s = val % 60
    return f"{m:02d}:{s:02d}"

# -----------------------------------------------------------------------------
# Create Figure 8 (main plot + two zoomed inset panels)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))

# Use a thin line width because these signals can be dense/noisy
lw_main = 0.3

# Plot original and denoised blue-channel mean intensity
ax.plot(
    x_org[mask_org], df_org.iloc[:, 1][mask_org],
    label="Blue (original)", color="blue", linewidth=lw_main
)
ax.plot(
    x_dn[mask_dn], df_dn.iloc[:, 1][mask_dn],
    label="Blue (denoised)", color="red", linewidth=lw_main
)

# Main axis labels, limits, and styling
ax.set_xlabel("Video time (MM:SS)")
ax.set_ylabel("Mean Intensity")
ax.set_xlim([-50, 85 * 60])   # keep your margin and extended end to 85 min
ax.set_ylim([35, 105])
ax.legend()
ax.grid(True)

# -----------------------------------------------------------------------------
# X-axis tick formatting: show major ticks every 10 minutes (600 seconds)
# -----------------------------------------------------------------------------
formatter = FuncFormatter(sec_to_mmss)
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(MultipleLocator(600))

# -----------------------------------------------------------------------------
# Zoom-in regions (two separate windows)
# You define the zoom rectangles in terms of x-range and y-range:
#   Window 1: x1~x2 and y1~y2
#   Window 2: x3~x4 and y3~y4
# -----------------------------------------------------------------------------
x1, x2 = 1222, 1538      # approx. 20:22 to 25:38 (seconds)
y1, y2 = 90, 93          # intensity range for zoom 1

x3, x4 = 3695, 4002      # approx. 61:35 to 66:42 (seconds)
y3, y4 = 57.5, 60.5      # intensity range for zoom 2

# -----------------------------------------------------------------------------
# Draw the zoom rectangles on the main plot
# These visually indicate which regions are magnified in the inset axes.
# -----------------------------------------------------------------------------
rect1_x = [x1, x2, x2, x1, x1]
rect1_y = [y1, y1, y2, y2, y1]
rect2_x = [x3, x4, x4, x3, x3]
rect2_y = [y3, y3, y4, y4, y3]

ax.plot(rect1_x, rect1_y, color="red", linewidth=0.8)
ax.plot(rect2_x, rect2_y, color="red", linewidth=0.8)

# -----------------------------------------------------------------------------
# Create inset axes
# inset_axes allows flexible placement using bbox_to_anchor in axes coordinates.
#
# Notes:
# - bbox_to_anchor uses a rectangle (x0, y0, width, height) in the coordinate
#   system given by bbox_transform.
# - Here, bbox_transform=ax.transAxes makes (0,0) the lower-left and (1,1)
#   the upper-right of the main axes.
# -----------------------------------------------------------------------------
inset1 = inset_axes(
    ax,
    width="35%",
    height="40%",
    bbox_to_anchor=(-0.4, -0.55, 1, 1),  # custom placement (left/bottom outside)
    bbox_transform=ax.transAxes
)

inset2 = inset_axes(
    ax,
    width="35%",
    height="40%",
    bbox_to_anchor=(0.0, 0.0, 1, 1),     # custom placement (near right-middle)
    bbox_transform=ax.transAxes
)

# -----------------------------------------------------------------------------
# Masks for zoom windows (for both original and denoised signals)
# -----------------------------------------------------------------------------
mask_zoom1_org = (x_org >= x1) & (x_org <= x2)
mask_zoom1_dn  = (x_dn  >= x1) & (x_dn  <= x2)

mask_zoom2_org = (x_org >= x3) & (x_org <= x4)
mask_zoom2_dn  = (x_dn  >= x3) & (x_dn  <= x4)

# -----------------------------------------------------------------------------
# Plot zoomed curves in inset 1
# Use slightly thicker lines than the main plot for visibility
# -----------------------------------------------------------------------------
lw_inset = 0.6

inset1.plot(x_org[mask_zoom1_org], df_org.iloc[:, 1][mask_zoom1_org],
            color="blue", linewidth=lw_inset)
inset1.plot(x_dn[mask_zoom1_dn], df_dn.iloc[:, 1][mask_zoom1_dn],
            color="red", linewidth=lw_inset)

inset1.set_xlim(x1, x2)
inset1.set_ylim(y1, y2)
inset1.grid(True, linewidth=0.3)

# Apply the same MM:SS formatter, but hide tick labels to reduce clutter
inset1.xaxis.set_major_formatter(formatter)
inset1.set_xticklabels([])
inset1.set_yticklabels([])

# -----------------------------------------------------------------------------
# Plot zoomed curves in inset 2
# -----------------------------------------------------------------------------
inset2.plot(x_org[mask_zoom2_org], df_org.iloc[:, 1][mask_zoom2_org],
            color="blue", linewidth=lw_inset)
inset2.plot(x_dn[mask_zoom2_dn], df_dn.iloc[:, 1][mask_zoom2_dn],
            color="red", linewidth=lw_inset)

inset2.set_xlim(x3, x4)
inset2.set_ylim(y3, y4)
inset2.grid(True, linewidth=0.3)

inset2.xaxis.set_major_formatter(formatter)
inset2.set_xticklabels([])
inset2.set_yticklabels([])

# -----------------------------------------------------------------------------
# Connect each zoom rectangle to its corresponding inset axes
# mark_inset draws connector lines and an inset boundary indication.
# loc1/loc2 specify which corners to connect.
# -----------------------------------------------------------------------------
mark_inset(ax, inset1, loc1=1, loc2=2, fc="none", ec="gray", linewidth=0.6)
mark_inset(ax, inset2, loc1=3, loc2=4, fc="none", ec="gray", linewidth=0.6)

# -----------------------------------------------------------------------------
# Save and display
# -----------------------------------------------------------------------------
plt.tight_layout()

out_path = os.path.join("Figures", "Figure 8.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

plt.savefig(out_path, dpi=300)
plt.show()

print(f"Saved: {out_path}")
