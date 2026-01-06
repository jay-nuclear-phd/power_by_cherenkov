import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, MultipleLocator

mpl.rcParams["font.family"] = "Times New Roman"

# ---- Load original aligned data ----
df_lin = pd.read_csv("aligned_linear_fit_LinPower.csv")

t = df_lin["t_video_s"].to_numpy()
y_true = df_lin["LinPower_true"].to_numpy()
x = df_lin["video"].to_numpy()

# ---- Compute 2nd order polynomial fit ----
coeffs = np.polyfit(x, y_true, 3)
y_pred = np.polyval(coeffs, x)

print("2nd-order polynomial coefficients:", coeffs)

# ---- Time formatter (MM:SS) ----
def sec_to_mmss(x, pos):
    x = int(x)
    if x < 0:
        return ""
    m = x // 60
    s = x % 60
    return f"{m:02d}:{s:02d}"

# ---- Plot ----
plt.figure(figsize=(10, 4))

plt.plot(t, y_true / 1000, linewidth=0.7, label="Detector (NM) power")
plt.plot(t, y_pred / 1000, linewidth=0.7, label="3rd-order Polynomial Fit")

plt.xlabel("Video time (MM:SS)")
plt.ylabel("Power (kW)")

plt.grid(True, alpha=0.3)
plt.legend()

ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(sec_to_mmss))
ax.xaxis.set_major_locator(MultipleLocator(600))   # every 10 minutes
ax.yaxis.set_major_locator(MultipleLocator(100))   # 100 kW spacing

plt.tight_layout()
plt.savefig("Figures/Figure 13.png", dpi=300)
plt.show()
