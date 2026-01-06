import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Set global font ----
mpl.rcParams["font.family"] = "Times New Roman"

# ---- Load data ----
df_lin = pd.read_csv("aligned_linear_fit_LinPower.csv")

t = df_lin["t_video_s"].to_numpy()
y_true = df_lin["LinPower_true"].to_numpy()
x = df_lin["video"].to_numpy()

# ---- Polynomial degrees to test ----
degrees = [1, 2, 3, 4, 5]
mses = []

# ---- Initialize output DataFrame ----
df_out = pd.DataFrame({
    "t_video_s": t,
    "LinPower_true": y_true,
    "video": x
})

# ---- Compute MSE for polynomial degrees ----
for deg in degrees:
    coeffs = np.polyfit(x, y_true, deg)
    y_pred = np.polyval(coeffs, x)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mses.append(mse)

    # Add each polynomial fit as a new column
    df_out[f"LinPower_fit_poly{deg}"] = y_pred

# ---- Save ONE combined CSV ----
out_csv = "aligned_polynomial_fits_LinPower.csv"
df_out.to_csv(out_csv, index=False)
print(f"Saved combined polynomial fit CSV: {out_csv}")

# ---- Matplotlib Plot: MSE vs Polynomial Degree ----
plt.figure(figsize=(10, 4))

plt.plot(
    degrees,
    mses,
    "-o",
    linewidth=2,
    markersize=6
)

plt.yscale("log")

plt.xlabel("Polynomial Degree", fontsize=12)
plt.ylabel("MSE (log scale)", fontsize=12)

plt.xticks(degrees, fontsize=11)
plt.yticks(fontsize=11)

plt.grid(True, which="both", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.savefig("Figures/Figure 12.png", dpi=300)
plt.show()

# ---- Print MSE values ----
print("\nMSE by polynomial degree:")
for d, m in zip(degrees, mses):
    print(f"Degree {d}: MSE = {m:.6e}")
