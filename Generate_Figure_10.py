import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.signal import correlate

# -----------------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------------
mpl.rcParams["font.family"] = "Times New Roman"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# (1) Video CSV trimming (denoised blue intensity)
VIDEO_START_S = 60.0       # sec
VIDEO_END_S   = 4400.0     # sec
VIDEO_IN_CSV  = "mean_blue_denoised.csv"
VIDEO_OUT_CSV = "mean_blue_denoised_trimmed.csv"

# (2) Reactor CSV window (time-indexed CSV, first column is datetime index)
REACTOR_CSV_PATH = "2025_12_12.csv"
REACTOR_START    = "2025-12-12 13:30:00"
REACTOR_END      = "2025-12-12 15:30:00"
REACTOR_SIGNAL   = "LinPower"   # signal used for alignment + fitting

# (3) Sampling / alignment
FS_HZ   = 10.0
ANCHOR1 = (0.1, 400.0)     # seconds (w1) within aligned time axis
ANCHOR2 = (630.0, 930.0)   # seconds (w2) within aligned time axis

# (4) Outputs
OUT_DIR = Path("Figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = OUT_DIR / "Figure 10.png"

ALIGNED_LINEAR_CSV = "aligned_linear_fit_LinPower.csv"


# -----------------------------------------------------------------------------
# Helpers: cleaning, standardization, resampling
# -----------------------------------------------------------------------------
def sanitize_series(s: pd.Series) -> pd.Series:
    """
    Coerce to numeric, replace inf with NaN, and fill gaps.
    Interpolation uses the Series index (must be time-like for method='time').
    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.interpolate("time").bfill().ffill()


def zscore(x: np.ndarray) -> np.ndarray:
    """
    Z-score with NaN-safe statistics; remaining non-finite values become 0.
    """
    x = np.asarray(x, float)
    x[~np.isfinite(x)] = np.nan
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std == 0:
        std = 1.0
    z = (x - mean) / std
    z[~np.isfinite(z)] = 0.0
    return z


def resample_to_10hz_from_datetime_index(df_win: pd.DataFrame, col: str, fs_hz: float = 10.0) -> pd.Series:
    """
    Take a dataframe window indexed by datetime, and produce a uniformly sampled
    Series at fs_hz, indexed by timedelta from the first sample.
    """
    # Build a time-indexed Series (datetime index is required for method='time')
    s = pd.Series(df_win[col].values, index=pd.to_datetime(df_win.index))
    s = sanitize_series(s).dropna()

    # Rebase to t=0 using timedeltas
    t0 = s.index[0]
    s0 = pd.Series(s.values, index=(s.index - t0))

    # Create uniform grid and interpolate
    grid = pd.timedelta_range(0, s0.index[-1], freq=pd.Timedelta(seconds=1 / fs_hz))
    sfs = s0.reindex(s0.index.union(grid)).interpolate("time").reindex(grid)
    return sfs.bfill().ffill()


def pick_video_value_col(df: pd.DataFrame) -> str:
    """
    Choose a plausible intensity column from known candidates; otherwise pick
    the first numeric column.
    """
    for c in [
        "mean_blue_intensity", "mean_blue", "blue", "intensity",
        "mean_green", "mean_green_intensity"
    ]:
        if c in df.columns:
            return c

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise ValueError("No numeric intensity column found in video CSV.")
    return num_cols[0]


def video_csv_to_10hz(path: str, fs_hz: float = 10.0) -> pd.Series:
    """
    Load a video-derived CSV and resample to fs_hz. Returns a Series indexed
    by timedelta from 0 to end (seconds).
    """
    b = pd.read_csv(path)

    # Determine time column
    tcol = "time_s" if "time_s" in b.columns else b.columns[0]
    vcol = pick_video_value_col(b)

    tt = pd.to_numeric(b[tcol], errors="coerce")
    yy = pd.to_numeric(b[vcol], errors="coerce")
    m = tt.notna() & yy.notna()

    tt = tt[m].to_numpy()
    yy = yy[m].to_numpy()

    # Sort by time and remove duplicates (strictly increasing time required)
    order = np.argsort(tt)
    tt, yy = tt[order], yy[order]
    keep = np.concatenate(([True], np.diff(tt) > 0))
    tt, yy = tt[keep], yy[keep]

    # Convert to timedelta index and interpolate onto uniform grid
    t = pd.to_timedelta(tt, unit="s")
    s = pd.Series(yy, index=t)

    grid = pd.timedelta_range(0, s.index[-1], freq=pd.Timedelta(seconds=1 / fs_hz))
    sfs = s.reindex(s.index.union(grid)).interpolate("time").reindex(grid)
    return sfs.bfill().ffill()


def best_shift(long_s: pd.Series, short_s: pd.Series, fs_hz: float = 10.0) -> tuple[int, float, float]:
    """
    Find the best alignment shift (in samples and seconds) by maximizing the
    normalized cross-correlation of z-scored signals.

    long_s: reference signal (typically reactor signal)
    short_s: signal to align to the start of long_s (typically video signal)
    """
    L = sanitize_series(long_s).to_numpy(float)
    S = sanitize_series(short_s).to_numpy(float)

    corr = correlate(zscore(L), zscore(S), mode="valid", method="fft")
    corr[~np.isfinite(corr)] = -np.inf

    k = int(np.argmax(corr))
    return k, k / fs_hz, float(corr[k] / len(S))


# -----------------------------------------------------------------------------
# Fitting methods
# -----------------------------------------------------------------------------
def anchored_scale(seg_csv: np.ndarray, seg_vid: np.ndarray, fs_hz: float,
                   w1: tuple[float, float], w2: tuple[float, float]) -> tuple[float, float, np.ndarray, dict]:
    """
    Fit y = a*x + b using two anchor windows (w1, w2) defined in seconds over
    the aligned timeline, by matching mean values in those windows.
    """
    t = np.arange(len(seg_vid)) / fs_hz

    def mean_in(t0: float, t1: float, y: np.ndarray) -> float:
        m = (t >= t0) & (t <= t1)
        vals = y[m][np.isfinite(y[m])]
        if len(vals) == 0:
            raise ValueError(f"No valid samples in [{t0},{t1}] sec")
        return float(vals.mean())

    y1, x1 = mean_in(*w1, seg_csv), mean_in(*w1, seg_vid)
    y2, x2 = mean_in(*w2, seg_csv), mean_in(*w2, seg_vid)

    a = (y2 - y1) / (x2 - x1 + 1e-12)
    b = y1 - a * x1

    yhat = a * seg_vid + b
    resid = seg_csv - yhat

    mse = float((resid ** 2).mean())
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((seg_csv - seg_csv.mean()) ** 2).sum()) + 1e-12
    r2 = 1 - ss_res / ss_tot
    r = float(np.corrcoef(seg_csv, yhat)[0, 1])

    return a, b, yhat, {"mse": mse, "r2": r2, "r": r}


def mse_linear_fit(seg_csv: np.ndarray, seg_vid: np.ndarray) -> tuple[float, float, np.ndarray, dict]:
    """
    Fit y = a*x + b by least squares (minimizes MSE globally).
    """
    X = np.vstack([seg_vid, np.ones_like(seg_vid)]).T
    a, b = np.linalg.lstsq(X, seg_csv, rcond=None)[0]

    yhat = a * seg_vid + b
    resid = seg_csv - yhat

    mse = float((resid ** 2).mean())
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((seg_csv - seg_csv.mean()) ** 2).sum()) + 1e-12
    r2 = 1 - ss_res / ss_tot
    r = float(np.corrcoef(seg_csv, yhat)[0, 1])

    return a, b, yhat, {"mse": mse, "r2": r2, "r": r}


# -----------------------------------------------------------------------------
# Step 1) Trim/rebase the video CSV (denoised blue intensity)
# -----------------------------------------------------------------------------
def trim_video_csv(in_csv: str, out_csv: str, start_s: float, end_s: float) -> Path:
    """
    Trim a video CSV to [start_s, end_s] seconds and rebase time so start_s -> 0.
    Supports either:
      - a 'time_s' column, or
      - first column as time
    """
    df = pd.read_csv(in_csv)

    # Normalize/standardize time column to be the index named "time_s"
    if "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df = df.dropna(subset=["time_s"]).set_index("time_s")
    else:
        # Fallback: first column is time, remaining columns are data
        df.index = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        df = df[~df.index.isna()]
        df.index.name = "time_s"
        df = df.iloc[:, 1:]

    # Trim
    df_trim = df.loc[(df.index >= start_s) & (df.index <= end_s)].copy()

    # Rebase time so that start_s becomes 0
    df_trim.index = df_trim.index - start_s
    df_trim.index.name = "time_s"

    # Save
    out_path = Path(out_csv)
    df_trim.to_csv(out_path)

    return out_path


# -----------------------------------------------------------------------------
# Step 2) Align reactor signal to video signal, then compute two fits
# -----------------------------------------------------------------------------
def build_aligned_fit_table() -> pd.DataFrame:
    """
    Main alignment + fitting pipeline (single output table):
      - Load reactor CSV and select a time window
      - Resample reactor signal and video signal to FS_HZ
      - Estimate time shift via cross-correlation
      - Fit linear mapping using (A) anchor scaling and (B) MSE regression
      - Save ONE combined CSV with both fit results
    """
    # Load reactor data (datetime index in first column)
    df_reactor = pd.read_csv(REACTOR_CSV_PATH, index_col=0, parse_dates=True)
    df_win = df_reactor.loc[REACTOR_START:REACTOR_END].copy()

    if REACTOR_SIGNAL not in df_win.columns:
        raise KeyError(f"Required column '{REACTOR_SIGNAL}' not found in {REACTOR_CSV_PATH}.")

    # Resample both signals to FS_HZ
    s_csv = resample_to_10hz_from_datetime_index(df_win, REACTOR_SIGNAL, fs_hz=FS_HZ)
    s_vid = video_csv_to_10hz(VIDEO_OUT_CSV, fs_hz=FS_HZ)

    # Estimate shift (align video start to reactor timeline)
    k, sec_shift, ncc = best_shift(s_csv, s_vid, fs_hz=FS_HZ)
    print(f"\n[{REACTOR_SIGNAL}] shift = {sec_shift:.2f} s, NCC = {ncc:.4f}")

    # Create aligned segments with common length
    L = min(len(s_vid), len(s_csv) - k)
    if L <= 0:
        raise ValueError("Alignment produced non-positive overlap length. Check time windows and data lengths.")

    seg_true = s_csv.iloc[k:k + L].to_numpy(float)   # reactor (true)
    seg_vid  = s_vid.iloc[:L].to_numpy(float)        # video
    t = np.arange(L) / FS_HZ

    # (A) Anchor-based fit
    aA, bA, yA, infoA = anchored_scale(seg_true, seg_vid, fs_hz=FS_HZ, w1=ANCHOR1, w2=ANCHOR2)
    print("\n[Anchor Fit]")
    print(f"a = {aA}, b = {bA}, info = {infoA}")

    # (B) MSE-based fit
    aM, bM, yM, infoM = mse_linear_fit(seg_true, seg_vid)
    print("\n[MSE Fit]")
    print(f"a = {aM}, b = {bM}, info = {infoM}")

    # Build ONE combined table
    df_out = pd.DataFrame({
        "t_video_s": t,
        f"{REACTOR_SIGNAL}_true": seg_true,
        "video": seg_vid,
        f"{REACTOR_SIGNAL}_fit_mse": yM,
        f"{REACTOR_SIGNAL}_fit_anchor": yA,
    })

    df_out.to_csv(ALIGNED_LINEAR_CSV, index=False)
    print(f"\nSaved: {ALIGNED_LINEAR_CSV}")

    return df_out


# -----------------------------------------------------------------------------
# Step 3) Plot Figure 10 (Detector vs. video-based fitted power)
# -----------------------------------------------------------------------------
def sec_to_mmss(val, pos):
    """
    Convert seconds to MM:SS tick labels.
    """
    val = int(val)
    if val < 0:
        return ""
    m = val // 60
    s = val % 60
    return f"{m:02d}:{s:02d}"


def plot_figure_10(df: pd.DataFrame) -> None:
    """
    Produce Figure 10 from a single combined dataframe:
      - True reactor signal (kW)
      - Anchor fit (kW)
      - MSE fit (kW)
    """
    true_col = f"{REACTOR_SIGNAL}_true"
    fitA_col = f"{REACTOR_SIGNAL}_fit_anchor"
    fitM_col = f"{REACTOR_SIGNAL}_fit_mse"

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df["t_video_s"], df[true_col] / 1000, linewidth=0.7, label="Detector (NM) power")
    ax.plot(df["t_video_s"], df[fitA_col] / 1000, linewidth=0.7, label="Anchor Fit")
    ax.plot(df["t_video_s"], df[fitM_col] / 1000, linewidth=0.7, label="MSE Fit")

    ax.set_xlabel("Video time (MM:SS)")
    ax.set_ylabel("Power (kW)")

    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_formatter(FuncFormatter(sec_to_mmss))
    ax.xaxis.set_major_locator(MultipleLocator(600))
    ax.yaxis.set_major_locator(MultipleLocator(100))

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=300)
    plt.show()

    print(f"Saved figure: {FIG_PATH}")



# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Trim/rebase the denoised video CSV
    out_video_path = trim_video_csv(
        in_csv=VIDEO_IN_CSV,
        out_csv=VIDEO_OUT_CSV,
        start_s=VIDEO_START_S,
        end_s=VIDEO_END_S
    )
    print(f"Saved trimmed CSV: {out_video_path}")

    # 2) Align + fit and save ONE combined table
    df_linear = build_aligned_fit_table()

    # 3) Plot and save Figure 10
    plot_figure_10(df_linear)
