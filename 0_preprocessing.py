import subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm  # Changed from tqdm.notebook for Terminal/Prompt compatibility

# =====================================================
# Configuration
# =====================================================
# Define the absolute path to your ffmpeg executable on Windows
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe" 

# Use the directory where the script is located
CWD = Path(__file__).resolve().parent

# List of raw video segments to be processed
FILES = [
    "GX010056 - Part 1.MP4",
    "GX020056 - Part 2.MP4",
    "GX030056 - Part 3.MP4",
    "GX040056 - Part 4.MP4",
    "GX050056 - Part 5.MP4",
]

# Region of Interest (ROI) coordinates for cropping the reactor core area
# Format: X-offset, Y-offset, Width, Height
X, Y, W, H = 1341, 528, 1422, 996

# Output filenames
MERGED_VIDEO   = CWD / "merged_crop.mp4"
DENOISED_VIDEO = CWD / "merged_crop_denoised.mp4"

# High-quality 3D denoiser filter settings
DENOISE_FILTER = "hqdn3d=6:4:8:6"

# =====================================================
# Helpers
# =====================================================
def run_ffmpeg(cmd):
    """Executes the ffmpeg command using subprocess."""
    # Using shell=True for Windows environment stability
    subprocess.run(cmd, check=True, shell=True)

# -----------------------------------------------------
# 1) Crop each part, then concat
# -----------------------------------------------------
def crop_and_concat(files, output_video):
    """Crops individual segments and merges them into one video."""
    cropped_parts = []

    for name in files:
        inp = CWD / name
        if not inp.exists():
            print(f"⚠️ Warning: File not found: {name}")
            continue

        out = CWD / f"{inp.stem}_crop.mp4"
        cropped_parts.append(out)

        # FFMPEG Command: Crop + Re-encode
        cmd = [
            f'"{FFMPEG}"', "-y",
            "-i", f'"{str(inp)}"',
            "-vf", f"crop={W}:{H}:{X}:{Y}",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            f'"{str(out)}"',
        ]
        print(f"-> Processing: {name}")
        run_ffmpeg(" ".join(cmd))

    # Create a temporary list for concatenation
    concat_list = CWD / "concat_list.txt"
    with open(concat_list, "w") as f:
        for v in cropped_parts:
            f.write(f"file '{v.resolve().as_posix()}'\n")

    # Final Merge
    concat_cmd = [
        f'"{FFMPEG}"', "-y",
        "-f", "concat", "-safe", "0",
        "-i", f'"{str(concat_list)}"',
        "-c", "copy",
        f'"{str(output_video)}"',
    ]
    run_ffmpeg(" ".join(concat_cmd))

    # Cleanup temporary files
    if concat_list.exists(): concat_list.unlink()
    for v in cropped_parts:
        if v.exists(): v.unlink()

# -----------------------------------------------------
# 2) Denoise
# -----------------------------------------------------
def denoise_video(inp, outp):
    """Applies hqdn3d denoising filter."""
    cmd = [
        f'"{FFMPEG}"', "-y",
        "-i", f'"{str(inp)}"',
        "-vf", DENOISE_FILTER,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        f'"{str(outp)}"',
    ]
    run_ffmpeg(" ".join(cmd))

# -----------------------------------------------------
# 3) Per-frame RGB mean
# -----------------------------------------------------
def per_frame_rgb_mean(video_path: Path):
    """Extracts mean RGB values for every frame with a progress bar."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    t, r, g, b = [], [], [], []
    
    # tqdm will now show a standard text progress bar in Anaconda Prompt
    with tqdm(total=nframes, desc=f"Analyzing {video_path.name}", unit="fr") as pbar:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
                
            mean_bgr = frame.mean(axis=(0, 1))
            b.append(mean_bgr[0])
            g.append(mean_bgr[1])
            r.append(mean_bgr[2])

            t.append(i / fps)
            i += 1
            pbar.update(1)

    cap.release()
    return pd.DataFrame({"time_s": t, "mean_red": r, "mean_green": g, "mean_blue": b})

# =====================================================
# Main execution
# =====================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" Reactor Power Monitoring - Video Preprocessing ")
    print("="*50 + "\n")

    print("[Step 1/4] Cropping and Concatenating...")
    crop_and_concat(FILES, MERGED_VIDEO)

    print("\n[Step 2/4] Denoising (This may take a while)...")
    denoise_video(MERGED_VIDEO, DENOISED_VIDEO)

    print("\n[Step 3/4] Extracting RGB values from Original...")
    df_org = per_frame_rgb_mean(MERGED_VIDEO)
    df_org.to_csv(CWD / "rgb_means_original.csv", index=False)
    
    print("\n[Step 4/4] Extracting RGB values from Denoised...")
    df_dn = per_frame_rgb_mean(DENOISED_VIDEO)
    df_dn.to_csv(CWD / "rgb_means_denoised.csv", index=False)

    print("\n" + "="*50)
    print("✅ COMPLETED SUCCESSFULLY")
    print(f"Data saved in: {CWD}")
    print("="*50)