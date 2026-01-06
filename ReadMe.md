# A Video-Based Optical Approach to Reactor Power Monitoring using Cherenkov Emission

This repository contains the official source code and data processing pipeline for the research paper: **"A Video-Based Optical Approach to Reactor Power Monitoring using Cherenkov Emission."**

---

## 🚀 Getting Started

### Prerequisites
To process the raw video data, you must have **FFmpeg** installed on your system.
* **FFmpeg**: [Download here](https://ffmpeg.org/download.html) or install via your package manager (e.g., `brew install ffmpeg` or `sudo apt install ffmpeg`).

### Data Acquisition
Due to the large file size, the raw video datasets used in this study are hosted externally.
* **Download Link:** [UTexas Box - Video Dataset](https://utexas.box.com/s/bd80q5yukeabywpuq04h31nhen6z6tbj)

---

## 🛠️ Data Processing

### 1. Preprocessing Raw Videos
Once you have downloaded the videos, place them in the appropriate directory and run the following script to extract frames and prepare the data:

```bash
python 0_preprocessing.py