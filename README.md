<div align="center">

# 🧭 Gyro Predictor
**Deep Learning & Classic Filtering for IMU Gyroscope Forecasting**  
Predict *k*-steps-ahead gyro readings to compensate AR/MR latency.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)]()
[![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-informational)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>

---

## ✨ Overview
This repository trains **1D CNN**, **LSTM**, and **Transformer** models to predict future gyroscope samples from the recent history of IMU data.  
It also includes two **classical baselines** — **RLS** and **RLS Lattice** — optimized with NumPy/Numba.  
Everything runs **from the repository root** (no absolute paths); artifacts are saved under `./artifacts/`.

**Highlights**
- ✅ **Strict calibration** from the CSV row named **`calibration_file`** (JSON with `S*` 3×3 and `b_*` biases as 9‑element vectors). **Raw is never used.**
- ✅ **Time‑jump aware segmentation** using microsecond timestamps (Bluetooth jitter); only contiguous segments are used.
- ✅ **Multi‑horizon prediction**: *k* ∈ {1,2,3,4}, at **52 Hz** and **104 Hz**.
- ✅ **Reproducible training**: AMP, early stopping, CUDA‑pinned multi‑worker DataLoaders (Windows‑safe).
- ✅ **Comprehensive evaluation**: L2/MAE/RMSE, latency/sample, parameter counts, heatmaps (mean/median/p90/p99), overlays, histograms.
- ✅ **Readable model tables**: CSV per architecture with layer names, output shapes, kernel/stride, activations, and params.

---

## 📦 Project Structure
```
.
├─ epsilon_52Hz/                 # place CSVs here (repo root)
├─ epsilon_104Hz/                # place CSVs here (repo root)
├─ gyro_predictor/
│  ├─ data/                      # CSV parsing, calibration, time-jump split
│  ├─ train/                     # datasets, models, trainer, factory
│  ├─ classic/                   # RLS & RLS Lattice (+ parallel runner & caching)
│  ├─ viz/                       # plots, summaries, model layer tables
│  └─ utils/                     # config, paths, small helpers
├─ artifacts/                    # outputs (created automatically)
├─ main.ipynb                    # end-to-end: sanity → train → eval → baselines → plots
└─ README.md
```

---

## ⚙️ Installation
> Python **3.10+** is recommended. Install a matching CUDA build of PyTorch if you use GPU.

```bash
# 1) Create & activate a clean environment (example with venv)
python -m venv .venv
# Windows (PowerShell):
. .venv/Scripts/activate
# Linux/Mac:
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

---

## 📁 Data Layout (relative paths only)
Put your CSV files under the repo root:
```
.
├─ epsilon_52Hz/*.csv
└─ epsilon_104Hz/*.csv
```
Each CSV contains a row `calibration_file` with JSON like:
```json
{ 
  "Sgyro1": [s11, s12, s13, s21, s22, s23, s31, s32, s33],
  "b_gyro1": [bx, by, bz],
  "Sacc1":  [ ... 9 vals ... ],
  "b_acc1": [ ... ]
}
```
Calibration is always applied:
```
calibrated = S @ (raw - b)
```
If a file lacks valid calibration JSON, it is **skipped** (never train on raw).

---

## 🚀 Quick Start (VS Code Notebook)
Open **`main.ipynb`** from the repo root and run top→bottom. No path editing needed.

The notebook performs:
1. **Sanity checks** (data counts, time‑jumps, calibration preview).
2. **Training** CNN / LSTM / Transformer for *k* ∈ {1..4} and both rates.
3. **Evaluation** → saves `pred.npy`, `true.npy`, `metrics.json`, and latency.
4. **Baselines** (RLS & RLS Lattice) in parallel with caching.
5. **Summaries** → heatmaps, scatter plots (latency/params vs L2), overlays, error histograms.
6. **Model layer tables** → CSVs per architecture.

> Models/baselines are **skipped** if outputs exist (toggle with `cfg.retrain=True`).  
> Artifacts live under `./artifacts/` (created automatically).

---

## 🔧 Configuration (minimal)
Configuration lives in `gyro_predictor/utils/config.py` and uses **relative paths by default**:
```python
from gyro_predictor.utils.config import Config
class Config:
class Config:
    data_root: Path = Path(".")                      # project root for data (expects ./epsilon_52Hz and ./epsilon_104Hz)
    artifacts: Path = Path("./artifacts")            # output dir for models, metrics, plots, caches (splits), etc.
    use_acc: bool = True                             # include accelerometer (ax, ay, az) alongside gyro inputs
    skip_uncalibrated: bool = True                   # strictly skip files without valid calibration (never use raw)
    jump_factor: float = 1.5                         # split where dt > jump_factor × nominal dt (handles BT time jumps)
    min_segment_len: int = 32                        # minimum contiguous segment length (in samples) kept after splitting
    hz_list: tuple = (52, 104)                       # sampling rates (Hz) to process; must match dataset folders
    horizons: tuple = (1,2,3,4)                      # prediction horizons k (steps ahead) to train/evaluate
    history_map: dict = {1:8, 2:10, 3:10, 4:12},     # mapping k → history length H (past samples fed to the model)
    split_ratios: tuple = (0.70, 0.15, 0.15)         # train/val/test split fractions applied per-Hz to segments
    seed: int = 42                                   # global RNG seed (splits, init, augmentation)
    epochs: int = 50                                 # max training epochs (early stopping may stop earlier)
    batch_size: int = 2048                           # DataLoader batch size (adjust to your VRAM)
    lr: float = 1e-3                                 # optimizer learning rate (Adam/AdamW)
    weight_decay: float = 1e-4                       # L2 weight decay (AdamW) strength
    early_patience: int = 5                          # stop after this many epochs with no val-loss improvement
    retrain: bool = False                            # if True, ignore cached weights/metrics and retrain from scratch
    
    num_workers: int = 8
    persistent_workers: bool = True                  # only takes effect if num_workers > 0
    prefetch_factor: int = 4                         # only used if num_workers > 0
    pin_memory: bool = True                          # CUDA input pipeline
    seed: int = 42                                   # for worker seeding
)
```
> Run the notebook as-is; you don’t need to touch paths.

---

## 🧪 Models & Baselines
- **Neural Nets** (via `train/factory.py`)
  - `cnn`: 1D convolutional network for sequences
  - `lstm`: multi-layer LSTM
  - `transformer`: Transformer encoder for 1D sequences
- **Classical** (via `classic/`)
  - `rls.py`: decaying RLS (robust update; NumPy/Numba)
  - `rls_lattice.py`: RLS lattice (a‑posteriori, numerically stable)
  - `baseline_runner.py`: runs all (Hz, k) in parallel with disk caching

---

## 📊 Outputs & Visualizations
Artifacts are saved under `./artifacts/`:
```
artifacts/
  52Hz/
    cnn/k1/H8/          model.pt(.safe), meta.json, metrics.json, pred.npy, true.npy
    lstm/...            (... similar ...)
    transformer/...
    RLS/...             pred.npy, true.npy, metrics.json, meta.json
    RLSLpos/...
  104Hz/...
  summaries/
    metrics_latency_params.csv
    heatmap_l2_mean.png / heatmap_l2_median.png / heatmap_l2_p90.png / heatmap_l2_p99.png
    scatter_latency_vs_l2_k*.png, scatter_params_vs_l2_k*.png
    model_tables/{cnn,lstm,transformer}_layers.csv
```
**Metrics**: L2 mean/median/p90/p99, MAE, RMSE, latency per sample, parameter counts.  
**Plots**: heatmaps, latency/params vs L2 (per *k*), overlays, histograms.  
**Tables**: CSV layer tables for each model (clean ASCII shapes).

---

## ⚡ Tips
- Use **AMP** on RTX 40‑series for faster training (enabled by default).
- Tune `num_workers` (8–12 on a 12‑core is typical). Keep `persistent_workers=True`.
- On Windows, all DataLoader callables are **top‑level** (no lambdas) to avoid pickling errors.

---

## 📝 License
MIT for code. IMU datasets remain the property of their owners.
