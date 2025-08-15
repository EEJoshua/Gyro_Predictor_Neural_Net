# gyro_predictor/viz/summaries.py
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from gyro_predictor.utils.paths import artifact_path
from gyro_predictor.train.factory import build_model
from gyro_predictor.viz.plots import plot_timeseries, plot_error_hist

BASELINES = {"RLS", "RLSLpos"}  # <- excluded from latency comparisons


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _count_params(arch: str, in_ch: int) -> float:
    model = build_model(arch, in_ch=in_ch)
    return float(sum(p.numel() for p in model.parameters()))


def _load_pred_true(base: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load pred/true arrays if present; else None."""
    p, t = base / "pred.npy", base / "true.npy"
    if p.exists() and t.exists():
        return np.load(p), np.load(t)
    return None


def _compute_mse_percentiles(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute global mean MSE and per-sample MSE percentiles.
    - per-sample MSE = mean over 3 axes of squared error at each sample
    - returns (mse_mean, mse_median, mse_p90, mse_p99)
    """
    if pred.size == 0 or true.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    se = (pred - true) ** 2                    # (N, 3)
    per_sample_mse = se.mean(axis=1)           # (N,)
    mse_mean = float(per_sample_mse.mean())
    return (
        mse_mean,
        float(np.median(per_sample_mse)),
        float(np.percentile(per_sample_mse, 90)),
        float(np.percentile(per_sample_mse, 99)),
    )


def _collect_dataframe(
    cfg,
    results: Dict[Tuple[int, str, int], Dict[str, Any]],
    baselines: Dict[Tuple[int, str, int], Dict[str, Any]],
) -> pd.DataFrame:
    """
    Returns a tidy dataframe with:
      hz, family, model, k, mse, mse_med, mse_p90, mse_p99, latency_ms, params, source
    """
    ARCHS = ["cnn", "lstm", "transformer"]

    records = []
    param_cache: Dict[Tuple[str, int], float] = {}

    # NN models
    for (hz, arch, k), m in results.items():
        base = artifact_path(cfg.artifacts, hz, arch, k, cfg.history_map[k], "model.pt").parent
        meta_path = base / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            in_ch = int(meta.get("in_ch", 6 if cfg.use_acc else 3))
        else:
            in_ch = 6 if cfg.use_acc else 3

        key = (arch, in_ch)
        if key not in param_cache:
            try:
                param_cache[key] = _count_params(arch, in_ch)
            except Exception:
                param_cache[key] = np.nan

        # Prefer computing from saved pred/true; fall back to RMSE in metrics if needed
        mse_mean = mse_med = mse_p90 = mse_p99 = np.nan
        pt = _load_pred_true(base)
        if pt is not None:
            pred, true = pt
            mse_mean, mse_med, mse_p90, mse_p99 = _compute_mse_percentiles(pred, true)
        else:
            # fallback from per-axis RMSE if available: MSE = mean(RMSE^2 over axes)
            rmse = m.get("rmse", None)
            if isinstance(rmse, (list, tuple)) and len(rmse) == 3:
                try:
                    rmse_arr = np.asarray(rmse, dtype=float)
                    mse_mean = float(np.mean(rmse_arr ** 2))
                except Exception:
                    pass

        records.append(dict(
            hz=int(hz), family=arch, model=arch, k=int(k),
            mse=mse_mean,
            mse_med=mse_med, mse_p90=mse_p90, mse_p99=mse_p99,
            latency_ms=_safe_float(m.get("latency_ms_per_sample")),
            params=param_cache[key],
            source="nn",
        ))

    # Baselines (RLS / RLSLpos)
    for (hz, algo, k), m in baselines.items():
        base = artifact_path(cfg.artifacts, hz, algo, k, cfg.history_map[k], "pred.npy").parent
        meta_path = base / "meta.json"
        N = 8
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                N = int(meta.get("N", N))
            except Exception:
                pass

        params = np.nan
        thr = m.get("throughput_samples_per_s", m.get("throughput_sps", None))
        latency_ms = 1000.0 / thr if thr and thr > 0 else np.nan

        mse_mean = mse_med = mse_p90 = mse_p99 = np.nan
        pt = _load_pred_true(base)
        if pt is not None:
            pred, true = pt
            mse_mean, mse_med, mse_p90, mse_p99 = _compute_mse_percentiles(pred, true)
        else:
            rmse = m.get("rmse", None)
            if isinstance(rmse, (list, tuple)) and len(rmse) == 3:
                try:
                    rmse_arr = np.asarray(rmse, dtype=float)
                    mse_mean = float(np.mean(rmse_arr ** 2))
                except Exception:
                    pass

        records.append(dict(
            hz=int(hz), family=algo, model=algo, k=int(k),
            mse=mse_mean,
            mse_med=mse_med, mse_p90=mse_p90, mse_p99=mse_p99,
            latency_ms=_safe_float(m.get("latency_ms_per_sample", latency_ms)),
            params=params,
            source="baseline",
        ))

    df = pd.DataFrame.from_records(records)
    return df


def _heatmap_from_df(df: pd.DataFrame, Ks: list[int], families: list[str],
                     title: str, value_col: str, out_path: Path) -> None:
    """Generic heatmap builder over df with rows=(Hz-family), cols=k."""
    row_labels = []
    for hz in sorted(df["hz"].unique()):
        for fam in families:
            if ((df["hz"] == hz) & (df["family"] == fam)).any():
                row_labels.append(f"{hz}Hz-{fam}")

    Z = np.full((len(row_labels), len(Ks)), np.nan, dtype=float)
    for r, label in enumerate(row_labels):
        hz_str, fam = label.split("Hz-")
        hz_val = int(hz_str)
        for c, k in enumerate(Ks):
            subset = df[(df.hz == hz_val) & (df.family == fam) & (df.k == k)]
            if not subset.empty and value_col in subset:
                try:
                    Z[r, c] = float(subset[value_col].mean())
                except Exception:
                    Z[r, c] = np.nan

    plt.figure(figsize=(1.6 * len(Ks) + 4, 0.6 * len(row_labels) + 2.5))
    cmap = plt.cm.viridis.copy(); cmap.set_bad(color="#eeeeee")
    im = plt.imshow(np.ma.masked_invalid(Z), aspect="auto", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=value_col.replace("_", " "))
    plt.xticks(range(len(Ks)), [f"k={k}" for k in Ks])
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    for r in range(Z.shape[0]):
        for c in range(Z.shape[1]):
            if not np.isnan(Z[r, c]):
                val = Z[r, c]
                col = "white" if np.mean(im.cmap(im.norm(val))[:3]) < 0.5 else "black"
                plt.text(c, r, f"{val:.6f}", ha="center", va="center", fontsize=8, color=col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show(); plt.close()


def generate_all_summary_plots(
    cfg,
    results: Dict[Tuple[int, str, int], Dict[str, Any]],
    baselines: Dict[Tuple[int, str, int], Dict[str, Any]],
    save_dir: str | Path | None = None,
    seed: int = 42,
    max_points: int = 2000,
    show: bool = True,
) -> pd.DataFrame:
    """
    Builds a tidy dataframe, saves it to CSV, and generates:
      - MSE heatmap (rows=Hz-family, cols=k)
      - MSE median heatmap
      - MSE 90th percentile heatmap
      - MSE 99th percentile heatmap
      - Per-Hz lines: MSE vs k
      - Per-k scatter: latency vs MSE (NN only)
      - Per-k scatter: params vs MSE (all)
      - Per-Hz lines: mean latency vs k (NN only)
      - One random spotlight run (unchanged visuals)
    """
    random.seed(seed)
    ARCHS = ["cnn", "lstm", "transformer"]
    BASELINES_LST = ["RLS", "RLSLpos"]
    Ks = sorted(list(cfg.horizons))

    summary_dir = Path(save_dir) if save_dir else (Path(cfg.artifacts) / "summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_dataframe(cfg, results, baselines)
    csv_path = summary_dir / "metrics_latency_params.csv"
    df.to_csv(csv_path, index=False)
    print(f"[summary] rows={len(df)} saved -> {csv_path}")

    families_all = ARCHS + BASELINES_LST

    # ---- Heatmaps: mean, median, p90, p99 (MSE) ----
    _heatmap_from_df(df, Ks, families_all, "MSE heatmap (all models × k × Hz)",
                     "mse", summary_dir / "heatmap_mse_mean.png")
    _heatmap_from_df(df, Ks, families_all, "MSE median heatmap (all models × k × Hz)",
                     "mse_med", summary_dir / "heatmap_mse_median.png")
    _heatmap_from_df(df, Ks, families_all, "MSE p90 heatmap (all models × k × Hz)",
                     "mse_p90", summary_dir / "heatmap_mse_p90.png")
    _heatmap_from_df(df, Ks, families_all, "MSE p99 heatmap (all models × k × Hz)",
                     "mse_p99", summary_dir / "heatmap_mse_p99.png")

    # ---- Per-Hz: MSE vs k (lines) ----
    for hz in sorted(df["hz"].unique()):
        plt.figure(figsize=(8, 4))
        for fam in families_all:
            d = df[(df.hz == hz) & (df.family == fam)].groupby("k")["mse"].mean().reindex(Ks)
            if d.notna().any():
                plt.plot(Ks, d.values, marker="o", label=fam)
        plt.title(f"{hz}Hz — MSE vs k")
        plt.xlabel("k (steps ahead)"); plt.ylabel("MSE (lower is better)")
        plt.xticks(Ks); plt.grid(alpha=0.3); plt.legend(ncol=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(summary_dir / f"mse_vs_k_{hz}Hz.png", dpi=150)
        if show: plt.show(); plt.close()

    # ---- Scatter: latency vs MSE (per k, NN only) ----
    markers = {52: "o", 104: "s"}
    colors  = {"cnn": "C0", "lstm": "C1", "transformer": "C2"}  # baselines intentionally omitted

    for k in Ks:
        d = df[(df.k == k) & (~df.family.isin(BASELINES))].dropna(subset=["mse", "latency_ms"])
        if d.empty:
            continue
        plt.figure(figsize=(8, 6))
        for fam in sorted(d.family.unique()):
            d_f = d[d.family == fam]
            for hz in sorted(d_f.hz.unique()):
                d_h = d_f[d_f.hz == hz]
                plt.scatter(d_h["mse"], d_h["latency_ms"], s=60,
                            marker=markers.get(hz, "o"),
                            edgecolors="none",
                            label=f"{fam}-{hz}Hz",
                            c=colors.get(fam, None))
        plt.title(f"Latency vs MSE — k={k}  (neural nets only)")
        plt.xlabel("MSE (lower is better)")
        plt.ylabel("Latency per sample (ms)")
        plt.grid(alpha=0.3)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(summary_dir / f"scatter_latency_vs_mse_k{k}_NNonly.png", dpi=150)
        if show: plt.show(); plt.close()

    # ---- Scatter: params vs MSE (per k, all) ----
    markers = {52: "o", 104: "s"}
    colors_nn = {"cnn": "C0", "lstm": "C1", "transformer": "C2"}

    for k in Ks:
        # exclude baselines here
        d = df[(df.k == k) & (~df.family.isin(BASELINES))].dropna(subset=["mse", "params"])
        if d.empty:
            continue

        plt.figure(figsize=(8, 6))
        for fam in sorted(d.family.unique()):
            d_f = d[d.family == fam]
            for hz in sorted(d_f.hz.unique()):
                d_h = d_f[d_f.hz == hz]
                plt.scatter(d_h["mse"], d_h["params"] / 1e6, s=60,
                            marker=markers.get(hz, "o"),
                            edgecolors="none",
                            label=f"{fam}-{hz}Hz",
                            c=colors_nn.get(fam, None))

        plt.title(f"Model size vs MSE — k={k}  (neural nets only)")
        plt.xlabel("MSE (lower is better)")
        plt.ylabel("# parameters (millions)")
        plt.grid(alpha=0.3)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(summary_dir / f"scatter_params_vs_mse_k{k}_NNonly.png", dpi=150)
        if show: plt.show(); plt.close()

    # ---- Per-Hz: mean latency vs k (NN only) ----
    for hz in sorted(df["hz"].unique()):
        plt.figure(figsize=(8, 4))
        for fam in ["cnn", "lstm", "transformer"]:
            d_line = (
                df[(df.hz == hz) & (df.family == fam)]
                .groupby("k")["latency_ms"].mean()
                .reindex(Ks)
            )
            if d_line.notna().any():
                plt.plot(Ks, d_line.values, marker="o", label=fam)
        plt.title(f"{hz}Hz — mean latency vs k (neural nets only)")
        plt.xlabel("k (steps ahead)")
        plt.ylabel("Latency per sample (ms)")
        plt.xticks(Ks)
        plt.grid(alpha=0.3)
        plt.legend(ncol=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(summary_dir / f"latency_vs_k_{hz}Hz_NNonly.png", dpi=150)
        if show: plt.show(); plt.close()

    # ---- Random spotlight (unchanged visuals) ----
    _rng = random.Random()
    if len(results):
        rand_key = _rng.choice(list(results.keys()))
        hz_r, arch_r, k_r = rand_key
        base_r = artifact_path(cfg.artifacts, hz_r, arch_r, k_r, cfg.history_map[k_r], "model.pt").parent
        pred_p, true_p = base_r / "pred.npy", base_r / "true.npy"
        if pred_p.exists() and true_p.exists():
            pred_r = np.load(pred_p); true_r = np.load(true_p)
            plot_timeseries(true_r, pred_r, title=f"RANDOM PICK → {hz_r}Hz {arch_r} k={k_r}", n_points=max_points)
            if show: plt.show()
            err_l2 = np.linalg.norm(pred_r - true_r, axis=1)
            plot_error_hist(err_l2, title=f"{hz_r}Hz {arch_r} k={k_r} — L2 histogram")
            if show: plt.show()

    return df
