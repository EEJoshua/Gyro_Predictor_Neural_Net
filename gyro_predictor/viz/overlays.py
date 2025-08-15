# gyro_predictor/viz/overlays.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from gyro_predictor.utils.paths import artifact_path


def _load_pred_true(base: Path) -> tuple[np.ndarray, np.ndarray] | None:
    pred_p, true_p = base / "pred.npy", base / "true.npy"
    if pred_p.exists() and true_p.exists():
        return np.load(pred_p), np.load(true_p)
    return None


def _mse_mean(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Global mean MSE across axes:
      per-sample MSE = mean((pred - true)^2 over axis=1)
      return mean(per-sample MSE)
    """
    if pred.size == 0 or true.size == 0:
        return float("nan")
    per_sample_mse = ((pred - true) ** 2).mean(axis=1)
    return float(per_sample_mse.mean()) if per_sample_mse.size else float("nan")


def generate_allk_overlays(
    cfg,
    results: Dict[Tuple[int, str, int], Dict[str, Any]],          # NN results dict
    baselines: Dict[Tuple[int, str, int], Dict[str, Any]] | None, # baseline metrics dict
    save_dir: str | Path | None = None,
    max_points: int = 2000,
    show: bool = True,
) -> None:
    """
    Overlay plots ONLY (everything else lives in summaries.py).

    For each Hz and for each family (cnn/lstm/transformer/RLS/RLSLpos),
    overlay ALL available horizons k on the same samples, per axis (gx/gy/gz).
    Legend entries include the global MSE for that (family, k).

    Saves under:
      artifacts/<Hz>Hz/overlays_allk/<family>/overlay_<family>_allk_<axis>.png
      artifacts/<Hz>Hz/overlays_allk/<family>/mse_summary.csv

    Skips gracefully if any (hz, family) has no artifacts.
    """
    ARCHS = ["cnn", "lstm", "transformer"]
    BASELINES = ["RLS", "RLSLpos"]
    AXES = ["gx", "gy", "gz"]
    Ks = sorted(list(cfg.horizons))

    # Collect available predictions for NNs
    nn_by_hz_family: dict[int, dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]] = {}
    for (hz, arch, k), _m in results.items():
        base = artifact_path(cfg.artifacts, hz, arch, k, cfg.history_map[k], "model.pt").parent
        pair = _load_pred_true(base)
        if pair is None:
            print(f"[overlays] [skip] missing pred/true for {hz}Hz {arch} k={k}")
            continue
        nn_by_hz_family.setdefault(hz, {})[(arch, k)] = pair

    # Collect available predictions for baselines
    bl_by_hz_family: dict[int, dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]] = {}
    if baselines:
        for (hz, algo, k), _m in baselines.items():
            base = artifact_path(cfg.artifacts, hz, algo, k, cfg.history_map[k], "pred.npy").parent
            pair = _load_pred_true(base)
            if pair is None:
                print(f"[overlays] [skip] missing pred/true for {hz}Hz {algo} k={k}")
                continue
            bl_by_hz_family.setdefault(hz, {})[(algo, k)] = pair

    # Union of families (NN + baselines)
    families = ARCHS + BASELINES

    # Where to save
    for hz in sorted(set(list(nn_by_hz_family.keys()) + list(bl_by_hz_family.keys()))):
        for family in families:
            # Gather all k for this (hz, family)
            entries: dict[int, tuple[np.ndarray, np.ndarray]] = {}

            # NNs
            for k in Ks:
                key = (family, k)
                pair = nn_by_hz_family.get(hz, {}).get(key, None)
                if pair is not None:
                    entries[k] = pair

            # Baselines
            if family in BASELINES:
                for k in Ks:
                    key = (family, k)
                    pair = bl_by_hz_family.get(hz, {}).get(key, None)
                    if pair is not None:
                        entries[k] = pair

            if not entries:
                continue  # nothing to plot for this family at this Hz

            # Determine common T across k (use shortest; also clamp to max_points)
            Ts = [min(len(p[0]), len(p[1])) for p in entries.values()]
            T = min(min(Ts), max_points) if Ts else 0
            if T <= 0:
                continue

            # Choose a single aligned truth reference (take the first k we found)
            first_k = sorted(entries.keys())[0]
            _, true_ref = entries[first_k]
            true_ref = true_ref[:T]

            outdir = (Path(cfg.artifacts) / f"{hz}Hz" / "overlays_allk" / family) if save_dir is None \
                else (Path(save_dir) / f"{hz}Hz" / "overlays_allk" / family)
            outdir.mkdir(parents=True, exist_ok=True)

            # Prepare MSE CSV rows
            mse_rows = []

            # Per-axis overlay
            for ax_i, ax_name in enumerate(AXES):
                plt.figure(figsize=(12, 5))
                plt.plot(np.arange(T), true_ref[:T, ax_i], label=f"true ({ax_name})", linewidth=1.8)

                for k in sorted(entries.keys()):
                    pred_k, true_k = entries[k]
                    # compute global MSE for this (family, k) over the shared window
                    mse_val = _mse_mean(pred_k[:T], true_k[:T])
                    mse_rows.append({"k": int(k), "axis": "all", "mse": float(mse_val), "T": int(T)})

                    plt.plot(np.arange(T), pred_k[:T, ax_i],
                             linestyle="--", linewidth=1.0,
                             label=f"{family}-k{k}")

                plt.title(f"{hz}Hz — {family} — overlay of all k — axis: {ax_name}")
                plt.xlabel("sample"); plt.ylabel("value")
                plt.legend(ncol=min(5, len(entries) + 1), fontsize=9)
                plt.tight_layout()

                fname = outdir / f"overlay_{family}_allk_{ax_name}.png"
                plt.savefig(fname, dpi=150)
                if show: plt.show()
                plt.close()

            # Write a small CSV with MSE per k (de-duplicated rows)
            if mse_rows:
                # keep unique k rows (global MSE is the same across axes by our definition here)
                by_k = {}
                for r in mse_rows:
                    by_k[r["k"]] = r  # last wins; same values anyway
                import pandas as pd
                df = pd.DataFrame(sorted(by_k.values(), key=lambda x: x["k"]))
                df.to_csv(outdir / "mse_summary.csv", index=False)
