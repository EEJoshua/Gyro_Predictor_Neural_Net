# gyro_predictor/classic/baseline_runner.py
from __future__ import annotations

import os, json, time, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from gyro_predictor.utils.paths import artifact_path
from gyro_predictor.classic.rls import rls_multiaxis
from gyro_predictor.classic.rls_lattice import rlslpos_multiaxis


@dataclass
class RLSConfig:
    a: float = 0.99         # forgetting factor
    e: float = 1e-2         # regularization (delta)
    N: int = 8              # filter/lattice order
    use_numba: bool = True  # use numba backends if available
    max_workers: int | None = None           # thread workers across (hz, k)
    show_algo_inner_bars: bool = False       # inner per-iteration bars (very chatty)
    # If cfg.retrain exists (main Config), honor it; otherwise use this:
    retrain_override: bool | None = None     # None => follow cfg.retrain if present


def _data_hash(arr: np.ndarray) -> str:
    h = hashlib.blake2b(arr.tobytes(), digest_size=16)
    return h.hexdigest()


def _pack_axis_k(preds_dict: dict, k: int) -> np.ndarray:
    # Stack gx,gy,gz from column 'k' of each axis output -> (T,3)
    return np.stack([preds_dict[ax][:, k] for ax in ("gx", "gy", "gz")], axis=1)


def _compute_metrics(P3: np.ndarray, true: np.ndarray) -> dict:
    mae = np.mean(np.abs(P3 - true), axis=0)
    rmse = np.sqrt(np.mean((P3 - true) ** 2, axis=0))
    l2 = np.mean(np.linalg.norm(P3 - true, axis=1))
    return {"mae": mae.tolist(), "rmse": rmse.tolist(), "l2_mean": float(l2), "n_samples": int(len(P3))}


def run_rls_baselines(cfg, splits: dict, rls_cfg: RLSConfig = RLSConfig()) -> Dict[Tuple[int, str, int], dict]:
    """
    Run (or load) RLS and RLS-Lattice baselines for all Hz and horizons in cfg,
    in parallel, with caching under artifacts. Returns a dict:
        baselines[(hz, algo_name, k)] = metrics_dict

    Args:
      cfg: your global Config (must have fields: artifacts, horizons, history_map, hz_list, retrain?)
      splits: dict like {hz: (train_segments, val_segments, test_segments)}
      rls_cfg: RLSConfig with hyperparams & performance knobs
    """
    # resolve workers
    max_workers = rls_cfg.max_workers
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 8)

    # figure out whether to recompute
    if rls_cfg.retrain_override is None:
        do_retrain = bool(getattr(cfg, "retrain", False))
    else:
        do_retrain = bool(rls_cfg.retrain_override)

    # build tasks & gather test data per Hz
    tasks: list[Tuple[int, int]] = []
    gyro_by_hz: dict[int, tuple[np.ndarray, str]] = {}

    for hz in cfg.hz_list:
        tr, va, te = splits[hz]
        if not te:
            print(f"[warn] {hz}Hz: no test segments, skipping baselines.")
            continue
        gyro_all = np.concatenate([seg[["gx", "gy", "gz"]].to_numpy(np.float32) for seg in te], axis=0)
        gyro_by_hz[hz] = (gyro_all, _data_hash(gyro_all))
        for k in cfg.horizons:
            tasks.append((hz, k))

    # inner task (one hz,k) – computes or loads both RLS & RLSLpos
    def baseline_task(hz: int, k: int, gyro_all: np.ndarray, dh: str):
        H = cfg.history_map[k]
        T = int(gyro_all.shape[0])
        out_len = max(0, T - k)
        out_metrics: dict[str, dict] = {}

        for algo_name, runner in [("RLS", rls_multiaxis), ("RLSLpos", rlslpos_multiaxis)]:
            base = artifact_path(cfg.artifacts, hz, algo_name, k, H, "pred.npy").parent
            base.mkdir(parents=True, exist_ok=True)
            meta_path = base / "meta.json"
            metrics_path = base / "metrics.json"
            pred_path = base / "pred.npy"
            true_path = base / "true.npy"

            # cache valid?
            valid = (
                meta_path.exists() and pred_path.exists() and true_path.exists() and metrics_path.exists()
                and json.loads(meta_path.read_text()).get("data_hash", "") == dh
                and json.loads(meta_path.read_text()).get("a") == rls_cfg.a
                and json.loads(meta_path.read_text()).get("e") == rls_cfg.e
                and json.loads(meta_path.read_text()).get("N") == rls_cfg.N
            )

            if valid and not do_retrain:
                print(f"[baseline cache] {algo_name} {hz}Hz k={k} | T={T} out_len={out_len}")
                out_metrics[algo_name] = json.loads(metrics_path.read_text())
                continue

            # compute fresh
            print(f"[baseline run]   {algo_name} {hz}Hz k={k} | T={T} out_len={out_len} N={rls_cfg.N} a={rls_cfg.a} e={rls_cfg.e}")
            t0 = time.perf_counter()
            preds_dict = runner(
                gyro_all, a=rls_cfg.a, e=rls_cfg.e, N=rls_cfg.N, pred_forward=k,
                progress=rls_cfg.show_algo_inner_bars,
                backend=("numba" if rls_cfg.use_numba else "numpy"),
            )
            dur = time.perf_counter() - t0

            true = gyro_all[k:]                           # (T-k, 3) target
            P3   = _pack_axis_k(preds_dict, k)            # (?, 3) from algo

            # ---- robust length alignment (handles stray T-length outputs) ----
            if P3.shape[0] != true.shape[0]:
                L = min(P3.shape[0], true.shape[0])
                print(f"[warn] {algo_name} {hz}Hz k={k}: pred/true len mismatch "
                    f"({P3.shape[0]} vs {true.shape[0]}). Trimming to {L}.")
                P3  = P3[:L]
                true = true[:L]

            metrics = _compute_metrics(P3, true)
            metrics["runtime_s"] = round(dur, 3)
            # write both keys for compatibility with earlier plotting cells
            thr = (out_len / dur) if dur > 0 else float("inf")
            metrics["throughput_sps"] = round(thr, 1)
            metrics["throughput_samples_per_s"] = metrics["throughput_sps"]
            metrics["latency_ms_per_sample"] = (1000.0 / thr) if np.isfinite(thr) and thr > 0 else float("nan")

            # save artifacts
            np.save(pred_path, P3)
            np.save(true_path, true)
            meta = {
                "algo": algo_name, "hz": int(hz), "k": int(k),
                "N": int(rls_cfg.N), "a": float(rls_cfg.a), "e": float(rls_cfg.e),
                "data_hash": dh
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            metrics_path.write_text(json.dumps(metrics, indent=2))

            print(f"[baseline done]  {algo_name} {hz}Hz k={k} | {dur:.2f}s  ~{metrics['throughput_sps']} samples/s")
            out_metrics[algo_name] = metrics

        return hz, k, out_metrics

    # execute in threads
    baselines: Dict[Tuple[int, str, int], dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for hz, k in tasks:
            gyro_all, dh = gyro_by_hz[hz]
            futs.append(ex.submit(baseline_task, hz, k, gyro_all, dh))

        for f in tqdm(as_completed(futs), total=len(futs), desc="Baselines (RLS + RLSLpos)"):
            hz, k, metrics_by_algo = f.result()
            for algo_name, metrics in metrics_by_algo.items():
                baselines[(hz, algo_name, k)] = metrics

    print("[done] Baselines ready & cached.")
    return baselines
