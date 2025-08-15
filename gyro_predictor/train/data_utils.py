from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from ..data.io import load_folder, load_and_calibrate_csv
from ..data.segment import split_on_time_jumps
from ..data.dataset import SeqDataset
from ..data.normalize import Standardizer

import json
import hashlib
from datetime import datetime

def _segment_signature(seg) -> dict:
    """
    A quick signature to detect if segments changed between runs.
    Uses first/last timestamp-ish column if present + length.
    """
    cols = seg.columns
    ts_col = next((c for c in ("timestamp", "t", "t_us", "time_us") if c in cols), None)
    if ts_col is not None and len(seg) > 0:
        first_ts = int(seg[ts_col].iloc[0])
        last_ts  = int(seg[ts_col].iloc[-1])
    else:
        first_ts = 0
        last_ts  = 0
    return {"len": int(len(seg)), "first": first_ts, "last": last_ts}

def _segments_digest(segments) -> str:
    # Stable digest of the list of signatures
    sigs = [ _segment_signature(seg) for seg in segments ]
    blob = json.dumps(sigs, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()

def split_segments_cached(
    segments,
    *,
    ratios=(0.7, 0.15, 0.15),
    seed=42,
    cache_file: Path,
    force_resplit: bool = False,
):
    """
    Deterministically split once and reuse on subsequent runs.
    - Writes/reads JSON with indices and a digest of segments.
    - If the digest mismatches or force_resplit=True, recomputes and overwrites.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # current segments digest
    cur_digest = _segments_digest(segments)

    if (not force_resplit) and cache_file.exists():
        try:
            obj = json.loads(cache_file.read_text())
            if obj.get("digest") == cur_digest:
                idx = obj["indices"]
                tr = [segments[i] for i in idx["train"]]
                va = [segments[i] for i in idx["val"]]
                te = [segments[i] for i in idx["test"]]
                print(f"[split-cache] loaded {cache_file.name}: "
                      f"{len(tr)}/{len(va)}/{len(te)} (n={len(segments)})")
                return tr, va, te
            else:
                print(f"[split-cache] digest changed -> recompute ({cache_file.name})")
        except Exception as e:
            print(f"[split-cache] ignore corrupt cache ({cache_file.name}): {e}")

    # fallback: compute fresh
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(segments))
    rng.shuffle(idx_all)
    n = len(idx_all)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    tr_idx = idx_all[:n_tr].tolist()
    va_idx = idx_all[n_tr:n_tr + n_va].tolist()
    te_idx = idx_all[n_tr + n_va:].tolist()

    tr = [segments[i] for i in tr_idx]
    va = [segments[i] for i in va_idx]
    te = [segments[i] for i in te_idx]

    # save JSON
    payload = {
        "created": datetime.utcnow().isoformat() + "Z",
        "ratios": ratios,
        "seed": int(seed),
        "n_segments": int(len(segments)),
        "indices": {"train": tr_idx, "val": va_idx, "test": te_idx},
        "digest": cur_digest,
    }
    cache_file.write_text(json.dumps(payload, indent=2))
    print(f"[split-cache] saved {cache_file.name}: "
          f"{len(tr)}/{len(va)}/{len(te)} (n={len(segments)})")
    return tr, va, te

def collect_segments(data_root: Path, hz: int, use_acc: bool, skip_uncalibrated: bool,
                     jump_factor: float, min_segment_len: int):
    paths = load_folder(data_root, hz)
    segments = []
    bad = 0
    for p in paths:
        try:
            df = load_and_calibrate_csv(p, hz)  # raises on any calibration/align issue
        except Exception as e:
            bad += 1
            print(f"[skip] {p.name}: {e}")
            continue
        segs = split_on_time_jumps(df, hz, jump_factor=jump_factor, min_len=min_segment_len)
        if not segs:
            print(f"[skip] {p.name}: no segments after time-jump split")
        segments.extend(segs)
    print(f"[summary] {hz}Hz: {len(segments)} segments from {len(paths)} files ({bad} skipped)")
    if len(segments) == 0:
        raise RuntimeError(f"No usable calibrated segments for {hz}Hz.")
    return segments



def split_segments(segments, ratios=(0.7,0.15,0.15), seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(segments)); rng.shuffle(idx)
    n=len(idx); n_tr=int(ratios[0]*n); n_va=int(ratios[1]*n)
    tr = [segments[i] for i in idx[:n_tr]]
    va = [segments[i] for i in idx[n_tr:n_tr+n_va]]
    te = [segments[i] for i in idx[n_tr+n_va:]]
    return tr, va, te

def make_datasets(tr, va, te, history: int, k: int, use_acc: bool=True):
    tr_ds = SeqDataset(tr, history=history, k_ahead=k, use_acc=use_acc)
    va_ds = SeqDataset(va, history=history, k_ahead=k, use_acc=use_acc)
    te_ds = SeqDataset(te, history=history, k_ahead=k, use_acc=use_acc)
    C = tr_ds.X.shape[-1] if len(tr_ds)>0 else (6 if use_acc else 3)
    stdz = Standardizer().fit(tr_ds.X if len(tr_ds)>0 else np.zeros((1, history, C)))
    tr_ds.X = stdz.transform(tr_ds.X); va_ds.X = stdz.transform(va_ds.X); te_ds.X = stdz.transform(te_ds.X)
    return tr_ds, va_ds, te_ds, stdz
