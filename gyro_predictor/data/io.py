from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict
import json
import numpy as np
import pandas as pd


def _parse_calibration_json_from_calibration_file_row(df: pd.DataFrame) -> Dict:
    """
    Your files have a row where x == 'calibration_file' and y holds the JSON blob.
    We do NOT use the 'tag' for this; we look directly for that sentinel in column x.
    """
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("expected columns 'x' and 'y' for calibration row")

    cand = df[df["x"] == "calibration_file"]
    if cand.empty:
        raise ValueError("no row with x == 'calibration_file'")

    yval = cand.iloc[0]["y"]
    if pd.isna(yval) or yval == "":
        raise ValueError("calibration_file row found but y is empty")

    s = str(yval).strip()
    # handle doubled/wrapped quotes from CSV
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    s = s.replace('""', '"')

    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValueError(f"failed to parse calibration JSON from y: {e}")

    # Validate keys and shapes (each is a vector of length 9 or 3)
    for k, n in [("Sgyro1", 9), ("Sacc1", 9), ("b_gyro1", 3), ("b_acc1", 3)]:
        if k not in obj:
            raise ValueError(f"calibration JSON missing key '{k}'")
        if not isinstance(obj[k], (list, tuple)) or len(obj[k]) != n:
            raise ValueError(f"'{k}' must be length {n}")

    return obj


def load_folder(root: Path, hz: int) -> List[Path]:
    folder = root / f"epsilon_{hz}Hz"
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder {folder}")
    return sorted(folder.glob("*.csv"))


def load_and_calibrate_csv(path: Path, hz: int) -> pd.DataFrame:
    """
    STRICT: returns only calibrated data; raises on any issue.
    Output columns: timestamp[int64], gx gy gz ax ay az [float64].
    """
    # Read as strings to avoid dtype warnings from the calibration row
    df = pd.read_csv(path, dtype="string", low_memory=False, keep_default_na=False)

    # Require needed columns
    for c in ("tag", "timestamp", "x", "y", "z"):
        if c not in df.columns:
            raise ValueError(f"{path.name}: missing required column '{c}'")

    # --- Calibration (mandatory) ---
    cal = _parse_calibration_json_from_calibration_file_row(df)

    # Row-major reshape (your vectors are length-9)
    Sg = np.array(cal["Sgyro1"], dtype=np.float64).reshape(3, 3, order="C")
    Sa = np.array(cal["Sacc1"],  dtype=np.float64).reshape(3, 3, order="C")
    bg = np.array(cal["b_gyro1"], dtype=np.float64).reshape(3)
    ba = np.array(cal["b_acc1"],  dtype=np.float64).reshape(3)

    print(f"[calib] {path.name}: |Sg|={np.linalg.norm(Sg):.6g} |bg|={np.linalg.norm(bg):.6g} |Sa|={np.linalg.norm(Sa):.6g} |ba|={np.linalg.norm(ba):.6g}")

    # --- Acc/Gyro extraction and numeric coercion ---
    need_cols = ["timestamp", "x", "y", "z"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"{path.name}: missing required column '{c}'")

    acc = df[df["tag"] == "acc"][need_cols].copy()
    gyr = df[df["tag"] == "gyro"][need_cols].copy()
    if acc.empty or gyr.empty:
        missing = "acc" if acc.empty else ""
        missing += " & " if acc.empty and gyr.empty else ""
        missing += "gyro" if gyr.empty else ""
        raise ValueError(f"{path.name}: missing {missing} rows")

    for c in need_cols:
        acc[c] = pd.to_numeric(acc[c], errors="coerce")
        gyr[c] = pd.to_numeric(gyr[c], errors="coerce")
    acc.dropna(inplace=True)
    gyr.dropna(inplace=True)

    # Exact timestamp align (inner join)
    merged = pd.merge(gyr, acc, on="timestamp", suffixes=("_g", "_a"))
    if merged.empty:
        raise ValueError(f"{path.name}: no aligned timestamps between gyro/acc")

    # --- Apply calibration: calibrated = S @ (raw - b) ---
    G_raw = merged[["x_g", "y_g", "z_g"]].to_numpy(dtype=np.float64)
    A_raw = merged[["x_a", "y_a", "z_a"]].to_numpy(dtype=np.float64)

    G_cal = (Sg @ (G_raw - bg).T).T
    A_cal = (Sa @ (A_raw - ba).T).T

    out = pd.DataFrame({
        "timestamp": merged["timestamp"].astype("int64"),
        "gx": G_cal[:, 0], "gy": G_cal[:, 1], "gz": G_cal[:, 2],
        "ax": A_cal[:, 0], "ay": A_cal[:, 1], "az": A_cal[:, 2],
    }).reset_index(drop=True)

    if len(out):
        rx0, ry0, rz0 = G_raw[0]
        gx0, gy0, gz0 = out.loc[0, ["gx","gy","gz"]].to_numpy()
        print(f"[ok]  {path.name}: raw_g0=({rx0:.2f},{ry0:.2f},{rz0:.2f}) -> cal_g0=({gx0:.6f},{gy0:.6f},{gz0:.6f})")

    # Fail hard if any non-finite
    if not np.isfinite(out[["gx","gy","gz","ax","ay","az"]].to_numpy()).all():
        raise ValueError(f"{path.name}: non-finite values after calibration")

    return out
