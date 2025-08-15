import numpy as np
import pandas as pd
from typing import List

def expected_dt_us(hz: int) -> int:
    return int(round(1e6 / hz))

def split_on_time_jumps(df: pd.DataFrame, hz: int, jump_factor: float = 1.5, min_len: int = 32) -> List[pd.DataFrame]:
    ts = df["timestamp"].values
    if len(ts) < min_len:
        return []
    dt = np.diff(ts)
    thr = jump_factor * expected_dt_us(hz)
    cuts = np.where(dt > thr)[0]
    starts = [0] + (cuts + 1).tolist()
    ends = (cuts + 1).tolist() + [len(df)]
    segs = []
    for s,e in zip(starts, ends):
        if e - s >= min_len:
            segs.append(df.iloc[s:e].reset_index(drop=True))
    return segs
