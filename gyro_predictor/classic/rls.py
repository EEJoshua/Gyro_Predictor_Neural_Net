# gyro_predictor/classic/rls.py
from __future__ import annotations
import numpy as np
from tqdm import tqdm

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

_EPS = 1e-6

def _build_vectors(u: np.ndarray, i: int, N: int, decorrelate: bool):
    """
    Build v (current regressor) and curr (history used for k-step recursion),
    both length N. Index i is the current time (0..T-1).
    v[0]=u[i], v[1]=u[i-1], ...
    curr[0]=u[i], curr[1]=u[i-1], ...
    If decorrelate (match your MATLAB path for predIters==1), apply
    simple first-difference along the stack.
    """
    v = np.zeros(N, dtype=np.float64)
    c = np.zeros(N, dtype=np.float64)
    for p in range(N):
        idx = i - p
        val = u[idx] if idx >= 0 else 0.0
        if p == 0:
            v[p] = val
            c[p] = val
        else:
            if decorrelate:
                v[p] = val - v[p-1]
                c[p] = val - c[p-1]
            else:
                v[p] = val
                c[p] = val
    return v, c  # most-recent first

def _rls_numpy_stable(u: np.ndarray, lam: float, delta: float, N: int,
                      pred_forward: int, progress: bool, desc: str | None,
                      decorrelate_if_k1: bool):
    """
    Robust RLS with P0 = delta*I, PSD guard, symmetrization, and recursive k-step prediction.
    Returns (T - k, k+1) with out[t, p] predicting sample (t+p) for p>=1.
    """
    u = np.asarray(u, np.float64).ravel()
    T = len(u)
    k = int(pred_forward)
    out_len = max(0, T - k)
    if out_len == 0:
        return np.zeros((0, k+1), np.float64)

    N = int(N)
    lam = float(max(lam, _EPS))
    delta = float(max(delta, _EPS))
    decorrelate = bool(decorrelate_if_k1 and k == 1)

    # State
    w = np.zeros(N, dtype=np.float64)
    P = (delta) * np.eye(N, dtype=np.float64)   # PI = delta * I (MATLAB convention)

    out = np.zeros((out_len, k+1), dtype=np.float64)
    rng = tqdm(range(out_len), desc=desc or f"RLS k={k}", ncols=0, leave=True) if progress else range(out_len)

    for t in rng:
        # Build current regressor and "curr" stack (most-recent first)
        v, curr = _build_vectors(u, t, N, decorrelate)

        # One-step prediction BEFORE update (y_hat for sample t)
        y = float(v @ w)
        alpha = u[t] - y                   # a.k.a. e(t)
        Pv = P @ v
        vPv = float(v @ Pv)                # should be >=0 for PSD P

        # PSD guard
        if vPv < 0.0:
            P = (delta) * np.eye(N, dtype=np.float64)
            Pv = delta * v
            vPv = float((v @ v) * delta)

        denom = lam + vPv
        if abs(denom) < _EPS:
            denom = _EPS
        g = Pv / denom                     # K(t)

        # Update
        w = w + g * alpha
        P = (P - np.outer(g, Pv)) / lam
        # Symmetrize to fight drift
        P = 0.5 * (P + P.T)

        # Save y at p=0 (diagnostic)
        out[t, 0] = y

        # Recursive k-step ahead: reg = [preds(reversed); curr[..]]
        # pred_list holds p=1..k predictions
        pred_list = []
        for p in range(1, k+1):
            reg = np.empty(N, dtype=np.float64)
            # fill previous predictions in reverse order
            npreds = len(pred_list)
            for m in range(npreds):
                reg[m] = pred_list[npreds - 1 - m]
            # fill the rest from curr (skip as many as we already placed)
            remain = N - npreds
            reg[npreds:] = curr[:remain]
            y_next = float(reg @ w)
            pred_list.append(y_next)
            out[t, p] = y_next

    return out

# -------- Optional numba kernel (kept simple); we still wrap with try/except and fallback --------
if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, nogil=True)
    def _rls_numba_core(u, lam, delta, N, k):
        EPS = 1e-8
        T = u.shape[0]
        out_len = T - k if T - k > 0 else 0
        out = np.zeros((out_len, k+1), np.float64)
        w = np.zeros((N,), np.float64)
        P = (delta) * np.eye(N, dtype=np.float64)
        for t in range(out_len):
            # Build v,curr (no decorrelation path in numba for simplicity)
            v = np.zeros((N,), np.float64)
            curr = np.zeros((N,), np.float64)
            for p in range(N):
                idx = t - p
                val = u[idx] if idx >= 0 else 0.0
                v[p] = val
                curr[p] = val

            y = 0.0
            for j in range(N):
                y += v[j] * w[j]
            alpha = u[t] - y

            # Pv = P @ v
            Pv = np.zeros((N,), np.float64)
            for r in range(N):
                s = 0.0
                for c in range(N):
                    s += P[r, c] * v[c]
                Pv[r] = s
            vPv = 0.0
            for j in range(N):
                vPv += v[j] * Pv[j]
            if vPv < 0.0:
                for r in range(N):
                    for c in range(N):
                        P[r, c] = 0.0
                for j in range(N):
                    P[j, j] = delta
                for j in range(N):
                    Pv[j] = delta * v[j]
                vPv = 0.0
                for j in range(N):
                    vPv += v[j] * v[j]
                vPv *= delta

            denom = lam + vPv
            if abs(denom) < EPS:
                denom = EPS
            g = np.zeros((N,), np.float64)
            for j in range(N):
                g[j] = Pv[j] / denom

            # w update
            for j in range(N):
                w[j] += g[j] * alpha

            # P update (and symmetrize crudely)
            for r in range(N):
                for c in range(N):
                    P[r, c] = (P[r, c] - g[r] * Pv[c]) / lam
            for r in range(N):
                for c in range(r+1, N):
                    av = 0.5 * (P[r, c] + P[c, r])
                    P[r, c] = av
                    P[c, r] = av

            out[t, 0] = y

            # recursive predictions
            pred = np.zeros((k,), np.float64)
            for p in range(1, k+1):
                npreds = p - 1
                # reg = [preds(reversed); curr[..]]
                # compute y_next
                y_next = 0.0
                # preds
                for m in range(npreds):
                    y_next += pred[npreds - 1 - m] * w[m]
                # remaining from curr
                for m in range(npreds, N):
                    y_next += curr[m - npreds] * w[m]
                pred[p-1] = y_next
                out[t, p] = y_next

        return out

def rls_predict(u: np.ndarray, d_unused: np.ndarray, a: float, e: float, N: int,
                pred_forward: int, progress: bool = False, desc: str | None = None,
                backend: str = "numba", decorrelate_if_k1: bool = True):
    """
    Robust RLS. 'd_unused' kept for API compatibility; the algorithm self-predicts u.
    """
    lam, delta = float(a), float(e)
    u = np.asarray(u, np.float64).ravel()
    if backend == "numba" and NUMBA_AVAILABLE:
        try:
            out = _rls_numba_core(u, np.float64(max(lam, _EPS)), np.float64(max(delta, _EPS)), int(N), int(pred_forward))
        except Exception as ex:
            print(f"[rls] numba failed ({type(ex).__name__}: {ex}) — falling back to NumPy.")
            out = _rls_numpy_stable(u, lam, delta, N, pred_forward, progress, desc, decorrelate_if_k1)
        else:
            if (not np.isfinite(out).all()) or (float(np.abs(out).sum()) == 0.0):
                print("[rls] numba output invalid/zero — falling back to NumPy.")
                out = _rls_numpy_stable(u, lam, delta, N, pred_forward, progress, desc, decorrelate_if_k1)
            elif progress:
                tqdm(total=1, desc=desc or f"RLS k={pred_forward} (numba)", leave=False).update(1)
        return out
    # NumPy robust path
    return _rls_numpy_stable(u, lam, delta, N, pred_forward, progress, desc, decorrelate_if_k1)

def rls_multiaxis(gyro: np.ndarray, a: float, e: float, N: int, pred_forward: int,
                  progress: bool = False, backend: str = "numba"):
    preds = {}
    T = int(gyro.shape[0])
    target_len = max(0, T - pred_forward)          # enforce T-k rows
    for j, ax in enumerate(["gx", "gy", "gz"]):
        u = np.asarray(gyro[:, j], np.float64)
        out = rls_predict(
            u, u, a, e, N, pred_forward,
            progress=progress, desc=f"RLS {ax} k={pred_forward}", backend=backend,
            decorrelate_if_k1=True
        )
        # --- enforce consistent length (T-k) ---
        if out.shape[0] != target_len:
            out = out[:target_len, :]
        preds[ax] = out
    return preds

