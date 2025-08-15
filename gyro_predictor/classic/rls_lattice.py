import numpy as np
from tqdm import tqdm

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

_EPS32 = np.float32(1e-8)

def _safe_div_num(a, b):
    return a / (b if abs(b) > _EPS32 else (_EPS32 if b >= 0 else -_EPS32))

def _rlsl_numpy(u: np.ndarray, d: np.ndarray, a: float, e: float, N: int,
                pred_forward: int, progress: bool, desc: str | None):
    u = np.asarray(u, np.float32).ravel()
    d = np.asarray(d, np.float32).ravel()
    T = len(u)
    out = np.zeros((T, pred_forward + 1), np.float32)

    Pf = e * (a ** (-2)) * np.ones((N,), np.float32)
    Pb = e * (a ** np.arange(-2, -N - 2, -1, dtype=np.float32))
    gconvp = np.ones((N + 1,), np.float32)
    bposp  = np.zeros((N + 1,), np.float32)
    ddot   = np.zeros((N,), np.float32)
    pdot   = np.zeros((N,), np.float32)

    gconv = np.ones((N + 1,), np.float32)
    bpos  = np.zeros((N + 1,), np.float32)
    fpos  = np.zeros((N + 1,), np.float32)
    rpos  = np.zeros((N + 1,), np.float32)
    kvec  = np.zeros((N,), np.float32)
    kf    = np.zeros((N,), np.float32)
    kb    = np.zeros((N,), np.float32)

    rng = tqdm(range(T), desc=desc or f"RLSL k={pred_forward}", ncols=0, leave=True) if progress else range(T)
    for i in rng:
        gconv[0] = 1.0
        bpos[0]  = u[i]
        fpos[0]  = u[i]
        rpos[0]  = d[i]

        for m in range(N):
            Pf[m] = a * Pf[m] + _safe_div_num(fpos[m] * fpos[m], gconvp[m])
            Pbp   = Pb[m]
            Pb[m] = a * Pb[m] + _safe_div_num(bpos[m] * bpos[m], gconv[m])
            ddot[m] = a * ddot[m] + _safe_div_num(fpos[m] * bposp[m], gconvp[m])
            pdot[m] = a * pdot[m] + _safe_div_num(rpos[m] * bpos[m], gconv[m])
            gconv[m + 1] = gconv[m] - _safe_div_num(bpos[m] * bpos[m], Pb[m])
            kb[m]  = _safe_div_num(ddot[m], Pf[m])
            kf[m]  = _safe_div_num(ddot[m], Pbp if abs(Pbp) > _EPS32 else (_EPS32 if Pbp >= 0 else -_EPS32))
            kvec[m]= _safe_div_num(pdot[m], Pb[m])
            bpos[m + 1] = bposp[m] - kb[m] * fpos[m]
            fpos[m + 1] = fpos[m] - kf[m] * bposp[m]
            rpos[m + 1] = rpos[m] - kvec[m] * bpos[m]

        gconvp[:] = gconv
        bposp[:]  = bpos
        denom = gconv[N] if abs(gconv[N]) > _EPS32 else (_EPS32 if gconv[N] >= 0 else -_EPS32)
        y_i = d[i] - rpos[N] / denom
        out[i, 0] = y_i

        # recursive multi-step
        y_next = y_i

        if pred_forward > 0:
            btemp = bpos.copy(); ftemp = fpos.copy()
            btemp[0] = y_next; ftemp[0] = y_next
            acc = 0.0
            for m in range(N):
                btemp[m + 1] = bposp[m] - kb[m] * ftemp[m]
                ftemp[m + 1] = fpos[m]  - kf[m] * bposp[m]
                acc += kvec[m] * btemp[m]
            out[i, 1] = acc
            y_next = acc

        # propagate for further steps
        gconv_pred = gconv.copy()
        bpos_pred  = bpos.copy()
        fpos_pred  = fpos.copy()
        rpos_pred  = rpos.copy()
        Pf_pred    = Pf.copy()
        gconvp_pred= gconvp.copy()
        Pb_pred    = Pb.copy()
        kf_pred    = kf.copy()
        kb_pred    = kb.copy()
        k_pred     = kvec.copy()
        bposp_pred = bposp.copy()
        ddot_pred  = ddot.copy()
        pdot_pred  = pdot.copy()

        for p in range(1, pred_forward):
            gconv_pred[0] = 1.0
            bpos_pred[0]  = y_next
            fpos_pred[0]  = y_next
            rpos_pred[0]  = y_next

            for m in range(N):
                Pf_pred[m] = a * Pf_pred[m] + _safe_div_num(fpos_pred[m] * fpos_pred[m], gconvp_pred[m])
                Pbp_pred   = Pb_pred[m]
                Pb_pred[m] = a * Pb_pred[m] + _safe_div_num(bpos_pred[m] * bpos_pred[m], gconv_pred[m])
                ddot_pred[m] = a * ddot_pred[m] + _safe_div_num(fpos_pred[m] * bposp_pred[m], gconvp_pred[m])
                pdot_pred[m] = a * pdot_pred[m] + _safe_div_num(rpos_pred[m] * bpos_pred[m], gconv_pred[m])
                gconv_pred[m + 1] = gconv_pred[m] - _safe_div_num(bpos_pred[m] * bpos_pred[m], Pb_pred[m])
                kb_pred[m] = _safe_div_num(ddot_pred[m], Pf_pred[m])
                kf_pred[m] = _safe_div_num(ddot_pred[m], Pbp_pred if abs(Pbp_pred) > _EPS32 else (_EPS32 if Pbp_pred >= 0 else -_EPS32))
                k_pred[m]  = _safe_div_num(pdot_pred[m], Pb_pred[m])
                bpos_pred[m + 1] = bposp_pred[m] - kb_pred[m] * fpos_pred[m]
                fpos_pred[m + 1] = fpos_pred[m] - kf_pred[m] * bposp_pred[m]
                rpos_pred[m + 1] = rpos_pred[m] - k_pred[m] * bpos_pred[m]

            gconvp_pred[:] = gconv_pred
            bposp_pred[:]  = bpos_pred

            btemp = bpos_pred.copy(); ftemp = fpos_pred.copy()
            btemp[0] = y_next; ftemp[0] = y_next
            acc = 0.0
            for m in range(N):
                btemp[m + 1] = bposp_pred[m] - kb_pred[m] * ftemp[m]
                ftemp[m + 1] = fpos_pred[m] - kf_pred[m] * bposp_pred[m]
                acc += k_pred[m] * btemp[m]
            out[i, p + 1] = acc
            y_next = acc

    return out

if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True, nogil=True)
    def _rlsl_numba_core(u, d, a, e, N, pred_forward):
        EPS = np.float32(1e-8)
        T = u.shape[0]
        out = np.zeros((T, pred_forward + 1), np.float32)

        Pf = e * (a ** (-2)) * np.ones((N,), np.float32)
        Pb = e * (a ** np.arange(-2, -N - 2, -1, dtype=np.float32))
        gconvp = np.ones((N + 1,), np.float32)
        bposp  = np.zeros((N + 1,), np.float32)
        ddot   = np.zeros((N,), np.float32)
        pdot   = np.zeros((N,), np.float32)

        gconv = np.ones((N + 1,), np.float32)
        bpos  = np.zeros((N + 1,), np.float32)
        fpos  = np.zeros((N + 1,), np.float32)
        rpos  = np.zeros((N + 1,), np.float32)
        kvec  = np.zeros((N,), np.float32)
        kf    = np.zeros((N,), np.float32)
        kb    = np.zeros((N,), np.float32)

        def sdiv(a, b):
            if b > 0:
                return a / (b if b > EPS else EPS)
            else:
                bb = -b
                return -a / (bb if bb > EPS else EPS)

        for i in range(T):
            gconv[0] = 1.0
            bpos[0]  = u[i]
            fpos[0]  = u[i]
            rpos[0]  = d[i]

            for m in range(N):
                Pf[m] = a * Pf[m] + sdiv(fpos[m] * fpos[m], gconvp[m])
                Pbp   = Pb[m]
                Pb[m] = a * Pb[m] + sdiv(bpos[m] * bpos[m], gconv[m])
                ddot[m] = a * ddot[m] + sdiv(fpos[m] * bposp[m], gconvp[m])
                pdot[m] = a * pdot[m] + sdiv(rpos[m] * bpos[m], gconv[m])
                gconv[m + 1] = gconv[m] - sdiv(bpos[m] * bpos[m], Pb[m])
                kb[m]  = sdiv(ddot[m], Pf[m])
                kf[m]  = sdiv(ddot[m], Pbp)
                kvec[m]= sdiv(pdot[m], Pb[m])
                bpos[m + 1] = bposp[m] - kb[m] * fpos[m]
                fpos[m + 1] = fpos[m] - kf[m] * bposp[m]
                rpos[m + 1] = rpos[m] - kvec[m] * bpos[m]

            for j in range(N + 1):
                gconvp[j] = gconv[j]
                bposp[j]  = bpos[j]

            denom = gconv[N]
            if denom >= 0:
                denom = denom if denom > EPS else EPS
            else:
                dd = -denom
                denom = -(dd if dd > EPS else EPS)

            y_i = d[i] - rpos[N] / denom
            out[i, 0] = y_i

            y_next = y_i

            if pred_forward > 0:
                btemp = bpos.copy(); ftemp = fpos.copy()
                btemp[0] = y_next; ftemp[0] = y_next
                acc = 0.0
                for m in range(N):
                    btemp[m + 1] = bposp[m] - kb[m] * ftemp[m]
                    ftemp[m + 1] = fpos[m]  - kf[m] * bposp[m]
                    acc += kvec[m] * btemp[m]
                out[i, 1] = acc
                y_next = acc

            gconv_pred = gconv.copy()
            bpos_pred  = bpos.copy()
            fpos_pred  = fpos.copy()
            rpos_pred  = rpos.copy()
            Pf_pred    = Pf.copy()
            gconvp_pred= gconvp.copy()
            Pb_pred    = Pb.copy()
            kf_pred    = kf.copy()
            kb_pred    = kb.copy()
            k_pred     = kvec.copy()
            bposp_pred = bposp.copy()
            ddot_pred  = ddot.copy()
            pdot_pred  = pdot.copy()

            for p in range(1, pred_forward):
                gconv_pred[0] = 1.0
                bpos_pred[0]  = y_next
                fpos_pred[0]  = y_next
                rpos_pred[0]  = y_next

                for m in range(N):
                    Pf_pred[m] = a * Pf_pred[m] + sdiv(fpos_pred[m] * fpos_pred[m], gconvp_pred[m])
                    Pbp_pred   = Pb_pred[m]
                    Pb_pred[m] = a * Pb_pred[m] + sdiv(bpos_pred[m] * bpos_pred[m], gconv_pred[m])
                    ddot_pred[m] = a * ddot_pred[m] + sdiv(fpos_pred[m] * bposp_pred[m], gconvp_pred[m])
                    pdot_pred[m] = a * pdot_pred[m] + sdiv(rpos_pred[m] * bpos_pred[m], gconv_pred[m])
                    gconv_pred[m + 1] = gconv_pred[m] - sdiv(bpos_pred[m] * bpos_pred[m], Pb_pred[m])
                    kb_pred[m] = sdiv(ddot_pred[m], Pf_pred[m])
                    kf_pred[m] = sdiv(ddot_pred[m], Pbp_pred)
                    k_pred[m]  = sdiv(pdot_pred[m], Pb_pred[m])
                    bpos_pred[m + 1] = bposp_pred[m] - kb_pred[m] * fpos_pred[m]
                    fpos_pred[m + 1] = fpos_pred[m] - kf_pred[m] * bposp_pred[m]
                    rpos_pred[m + 1] = rpos_pred[m] - k_pred[m] * bpos_pred[m]

                for j in range(N + 1):
                    gconvp_pred[j] = gconv_pred[j]
                    bposp_pred[j]  = bpos_pred[j]

                btemp = bpos_pred.copy(); ftemp = fpos_pred.copy()
                btemp[0] = y_next; ftemp[0] = y_next
                acc = 0.0
                for m in range(N):
                    btemp[m + 1] = bposp_pred[m] - kb_pred[m] * ftemp[m]
                    ftemp[m + 1] = fpos_pred[m] - kf_pred[m] * bposp_pred[m]
                    acc += k_pred[m] * btemp[m]
                out[i, p + 1] = acc
                y_next = acc

        return out

def rlslpos_predict(u: np.ndarray, d: np.ndarray, a: float, e: float, N: int,
                    pred_forward: int, progress: bool = False, desc: str | None = None,
                    backend: str = "numba"):
    u32 = np.asarray(u, np.float32).ravel()
    d32 = np.asarray(d, np.float32).ravel()

    if backend == "numba" and NUMBA_AVAILABLE:
        try:
            out = _rlsl_numba_core(u32, d32, np.float32(a), np.float32(e), N, pred_forward)
        except ZeroDivisionError:
            print("[rlslpos] numba ZeroDivisionError — falling back to NumPy.")
            out = _rlsl_numpy(u32, d32, a, e, N, pred_forward, progress, desc)
        except Exception as ex:
            print(f"[rlslpos] numba failed ({type(ex).__name__}: {ex}) — falling back to NumPy.")
            out = _rlsl_numpy(u32, d32, a, e, N, pred_forward, progress, desc)
        else:
            if (not np.isfinite(out).all()) or (float(np.abs(out).sum()) == 0.0):
                print("[rlslpos] numba output invalid/zero — falling back to NumPy.")
                out = _rlsl_numpy(u32, d32, a, e, N, pred_forward, progress, desc)
            elif progress:
                tqdm(total=1, desc=desc or f"RLSL k={pred_forward} (numba)", leave=False).update(1)
        return out

    return _rlsl_numpy(u32, d32, a, e, N, pred_forward, progress, desc)


def rlslpos_multiaxis(gyro: np.ndarray, a: float, e: float, N: int, pred_forward: int,
                      progress: bool = False, backend: str = "numba"):
    preds = {}
    T = int(gyro.shape[0])
    target_len = max(0, T - pred_forward)
    for j, ax in enumerate(["gx","gy","gz"]):
        u = gyro[:, j]
        out = rlslpos_predict(
            u, u, a, e, N, pred_forward,
            progress=progress, desc=f"RLSL {ax} k={pred_forward}", backend=backend
        )
        if out.shape[0] != target_len:
            out = out[:target_len, :]
        preds[ax] = out
    return preds

