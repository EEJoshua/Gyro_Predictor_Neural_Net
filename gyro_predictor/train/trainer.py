import time, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics import mae, rmse, l2_mean
import random
import numpy as np

# ---- picklable worker init (Windows-safe) ----
_WORKER_BASE_SEED = 42

def set_worker_seed_base(seed: int) -> None:
    global _WORKER_BASE_SEED
    _WORKER_BASE_SEED = int(seed)

def worker_init_fn(worker_id: int) -> None:
    s = _WORKER_BASE_SEED + int(worker_id)
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)

def _pinmem(device, user_flag: bool | None):
    if user_flag is not None:
        return bool(user_flag)
    return (device.type == "cuda")

def train_model(
        model,
        train_ds,
        val_ds,
        device,
        epochs=20,
        batch_size=256,
        lr=1e-3,
        weight_decay=0.0,
        early_patience=5,
        use_amp=True,
        *,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=None,
        seed=42,
    ):
    # perf toggles (optional)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    # set seed for workers (Windows-safe)
    set_worker_seed_base(int(seed))

    pm = _pinmem(device, pin_memory)
    pw = bool(persistent_workers and num_workers > 0)

    dl_kwargs_common = dict(
        num_workers=int(num_workers),
        pin_memory=pm,
        persistent_workers=pw,
        worker_init_fn=worker_init_fn,
    )
    if num_workers > 0:
        dl_kwargs_common["prefetch_factor"] = int(prefetch_factor)
    
    model = model.to(device)
    tr_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **dl_kwargs_common,
    )
    va_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **dl_kwargs_common,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    # NOTE: positional device string for broad torch version compatibility
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    best_val = float("inf")
    best_state = None
    patience = early_patience
    history = []

    # one progress bar that updates per epoch (single line)
    progress = tqdm(range(1, epochs + 1), desc="epochs", ncols=0, leave=True)
    for epoch in progress:
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for x, y in tr_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                pred = model(x)
                loss = torch.nn.functional.mse_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item() * len(x)

        train_loss /= max(1, len(train_ds))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in va_loader:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                pred = model(x)
                val_loss += torch.nn.functional.mse_loss(pred, y).item() * len(x)

        val_loss /= max(1, len(val_ds))
        scheduler.step(val_loss)

        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        # update the same line in-place
        progress.set_postfix_str(f"train={train_loss:.6f} val={val_loss:.6f} lr={lr_now:.2e} {dt:.1f}s")

        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss, "lr": lr_now, "time_s": dt})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = early_patience
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

def evaluate(
        model,
        ds,
        device,
        measure_latency=False,
        *,
        batch_size=4096,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=None,
        seed=42,
    ):
    set_worker_seed_base(int(seed))

    pm = _pinmem(device, pin_memory)
    pw = bool(persistent_workers and num_workers > 0)

    dl_kwargs_common = dict(
        num_workers=int(num_workers),
        pin_memory=pm,
        persistent_workers=pw,
        worker_init_fn=worker_init_fn,
    )
    if num_workers > 0:
        dl_kwargs_common["prefetch_factor"] = int(prefetch_factor)
    
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **dl_kwargs_common,
    )
    model = model.to(device).eval()

    total = 0
    all_pred, all_true = [], []

    if measure_latency and device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    t0 = time.time()
    with torch.no_grad():
        if measure_latency and device.type == "cuda":
            start.record()

        for x, y in tqdm(loader, desc="evaluate", ncols=0, leave=False):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            pred = model(x)
            all_pred.append(pred.cpu())
            all_true.append(y.cpu())
            total += len(x)

        if measure_latency and device.type == "cuda":
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            latency_ms = ms / max(1, total)
            throughput = 1000.0 / latency_ms if latency_ms > 0 else float('inf')
        else:
            dt = time.time() - t0
            latency_ms = (dt * 1000.0) / max(1, total)
            throughput = 1000.0 / latency_ms if latency_ms > 0 else float('inf')

    pred = torch.cat(all_pred, dim=0) if all_pred else torch.zeros(0, 3)
    true = torch.cat(all_true, dim=0) if all_true else torch.zeros(0, 3)

    m = {
        "mae": mae(pred, true).tolist(),
        "rmse": rmse(pred, true).tolist(),
        "l2_mean": float(l2_mean(pred, true)),
        "latency_ms_per_sample": float(latency_ms),
        "throughput_samples_per_s": float(throughput),
        "n_samples": int(total),
    }
    return m, pred, true
