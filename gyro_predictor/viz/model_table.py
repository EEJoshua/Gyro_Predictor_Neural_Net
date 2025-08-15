# gyro_predictor/viz/model_table.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from gyro_predictor.train.factory import build_model


@dataclass
class TableCfg:
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    save_csv: bool = True
    save_md: bool = False
    show_df: bool = True


# ---------- helpers ----------
def _get_one_batch(dataset, device: torch.device, cfg: TableCfg) -> torch.Tensor:
    dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                    drop_last=False)
    x, _ = next(iter(dl))
    return x.to(device)


def _infer_in_ch(x: torch.Tensor, default_in_ch: int) -> int:
    # accept (B, C, T) or (B, T, C)
    if x.ndim == 3:
        _, d1, d2 = x.shape
        if d1 in (3, 6): return int(d1)
        if d2 in (3, 6): return int(d2)
    return int(default_in_ch)


def _first_tensor(obj):
    """Return the first torch.Tensor inside obj (handles tuple/list/dict)."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for o in obj:
            if torch.is_tensor(o):
                return o
    if isinstance(obj, dict):
        for o in obj.values():
            if torch.is_tensor(o):
                return o
    return None


def _shape_ascii(t: torch.Tensor | None) -> str:
    """ASCII-only shape like 'BxCxT'. Empty if unknown."""
    if t is None:
        return ""
    try:
        dims = [int(d) for d in t.shape]
        return "x".join(str(d) for d in dims)  # ASCII 'x'
    except Exception:
        return ""


def _bct_columns(t: torch.Tensor | None) -> Tuple[Any, Any, Any]:
    """Return (B,C,T) if it looks like 3D [B,C,T] or [B,T,C]; else ('','','')."""
    if t is None or t.ndim < 2:
        return ("", "", "")
    B = int(t.shape[0])
    if t.ndim == 3:
        d1, d2 = int(t.shape[1]), int(t.shape[2])
        # heuristic: common channel counts
        if d1 in (1, 2, 3, 4, 6, 8, 16, 32, 64):  # treat as [B,C,T]
            return (B, d1, d2)
        if d2 in (1, 2, 3, 4, 6, 8, 16, 32, 64):  # treat as [B,T,C]
            return (B, d2, d1)
        return (B, d1, d2)
    return (B, int(t.shape[1]), "")


def _kernel_stride(mod: nn.Module) -> Tuple[str, str]:
    k = getattr(mod, "kernel_size", None)
    s = getattr(mod, "stride", None)
    if k is not None:
        k = "x".join(str(int(v)) for v in (k if isinstance(k, (tuple, list)) else (k,)))
    if s is not None:
        s = "x".join(str(int(v)) for v in (s if isinstance(s, (tuple, list)) else (s,)))
    return (k or "-", s or "-")


def _activation(mod: nn.Module) -> str:
    n = mod.__class__.__name__.lower()
    if "relu" in n: return "ReLU"
    if "gelu" in n: return "GELU"
    if "elu" in n:  return "ELU"
    if "leakyrelu" in n: return "LeakyReLU"
    if "sigmoid" in n: return "Sigmoid"
    if "tanh" in n: return "Tanh"
    if "softmax" in n: return "Softmax"
    return ""


def _notes(mod: nn.Module) -> str:
    if isinstance(mod, nn.LSTM):
        return f"hidden={mod.hidden_size}, layers={mod.num_layers}, bidir={mod.bidirectional}"
    if isinstance(mod, nn.Linear):
        return f"in={mod.in_features}, out={mod.out_features}"
    parts = []
    if hasattr(mod, "d_model"): parts.append(f"d_model={getattr(mod,'d_model')}")
    if hasattr(mod, "nhead"): parts.append(f"heads={getattr(mod,'nhead')}")
    if hasattr(mod, "dim_feedforward"): parts.append(f"ff={getattr(mod,'dim_feedforward')}")
    return ", ".join(parts)


def _short_alias(path: str) -> str:
    """Human-friendly leaf alias from dotted path."""
    if path == "" or path is None:
        return "(root)"
    return path.split(".")[-1]


# ---------- main API ----------
def render_layer_table_for_arch(
    cfg,
    dataset,
    device: torch.device,
    arch: str,
    save_dir: str | Path | None = None,
    tcfg: TableCfg | None = None,
) -> pd.DataFrame:
    """
    Build `arch`, run a forward pass, and return an ASCII table in **forward execution order** with:
    idx, path, name, type, out_shape, B, C, T, kernel, stride, activation, params, trainable, notes
    """
    tcfg = tcfg or TableCfg()
    out_dir = Path(save_dir) if save_dir else (Path(cfg.artifacts) / "summaries" / "model_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    # sample & model
    x = _get_one_batch(dataset, device, tcfg)
    in_ch = _infer_in_ch(x, 6 if getattr(cfg, "use_acc", False) else 3)
    model = build_model(arch, in_ch=in_ch).to(device).eval()

    # leaf modules (no children)
    leaves: List[Tuple[str, nn.Module]] = []
    for path, m in model.named_modules():
        if sum(1 for _ in m.children()) == 0:
            leaves.append((path, m))

    # forward hooks -> capture **call order** and outputs (forward execution order)
    calls: List[Dict[str, Any]] = []
    def _make_hook(path: str):
        def _hook(mod, inp, out):
            t = _first_tensor(out)
            calls.append({
                "path": path,
                "module": mod,
                "tensor": t.detach() if torch.is_tensor(t) else None,
            })
        return _hook

    hooks = [m.register_forward_hook(_make_hook(path)) for path, m in leaves]

    # run one dry forward; if shape orientation is wrong, try (B,C,T) <-> (B,T,C)
    with torch.no_grad():
        try:
            _ = model(x)
        except Exception:
            if x.ndim == 3:
                _ = model(x.permute(0, 2, 1))
            else:
                raise

    for h in hooks:
        h.remove()

    # rows in **forward execution order**
    rows = []
    for idx, call in enumerate(calls, start=1):
        path = call["path"]
        m: nn.Module = call["module"]
        t: torch.Tensor | None = call["tensor"]
        B, C, T = _bct_columns(t)
        p = sum(p_.numel() for p_ in m.parameters())
        trainable = any(p_.requires_grad for p_ in m.parameters())
        k, s = _kernel_stride(m)
        rows.append(dict(
            idx=idx,
            path=(path or "(root)"),
            name=_short_alias(path or ""),
            type=m.__class__.__name__,
            out_shape=_shape_ascii(t),
            B=B, C=C, T=T,
            kernel=k, stride=s,
            activation=_activation(m),
            params=int(p),
            trainable=bool(trainable),
            notes=_notes(m),
        ))

    # total row
    total_params = int(sum(r["params"] for r in rows))
    rows.append(dict(idx="(total)", path="", name="", type="", out_shape="",
                     B="", C="", T="", kernel="-", stride="-", activation="",
                     params=total_params, trainable=True, notes=""))

    cols = ["idx","path","name","type","out_shape","B","C","T","kernel","stride","activation","params","trainable","notes"]
    df = pd.DataFrame(rows, columns=cols)

    stem = f"{arch}_layers"
    if tcfg.save_csv:
        df.to_csv(out_dir / f"{stem}.csv", index=False)
    if tcfg.save_md:
        (out_dir / f"{stem}.md").write_text(df.to_markdown(index=False))

    if tcfg.show_df:
        print(f"[model-table] {arch}: {len(df)-1} layers, {total_params:,} params -> {out_dir/(stem+'.csv')}")
        print(df.head(min(12, len(df))))

    return df


def render_layer_tables(
    cfg,
    dataset,
    device: torch.device,
    archs: List[str] = ("cnn","lstm","transformer"),
    save_dir: str | Path | None = None,
    tcfg: TableCfg | None = None,
) -> Dict[str, pd.DataFrame]:
    res: Dict[str, pd.DataFrame] = {}
    for a in archs:
        try:
            res[a] = render_layer_table_for_arch(cfg, dataset, device, a, save_dir, tcfg)
        except Exception as ex:
            print(f"[model-table][skip] {a}: {type(ex).__name__}: {ex}")
    return res
