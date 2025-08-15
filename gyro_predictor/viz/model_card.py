# gyro_predictor/viz/model_card.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from gyro_predictor.train.factory import build_model


@dataclass
class ModelCardConfig:
    batch_size: int = 1              # use 1 to keep shapes exact and light
    num_workers: int = 0             # no workers to avoid Windows pickling issues here
    pin_memory: bool = False
    max_layers_in_bar: int = 20      # bar chart shows top-K layers by #params
    make_bar_plot: bool = True       # save per-arch bar plot of layer params
    show: bool = True                # show figures inline (notebook)


def _infer_in_ch_from_sample(x: torch.Tensor, default_in_ch: int) -> int:
    """
    Try to infer input channels from a sample tensor shape.
    Accepts (B, C, H) or (B, H, C). Fallback to default_in_ch.
    """
    if x.ndim == 3:
        b, d1, d2 = x.shape
        # common cases
        if d1 in (3, 6):
            return int(d1)
        if d2 in (3, 6):
            return int(d2)
    return int(default_in_ch)


def _get_sample_batch(dataset, device: torch.device, cfg_mc: ModelCardConfig) -> torch.Tensor:
    """Fetch a single batch (X only) from a dataset."""
    dl = DataLoader(dataset, batch_size=cfg_mc.batch_size, shuffle=False,
                    num_workers=cfg_mc.num_workers, pin_memory=cfg_mc.pin_memory,
                    drop_last=False)
    x, _ = next(iter(dl))
    return x.to(device)


def _collect_layer_summaries(model: torch.nn.Module, x: torch.Tensor) -> pd.DataFrame:
    """
    Register forward hooks on leaf modules, run a forward pass,
    and build a dataframe: name, type, out_shape, params, trainable.
    """
    # Map module -> dotted name; and register hooks only on leaves
    name_map: Dict[torch.nn.Module, str] = {}
    leaves: List[torch.nn.Module] = []
    for name, module in model.named_modules():
        # leaf if has no children
        if sum(1 for _ in module.children()) == 0:
            name_map[module] = name if name != "" else module.__class__.__name__
            leaves.append(module)

    outputs: Dict[str, Tuple[int, ...]] = {}

    def hook_fn(mod, inp, out):
        name = name_map.get(mod, mod.__class__.__name__)
        # record only tensor shapes; support tuple/list outs
        def to_shape(o):
            if torch.is_tensor(o):
                return tuple(int(d) for d in o.shape)
            return None
        shp = None
        if isinstance(out, (list, tuple)):
            for o in out:
                shp = to_shape(o)
                if shp is not None:
                    break
        else:
            shp = to_shape(out)
        if shp is not None:
            outputs[name] = shp

    hooks = [m.register_forward_hook(hook_fn) for m in leaves]
    model.eval()
    with torch.no_grad():
        _ = model(x)
    for h in hooks:
        h.remove()

    # Build dataframe
    rows = []
    for m in leaves:
        name = name_map[m]
        p = sum(p_.numel() for p_ in m.parameters())
        trainable = any(p_.requires_grad for p_ in m.parameters())
        rows.append(dict(
            name=name,
            type=m.__class__.__name__,
            out_shape=str(outputs.get(name, "")),
            params=int(p),
            trainable=bool(trainable),
        ))
    df = pd.DataFrame(rows)
    # totals row
    total_params = int(sum(r["params"] for r in rows))
    df_total = pd.DataFrame([dict(name="(total)", type="", out_shape="", params=total_params, trainable=True)])
    df = pd.concat([df, df_total], ignore_index=True)
    return df


def _save_bar_plot(df: pd.DataFrame, title: str, out_path: Path, top_k: int = 20, show: bool = True):
    """Bar chart of top-K layers by params (excludes the '(total)' row)."""
    d = df[df["name"] != "(total)"].sort_values("params", ascending=False).head(top_k)
    if d.empty:
        return
    plt.figure(figsize=(10, max(3, 0.35 * len(d))))
    plt.barh(d["name"], d["params"])
    plt.gca().invert_yaxis()
    plt.xlabel("# parameters"); plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show: plt.show()
    plt.close()


def render_model_summaries(
    cfg,
    dataset,
    device: torch.device,
    archs: List[str] = ("cnn", "lstm", "transformer"),
    save_dir: str | Path | None = None,
    cfg_mc: ModelCardConfig | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build each model (using cfg/use_acc to guess in_ch), run a forward with a real sample,
    and save a per-architecture table of layers and parameter counts.

    Returns dict: {arch: dataframe}
    """
    cfg_mc = cfg_mc or ModelCardConfig()
    out_root = Path(save_dir) if save_dir else (Path(cfg.artifacts) / "summaries" / "model_cards")
    out_root.mkdir(parents=True, exist_ok=True)

    # get one real sample (shape will drive correct model I/O)
    x = _get_sample_batch(dataset, device, cfg_mc)
    # infer channels from sample, with fallback to config
    default_in_ch = 6 if getattr(cfg, "use_acc", False) else 3
    in_ch = _infer_in_ch_from_sample(x, default_in_ch)

    print(f"[model-cards] using in_ch={in_ch} | sample shape={tuple(x.shape)}")

    dfs: Dict[str, pd.DataFrame] = {}
    for arch in archs:
        try:
            model = build_model(arch, in_ch=in_ch).to(device)
        except Exception as ex:
            print(f"[model-cards][skip] {arch}: build failed ({type(ex).__name__}: {ex})")
            continue

        # collect layer summaries
        df = _collect_layer_summaries(model, x)
        dfs[arch] = df

        # save CSV and pretty-print head
        csv_path = out_root / f"model_arch_{arch}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[model-cards] {arch}: layers={len(df)-1} total_params={int(df.iloc[-1]['params'])} -> {csv_path}")

        # save raw torch module string for reference
        txt_path = out_root / f"model_str_{arch}.txt"
        txt_path.write_text(str(model))
        print(f"[model-cards] {arch}: torch print -> {txt_path}")

        # optional bar plot of top layers by params
        if cfg_mc.make_bar_plot:
            png_path = out_root / f"model_arch_{arch}_top{cfg_mc.max_layers_in_bar}_layers.png"
            _save_bar_plot(df, f"{arch}: top-{cfg_mc.max_layers_in_bar} layers by #params", png_path, cfg_mc.max_layers_in_bar, cfg_mc.show)

    return dfs
