from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    data_root: Path = Path(".")                      # project root for data (expects ./epsilon_52Hz and ./epsilon_104Hz)
    artifacts: Path = Path("./artifacts")            # output dir for models, metrics, plots, caches (splits), etc.
    use_acc: bool = True                             # include accelerometer (ax, ay, az) alongside gyro inputs
    skip_uncalibrated: bool = True                   # strictly skip files without valid calibration (never use raw)
    jump_factor: float = 1.5                         # split where dt > jump_factor × nominal dt (handles BT time jumps)
    min_segment_len: int = 32                        # minimum contiguous segment length (in samples) kept after splitting
    hz_list: tuple = (52, 104)                       # sampling rates (Hz) to process; must match dataset folders
    horizons: tuple = (1,2,3,4)                      # prediction horizons k (steps ahead) to train/evaluate
    history_map: dict = field(                       # mapping k → history length H (past samples fed to the model)
        default_factory=lambda: {1:8, 2:10, 3:10, 4:12})
    split_ratios: tuple = (0.70, 0.15, 0.15)         # train/val/test split fractions applied per-Hz to segments
    seed: int = 42                                   # global RNG seed (splits, init, augmentation)
    epochs: int = 50                                 # max training epochs (early stopping may stop earlier)
    batch_size: int = 2048                           # DataLoader batch size (adjust to your VRAM)
    lr: float = 1e-3                                 # optimizer learning rate (Adam/AdamW)
    weight_decay: float = 1e-4                       # L2 weight decay (AdamW) strength
    early_patience: int = 5                          # stop after this many epochs with no val-loss improvement
    retrain: bool = False                            # if True, ignore cached weights/metrics and retrain from scratch
    
    num_workers: int = 8
    persistent_workers: bool = True                  # only takes effect if num_workers > 0
    prefetch_factor: int = 4                         # only used if num_workers > 0
    pin_memory: bool = True                          # CUDA input pipeline
    seed: int = 42                                   # for worker seeding
    