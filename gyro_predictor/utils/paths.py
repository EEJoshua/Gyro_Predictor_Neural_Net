from pathlib import Path
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path
def artifact_path(root: Path, hz: int, arch: str, k: int, history: int, tag: str):
    d = root / f"{hz}Hz" / arch / f"k{k}_h{history}"
    d.mkdir(parents=True, exist_ok=True)
    return d / tag
