
import numpy as np
import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, segments, history: int, k_ahead: int, use_acc: bool=True):
        self.X = []
        self.y = []
        for seg in segments:
            gx = seg[["gx","gy","gz"]].to_numpy(dtype=np.float32)
            ax = seg[["ax","ay","az"]].to_numpy(dtype=np.float32)
            feats = np.concatenate([gx, ax], axis=1) if use_acc else gx
            T = len(seg)
            for t in range(history, T - k_ahead):
                self.X.append(feats[t-history:t])
                self.y.append(gx[t + k_ahead])
        if self.X:
            self.X = np.stack(self.X, axis=0).astype(np.float32)
            self.y = np.stack(self.y, axis=0).astype(np.float32)
        else:
            C = 6 if use_acc else 3
            self.X = np.zeros((0, history, C), np.float32)
            self.y = np.zeros((0, 3), np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): 
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])
