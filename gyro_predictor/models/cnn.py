import torch, torch.nn as nn
class TimeCNN(nn.Module):
    def __init__(self, in_ch: int, hidden: int=64, num_layers: int=3, out_dim: int=3):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(num_layers):
            layers += [nn.Conv1d(ch, hidden, 3, padding=1), nn.GELU(), nn.BatchNorm1d(hidden)]
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden, out_dim))
    def forward(self, x):  # [B,T,C]
        h = self.net(x.transpose(1,2))   # [B,H,T]
        return self.head(h)
