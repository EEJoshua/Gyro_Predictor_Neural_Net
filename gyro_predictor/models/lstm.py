import torch.nn as nn
class TimeLSTM(nn.Module):
    def __init__(self, in_ch: int, hidden: int=128, num_layers: int=2, out_dim: int=3, dropout: float=0.0):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=num_layers, dropout=dropout if num_layers>1 else 0.0, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):  # [B,T,C]
        out, _ = self.lstm(x)
        return self.head(out[:,-1,:])
