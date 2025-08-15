import math, torch, torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:,:x.size(1)]
class TimeTransformer(nn.Module):
    def __init__(self, in_ch: int, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, out_dim=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.head = nn.Linear(d_model, out_dim)
    def forward(self, x):
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        return self.head(h[:,-1,:])
