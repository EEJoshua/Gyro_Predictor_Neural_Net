from ..models.cnn import TimeCNN
from ..models.lstm import TimeLSTM
from ..models.transformer import TimeTransformer
def build_model(arch: str, in_ch: int):
    a = arch.lower()
    if a == "cnn": return TimeCNN(in_ch=in_ch, hidden=64, num_layers=3, out_dim=3)
    if a == "lstm": return TimeLSTM(in_ch=in_ch, hidden=128, num_layers=2, out_dim=3, dropout=0.0)
    if a == "transformer": return TimeTransformer(in_ch=in_ch, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, out_dim=3, dropout=0.1)
    raise ValueError(f"Unknown arch: {arch}")
