import torch
def mae(pred, target): return torch.mean(torch.abs(pred - target), dim=0)
def rmse(pred, target): return torch.sqrt(torch.mean((pred - target)**2, dim=0))
def l2_mean(pred, target): return torch.mean(torch.linalg.norm(pred - target, dim=-1))
