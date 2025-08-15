import numpy as np
class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=(0,1), keepdims=True)
        self.std = x.std(axis=(0,1), keepdims=True) + 1e-8
        return self
    def transform(self, x: np.ndarray):
        return (x - self.mean) / self.std
    def inverse_transform(self, x: np.ndarray):
        return x * self.std + self.mean
