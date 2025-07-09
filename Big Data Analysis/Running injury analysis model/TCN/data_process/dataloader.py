import torch
from torch.utils.data import Dataset

class InjuryDataset(Dataset):
    def __init__(self, X_seq, X_weekly, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)   # [B, 28, 5]
        self.X_weekly = torch.tensor(X_weekly, dtype=torch.float32)  # [B, 维度]
        self.y = torch.tensor(y, dtype=torch.float32)           # [B]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_weekly[idx], self.y[idx]
