import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    # Simple implementation of triplet loss
    def __init__(self, m=1.0):
        super().__init__()
        self.m = m

    def forward(self, a, p, n):
        dists = torch.sum((a-p)**2, dim=1) - torch.sum((a-n)**2, dim=1) + self.m
        # Only sum positive values
        return torch.sum(dists[dists > 0]) / a.shape[0]
