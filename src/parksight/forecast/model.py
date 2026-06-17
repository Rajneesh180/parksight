from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class SpatioTemporalGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, gru_hidden: int = 32):
        super().__init__()
        self.gcn = GCNConv(in_dim, hidden)
        self.gru = nn.GRU(hidden, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, 1)

    def forward(
        self, sequence: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        steps = sequence.shape[0]
        spatial = [
            F.relu(self.gcn(sequence[t], edge_index, edge_weight)) for t in range(steps)
        ]
        stacked = torch.stack(spatial, dim=1)
        recurrent, _ = self.gru(stacked)
        return self.head(recurrent[:, -1, :]).squeeze(-1)


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, gru_hidden: int = 32):
        super().__init__()
        self.encode = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, 1)

    def forward(
        self, sequence: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        steps = sequence.shape[0]
        encoded = [F.relu(self.encode(sequence[t])) for t in range(steps)]
        stacked = torch.stack(encoded, dim=1)
        recurrent, _ = self.gru(stacked)
        return self.head(recurrent[:, -1, :]).squeeze(-1)
