from __future__ import annotations

import torch


def retrieval_recall_at_k(sim: torch.Tensor, k: int) -> float:
    # sim: [B, B] similarity matrix (image->text)
    b = sim.shape[0]
    ranks = torch.argsort(sim, dim=1, descending=True)
    targets = torch.arange(b, device=sim.device).unsqueeze(1)
    hit = (ranks[:, :k] == targets).any(dim=1).float().mean().item()
    return hit
