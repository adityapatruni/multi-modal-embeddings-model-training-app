from __future__ import annotations

from typing import Dict, List

import torch


def collate_fn(batch: List[Dict[str, object]], pad_id: int) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    lengths = [len(b["input_ids"]) for b in batch]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        ids = torch.tensor(b["input_ids"], dtype=torch.long)
        input_ids[i, : ids.numel()] = ids
        attention_mask[i, : ids.numel()] = 1
    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
