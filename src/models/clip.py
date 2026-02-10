from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ImageEncoder, TextEncoder


class CLIPModel(nn.Module):
    def __init__(
        self,
        image_backbone: str,
        image_embed_dim: int,
        text_vocab_size: int,
        text_max_len: int,
        text_embed_dim: int,
        text_num_layers: int,
        text_num_heads: int,
        text_ffn_dim: int,
        text_dropout: float,
        projection_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(image_backbone, image_embed_dim)
        self.text_encoder = TextEncoder(
            vocab_size=text_vocab_size,
            max_len=text_max_len,
            embed_dim=text_embed_dim,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            ffn_dim=text_ffn_dim,
            dropout=text_dropout,
        )
        self.image_proj = nn.Linear(image_embed_dim, projection_dim)
        self.text_proj = nn.Linear(text_embed_dim, projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature))

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        image_emb = self.image_encoder(images)
        text_emb = self.text_encoder(input_ids, attention_mask)
        image_z = F.normalize(self.image_proj(image_emb), dim=-1)
        text_z = F.normalize(self.text_proj(text_emb), dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_z @ text_z.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2
