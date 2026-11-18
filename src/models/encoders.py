from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str, embed_dim: int):
        super().__init__()
        if backbone == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            in_dim = model.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images).flatten(1)
        emb = self.proj(feats)
        return F.normalize(emb, dim=-1)


class TextTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.ln(x)
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.transformer = TextTransformerEncoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.transformer(input_ids, attention_mask)
        # CLS-style pool: use first token
        pooled = x[:, 0]
        emb = self.proj(pooled)
        return F.normalize(emb, dim=-1)
