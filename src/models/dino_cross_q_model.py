import torch
import torch.nn.functional as F
from torch import nn

from transformers import Dinov2Model



class _CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q = self.norm1(query)
        attn_out, _ = self.attn(q, key, value, need_weights=False)
        x = query + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class DinoCrossAttentionQNetwork(nn.Module):
    def __init__(
        self,
        action_dim: int = 6,
        num_outputs: int = 5,
        dino_model_name: str = "facebook/dinov2-base",
        image_size: int = 224,
        embed_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.0,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        # DINOv2는 ImageNet normalization 사용가정해서
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.image_encoder = Dinov2Model.from_pretrained(dino_model_name)
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

        dino_dim = self.image_encoder.config.hidden_size
        self.action_embed = nn.Linear(action_dim, embed_dim)
        self.k_proj = nn.Linear(dino_dim, embed_dim)
        self.v_proj = nn.Linear(dino_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, head_hidden_dim),
                    nn.GELU(),
                    nn.Linear(head_hidden_dim, 1),
                )
                for _ in range(num_outputs)
            ]
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        return self

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        image = (image - self.pixel_mean) / self.pixel_std
        return image

    def forward(self, image: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        image = self._preprocess_image(image)
        with torch.no_grad():
            image_tokens = self.image_encoder(pixel_values=image).last_hidden_state[:, 1:, :]

        key = self.k_proj(image_tokens)
        value = self.v_proj(image_tokens)
        query = self.action_embed(action).unsqueeze(1)

        for block in self.blocks:
            query = block(query, key, value)

        fused = query.squeeze(1)
        scores = [head(fused) for head in self.score_heads]
        return torch.cat(scores, dim=1)
