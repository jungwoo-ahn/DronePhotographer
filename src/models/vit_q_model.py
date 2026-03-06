import torch
from torch import nn


def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sincos embedding")
    y = torch.arange(h, device=device).float()
    x = torch.arange(w, device=device).float()
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(dim // 4, device=device).float() / (dim // 4)
    omega = 1.0 / (10000 ** omega)
    out_x = torch.einsum("hw,d->hwd", grid_x, omega)
    out_y = torch.einsum("hw,d->hwd", grid_y, omega)
    pos = torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=-1)
    return pos.view(h * w, dim)


class VisionTransformerQNetwork(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        action_dim: int = 6,
        num_outputs: int = 5,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            image_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.action_embed = nn.Linear(action_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, num_outputs),
        )

    def forward(self, image: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(image)
        b, c, h, w = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        pos = _build_2d_sincos_pos_embed(h, w, self.embed_dim, patches.device)
        pos = pos.unsqueeze(0).expand(b, -1, -1)
        patches = patches + pos

        action_token = self.action_embed(action).unsqueeze(1)
        tokens = torch.cat([action_token, patches], dim=1)
        encoded = self.encoder(tokens)
        fused = encoded[:, 0]
        return self.head(fused)
