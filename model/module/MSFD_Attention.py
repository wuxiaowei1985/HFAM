import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDeformableAttention3D(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.1):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = feature_dim // num_heads
        self.sampling_offsets = nn.Linear(feature_dim, num_heads * num_levels * num_points * 3)
        self.attention_weights = nn.Linear(feature_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.ModuleList([nn.Conv3d(feature_dim, feature_dim, kernel_size=1, bias=False) for _ in range(num_levels)])
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

    @staticmethod
    def _sample_level(value, locations):
        B, heads, Dh, Fz, Hy, Wx = value.shape
        _, _, N, P, _ = locations.shape
        value = value.reshape(B * heads, Dh, Fz, Hy, Wx)
        grid = locations[..., [2, 1, 0]] * 2.0 - 1.0
        grid = grid.reshape(B * heads, N * P, 1, 1, 3)
        sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.reshape(B, heads, Dh, N, P)
        return sampled.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, query, pyramid, reference_points):
        if query.ndim != 3:
            raise ValueError(f"Expected query [B,N,D], got {tuple(query.shape)}")
        B, N, D = query.shape
        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got {D}")
        query = self.output_proj(query)
        query_flat = query
        offsets = self.sampling_offsets(query_flat).view(B, N, self.num_heads, self.num_levels, self.num_points, 3).permute(0, 2, 1, 3, 4, 5).contiguous()
        weights = self.attention_weights(query_flat).view(B, N, self.num_heads, self.num_levels * self.num_points).permute(0, 2, 1, 3).contiguous()
        weights = torch.softmax(weights, dim=-1).view(B, self.num_heads, N, self.num_levels, self.num_points)
        ref = reference_points.to(device=query.device, dtype=query.dtype).view(1, 1, N, 1, 3)
        output = query.new_zeros(B, self.num_heads, N, self.head_dim)
        for level, feature in enumerate(pyramid):
            if feature.ndim != 5:
                raise ValueError(f"Pyramid level {level} must be [B,D,F,H,W], got {tuple(feature.shape)}")
            value = self.value_proj[level](feature)
            _, _, Fz, Hy, Wx = value.shape
            value = value.view(B, self.num_heads, self.head_dim, Fz, Hy, Wx)
            step = query.new_tensor([1.0 / max(Fz, 1), 1.0 / max(Hy, 1), 1.0 / max(Wx, 1)])
            level_offset = torch.tanh(offsets[:, :, :, level]) * (2.0 * step)
            locations = (ref + level_offset).clamp(0.0, 1.0)
            sampled = self._sample_level(value, locations)
            output = output + (sampled * weights[:, :, :, level].unsqueeze(-1)).sum(dim=3)
        output = output.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.dropout(output)

class Deformable3DBlock(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = MultiScaleDeformableAttention3D(feature_dim=feature_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, pyramid, reference_points):
        query = query + self.attn(self.norm1(query), pyramid, reference_points)
        query = query + self.ffn(self.norm2(query))
        return query

class MSFDAttention(nn.Module):
    """
    Input:
        query: [B,7*49,512]
        pyramid:
            [B,512,7,56,56]
            [B,512,7,28,28]
            [B,512,7,14,14]

    Output: [B,7*49,512]
    """
    def __init__(self, feature_dim=512, num_heads=8, depth=1, num_levels=3, num_points=4, num_focus=7, query_h=7, query_w=7, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_focus = num_focus
        self.query_h = query_h
        self.query_w = query_w
        self.num_tokens = num_focus * query_h * query_w
        self.encoder = nn.ModuleList([
            Deformable3DBlock(feature_dim=feature_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(feature_dim)
        self.register_buffer("reference_points", self._build_reference_points(), persistent=False)

    def _build_reference_points(self):
        points = []
        for f in range(self.num_focus):
            z = (f + 0.5) / self.num_focus
            for y in range(self.query_h):
                for x in range(self.query_w):
                    points.append((z, (y + 0.5) / self.query_h, (x + 0.5) / self.query_w))
        return torch.tensor(points, dtype=torch.float32)

    def forward(self, query, pyramid):
        if query.ndim != 3:
            raise ValueError(f"Expected [B,N,D], got {tuple(query.shape)}")
        if query.shape[1:] != (self.num_tokens, self.feature_dim):
            raise ValueError(f"Expected [B,{self.num_tokens},{self.feature_dim}], got {tuple(query.shape)}")
        x = query
        for block in self.encoder:
            x = block(x, pyramid, self.reference_points)
        return self.final_norm(x)

if __name__ == "__main__":
    model = MSFDAttention()
    query = torch.randn(2, 343, 512)
    pyramid = [
        torch.randn(2, 512, 7, 56, 56),
        torch.randn(2, 512, 7, 28, 28),
        torch.randn(2, 512, 7, 14, 14)
    ]
    with torch.no_grad():
        y = model(query, pyramid)
    print("Input :", query.shape)
    print("Output:", y.shape)