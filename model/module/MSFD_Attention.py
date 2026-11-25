import torch
import torch.nn as nn
from torch_ms_deform_attn import ms_deform_attn

class MSDeformableAttention(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.2):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = feature_dim // num_heads
        self.query_norm = nn.LayerNorm(feature_dim)
        self.value_proj = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(num_levels)
        ])
        self.sampling_offsets = nn.Linear(feature_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(feature_dim, num_heads * num_levels * num_points)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

    @staticmethod
    def _build_spatial_shapes(pyramid):
        return torch.tensor([[feature.shape[-2], feature.shape[-1]] for feature in pyramid], dtype=torch.long, device=pyramid[0].device)

    def _flatten_pyramid(self, pyramid):
        values = []
        spatial_shapes = []
        for level, feature in enumerate(pyramid):
            B, T, C, H, W = feature.shape
            spatial_shapes.append((H, W))
            value = feature.permute(0, 1, 3, 4, 2).contiguous()
            value = value.view(B * T, H * W, C)
            value = self.value_proj[level](value)
            value = value.view(B * T, H * W, self.num_heads, self.head_dim)
            values.append(value)
        value = torch.cat(values, dim=1)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=value.device)
        level_sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        level_start_index = torch.cat([torch.zeros(1, dtype=torch.long, device=value.device), level_sizes.cumsum(dim=0)[:-1]])
        return value, spatial_shapes, level_start_index

    def forward(self, query, pyramid, return_attention=False):
        if query.ndim != 4:
            raise ValueError(f"Expected query [B,T,N,D], got {tuple(query.shape)}")
        B, T, N, D = query.shape
        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got {D}")
        query_flat = query.reshape(B * T, N, D)
        query_norm = self.query_norm(query_flat)
        offsets = self.sampling_offsets(query_norm)
        offsets = offsets.view(B * T, N, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query_norm)
        attention_weights = attention_weights.view(B * T, N, self.num_heads, self.num_levels, self.num_points)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        reference_points = self._build_reference_points(N, query.device, query.dtype)
        reference_points = reference_points.view(1, N, 1, 1, 1, 2)
        spatial_shapes = self._build_spatial_shapes(pyramid)
        value, _, level_start_index = self._flatten_pyramid(pyramid)
        normalized_offsets = offsets.clone()
        for level in range(self.num_levels):
            H = spatial_shapes[level, 0]
            W = spatial_shapes[level, 1]
            normalized_offsets[:, :, :, level, :, 0] /= W
            normalized_offsets[:, :, :, level, :, 1] /= H
        sampling_locations = (reference_points + normalized_offsets).clamp(0.0, 1.0)
        output = ms_deform_attn(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        output = self.output_proj(output)
        output = self.dropout(output)
        output = output.view(B, T, N, D)
        if return_attention:
            return output, sampling_locations, attention_weights
        return output

    @staticmethod
    def _build_reference_points(num_tokens, device, dtype):
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise ValueError(f"num_tokens must be a square number, got {num_tokens}")
        y = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side
        x = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

class MSFDAttentionBlock(nn.Module):
    def __init__(self,feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attention = MSDeformableAttention(feature_dim=feature_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, pyramid, return_attention=False):
        normalized_query = self.norm1(query)
        if return_attention:
            attention_output, locations, weights = self.attention(normalized_query, pyramid, return_attention=True)
        else:
            attention_output = self.attention(normalized_query, pyramid, return_attention=False)
            locations = None
            weights = None
        query = query + attention_output
        query = query + self.ffn(self.norm2(query))
        if return_attention:
            return query, locations, weights
        return query

class MSFDAttention(nn.Module):
    """
    Query: [B,T,49,512]
    Pyramid:
        p2 [B,T,512,7,56,56]
        p3 [B,T,512,7,28,28]
        p4 [B,T,512,7,14,14]
    Focus weight: [B,T,7]
    Output: [B,T,49,512]
    """
    def __init__(self,feature_dim=512, num_heads=8, depth=2, num_levels=3, num_points=4, num_focus=7, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.depth = depth
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_focus = num_focus
        self.encoder = nn.ModuleList([
            MSFDAttentionBlock(feature_dim=feature_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(feature_dim)

    def _aggregate_focus(self, feature, focus_weight):
        if feature.ndim != 6:
            raise ValueError(f"Expected [B,T,D,F,H,W], got {tuple(feature.shape)}")
        B, T, D, F, H, W = feature.shape
        if F != self.num_focus:
            raise ValueError(f"Expected F={self.num_focus}, got {F}")
        if focus_weight.shape != (B, T, F):
            raise ValueError(f"Expected focus_weight [{B},{T},{F}], got {tuple(focus_weight.shape)}")
        weight = focus_weight.view(B, T, 1, F, 1, 1)
        return (feature * weight).sum(dim=3)

    def forward(self, query_tokens, pyramid, focus_weight, return_attention=False):
        if query_tokens.ndim != 4:
            raise ValueError(f"Expected [B,T,N,D], got {tuple(query_tokens.shape)}")
        aggregated_pyramid = [
            self._aggregate_focus(feature, focus_weight)
            for feature in pyramid
        ]
        x = query_tokens
        attention_outputs = []
        for block in self.encoder:
            if return_attention:
                x, locations, weights = block(x, aggregated_pyramid, return_attention=True)
                attention_outputs.append({"locations": locations, "weights": weights})
            else:
                x = block(x, aggregated_pyramid, return_attention=False)
        x = self.final_norm(x)
        if return_attention:
            return x, attention_outputs
        return x

if __name__ == "__main__":
    model = MSFDAttention()
    query = torch.randn(2, 5, 49, 512)
    pyramid = [
        torch.randn(2, 5, 512, 7, 56, 56),
        torch.randn(2, 5, 512, 7, 28, 28),
        torch.randn(2, 5, 512, 7, 14, 14)
    ]
    focus_weight = torch.softmax(torch.randn(2, 5, 7), dim=-1)
    with torch.no_grad():
        output = model(query, pyramid, focus_weight)
    print("query :", query.shape)
    print("output:", output.shape)