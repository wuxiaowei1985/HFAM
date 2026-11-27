import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDeformableAttention3D(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.2):
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
        grid = locations[..., [2, 1, 0]]
        grid = grid * 2.0 - 1.0
        grid = grid.reshape(B * heads, N * P, 1, 1, 3)
        sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.reshape(B, heads, Dh, N, P)
        return sampled.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, query, pyramid, reference_points, return_attention=False):
        if query.ndim != 4:
            raise ValueError(f"Expected query [B,T,N,D], got {tuple(query.shape)}")
        B, T, N, D = query.shape
        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got {D}")
        if len(pyramid) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} pyramid levels, got {len(pyramid)}")
        for level, feature in enumerate(pyramid):
            if feature.ndim != 6:
                raise ValueError(f"Pyramid level {level} must be [B,T,D,F,H,W], got {tuple(feature.shape)}")
            if feature.shape[:3] != (B, T, D):
                raise ValueError(f"Pyramid level {level} has invalid shape {tuple(feature.shape)}")
        if reference_points.shape != (N, 3):
            raise ValueError(f"Expected reference_points [{N},3], got {tuple(reference_points.shape)}")
        BT = B * T
        query_flat = query.reshape(BT, N, D)
        offsets = self.sampling_offsets(query_flat)
        offsets = offsets.view(BT, N, self.num_heads, self.num_levels, self.num_points, 3)
        offsets = offsets.permute(0, 2, 1, 3, 4, 5).contiguous()
        weights = self.attention_weights(query_flat)
        weights = weights.view(BT, N, self.num_heads, self.num_levels * self.num_points)
        weights = weights.permute(0, 2, 1, 3).contiguous()
        weights = torch.softmax(weights, dim=-1)
        weights = weights.view(BT, self.num_heads, N, self.num_levels, self.num_points)
        ref = reference_points.to(device=query.device, dtype=query.dtype)
        ref = ref.view(1, 1, N, 1, 3)
        output = query_flat.new_zeros(BT, self.num_heads, N, self.head_dim)
        attention_maps = []
        for level, feature in enumerate(pyramid):
            feature = feature.reshape(BT, D, feature.shape[3], feature.shape[4], feature.shape[5])
            value = self.value_proj[level](feature)
            _, _, Fz, Hy, Wx = value.shape
            value = value.view(BT, self.num_heads, self.head_dim, Fz, Hy, Wx)
            step = query_flat.new_tensor([1.0 / max(Fz, 1), 1.0 / max(Hy, 1), 1.0 / max(Wx, 1)])
            level_offset = torch.tanh(offsets[:, :, :, level]) * (2.0 * step)
            locations = (ref + level_offset).clamp(0.0, 1.0)
            sampled = self._sample_level(value, locations)
            weight = weights[:, :, :, level].unsqueeze(-1)
            output = output + (sampled * weight).sum(dim=3)
            if return_attention:
                attention_maps.append(weights[:, :, :, level])
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(BT, N, D)
        output = self.output_proj(output)
        output = self.dropout(output)
        output = output.view(B, T, N, D)
        if return_attention:
            return output, attention_maps
        return output

class Deformable3DBlock(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.2):
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

    def forward(self, query, pyramid, reference_points, return_attention=False):
        normalized_query = self.norm1(query)
        if return_attention:
            attn_out, attention = self.attn(normalized_query, pyramid, reference_points, return_attention=True)
        else:
            attn_out = self.attn(normalized_query, pyramid, reference_points, return_attention=False)
            attention = None
        query = query + attn_out
        query = query + self.ffn(self.norm2(query))
        return query, attention

class MSFDAttention(nn.Module):
    """
    输入:
        query_tokens: [B,T,343,512]
        pyramid:
            p2: [B,T,512,7,56,56]
            p3: [B,T,512,7,28,28]
            p4: [B,T,512,7,14,14]
    输出:
        fused_feature: [B,T,49,512]
        focus_weight: [B,T,7]
    处理逻辑:
        1. 每一个时间帧独立执行原始单帧 MSFD。
        2. 保持原始 3D deformable attention，不引入 temporal attention。
        3. MSFD 输出 [B,T,343,512]。
        4. 将 343 = 7 × 7 × 7 重排为 [B,T,7,49,512]。
        5. 仿照 FocusAttention，在 7 个焦平面之间进行 attention fusion。
        6. 最终得到 [B,T,49,512]，交给 TemporalTransformer。
    """
    def __init__(self, feature_dim=512, num_heads=8, depth=2, num_levels=3, num_points=4, num_focus=7, query_h=7, query_w=7, dropout=0.2):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.depth = depth
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_focus = num_focus
        self.query_h = query_h
        self.query_w = query_w
        self.num_tokens = num_focus * query_h * query_w
        self.encoder = nn.ModuleList([
            Deformable3DBlock(feature_dim=feature_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(feature_dim)
        self.score = nn.Linear(feature_dim, 1)
        self.register_buffer("reference_points", self._build_reference_points(), persistent=False)

    def _build_reference_points(self):
        points = []
        for f in range(self.num_focus):
            z = (f + 0.5) / self.num_focus
            for y in range(self.query_h):
                yy = (y + 0.5) / self.query_h
                for x in range(self.query_w):
                    xx = (x + 0.5) / self.query_w
                    points.append((z, yy, xx))
        return torch.tensor(points, dtype=torch.float32)

    def forward(self, query_tokens, pyramid, return_attention=False):
        if query_tokens.ndim != 4:
            raise ValueError(f"Expected query [B,T,N,D], got {tuple(query_tokens.shape)}")
        B, T, N, D = query_tokens.shape
        if N != self.num_tokens:
            raise ValueError(f"Expected N={self.num_tokens}, got {N}")
        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got {D}")
        if len(pyramid) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} pyramid levels, got {len(pyramid)}")
        for level, feature in enumerate(pyramid):
            if feature.ndim != 6:
                raise ValueError(f"Pyramid level {level} must be [B,T,D,F,H,W], got {tuple(feature.shape)}")
            if feature.shape[0] != B or feature.shape[1] != T or feature.shape[2] != D:
                raise ValueError(f"Pyramid level {level} shape mismatch: expected B={B}, T={T}, D={D}, got {tuple(feature.shape)}")
        x = query_tokens
        attention_maps = []
        for block in self.encoder:
            x, attention = block( x, pyramid, self.reference_points, return_attention=return_attention)
            if return_attention:
                attention_maps.append(attention)
        x = self.final_norm(x)
        # --------------------------------------------------
        # MSFD 原始输出:
        # [B,T,343,512]
        # 343 = 7 focus planes × 7 × 7 spatial tokens
        # --------------------------------------------------
        x = x.view(B, T, self.num_focus, self.query_h * self.query_w, D)
        # --------------------------------------------------
        # 仿照 FocusAttention:
        # 先计算每一个 343 token 的重要性
        # --------------------------------------------------
        token_score = self.score(x).squeeze(-1)
        # [B,T,7,49]
        token_weight = torch.softmax(token_score, dim=2)
        # --------------------------------------------------
        # 对每一个空间位置分别融合7个焦平面
        # [B,T,7,49,512] × [B,T,7,49,1] -> [B,T,49,512]
        # --------------------------------------------------
        focus_feature = torch.sum(x * token_weight.unsqueeze(-1), dim=2)
        # --------------------------------------------------
        # 每个时间帧的7个焦平面整体权重
        # 用于分析/可视化
        # --------------------------------------------------
        focus_weight = token_weight.mean(dim=-1)
        if return_attention:
            return focus_feature, focus_weight, attention_maps
        return focus_feature

if __name__ == "__main__":
    model = MSFDAttention(feature_dim=512, num_heads=8, depth=2, num_levels=3, num_points=4, num_focus=7, query_h=7, query_w=7, dropout=0.2)
    B = 2
    T = 5
    query = torch.randn(B, T, 343, 512)
    pyramid = [
        torch.randn(B, T, 512, 7, 56, 56),
        torch.randn(B, T, 512, 7, 28, 28),
        torch.randn(B, T, 512, 7, 14, 14)
    ]
    with torch.no_grad():
        feature, focus_weight, attention = model(query, pyramid, return_attention=True)
    print("Input:", query.shape)
    print("Output:", feature.shape)
    print("Focus weight:", focus_weight.shape)
    print("Expected output: [B,T,49,512]")