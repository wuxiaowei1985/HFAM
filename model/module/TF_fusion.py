import torch
import torch.nn as nn

class TFFusionBlock(nn.Module):
    def __init__(self, feature_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.norm_q = nn.LayerNorm(feature_dim)
        self.norm_kv = nn.LayerNorm(feature_dim)
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value, key_padding_mask=None):
        kv = self.norm_kv(key_value)
        fused, _ = self.cross_attention(self.norm_q(query), kv, kv, key_padding_mask=key_padding_mask, need_weights=False,)
        query = query + fused
        query = query + self.ffn(self.norm_ffn(query))
        return query

class TF_fusion(nn.Module):
    """
    Spatial branch: [5B,7*49,512]
    Temporal branch: [7B,5*49,512]
    Output:
        frame_features:  [B,5,512]
        sample_feature:  [B,512]
    """
    def __init__(self, feature_dim=512, num_heads=4, depth=2, sequence_length=5, num_focus=7, num_tokens=49, dropout=0.1):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        if depth < 1:
            raise ValueError("depth must be >= 1.")
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.num_focus = num_focus
        self.num_tokens = num_tokens
        self.depth = depth
        self.layers = nn.ModuleList([
            TFFusionBlock(feature_dim=feature_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(feature_dim)
        self.token_score = nn.Linear(feature_dim, 1)
        self.frame_score = nn.Linear(feature_dim, 1)

    def forward(self, spatial_feature, temporal_feature, temporal_mask=None):
        if spatial_feature.ndim != 3:
            raise ValueError(f"Expected spatial [BT,7*49,D], got {tuple(spatial_feature.shape)}")
        if temporal_feature.ndim != 3:
            raise ValueError(f"Expected temporal [BF,5*49,D], got {tuple(temporal_feature.shape)}")
        BT, spatial_tokens, D = spatial_feature.shape
        BF, temporal_tokens, Dt = temporal_feature.shape
        if D != self.feature_dim or Dt != self.feature_dim:
            raise ValueError("Invalid feature dimension.")
        if spatial_tokens != self.num_focus * self.num_tokens:
            raise ValueError(f"Expected spatial tokens={self.num_focus * self.num_tokens}, got {spatial_tokens}")
        if temporal_tokens != self.sequence_length * self.num_tokens:
            raise ValueError(f"Expected temporal tokens={self.sequence_length * self.num_tokens}, got {temporal_tokens}")
        if BT % self.sequence_length != 0:
            raise ValueError("Spatial batch dimension must be divisible by sequence_length.")
        B = BT // self.sequence_length
        if BF != B * self.num_focus:
            raise ValueError("Spatial and temporal batch dimensions are inconsistent.")
        spatial = spatial_feature.view(B, self.sequence_length, self.num_focus, self.num_tokens, D)
        temporal = temporal_feature.view(B, self.num_focus, self.sequence_length, self.num_tokens, D)
        query = spatial.permute(0, 2, 1, 3, 4).reshape(B * self.num_focus * self.sequence_length, self.num_tokens, D)
        key_value = temporal.reshape(B * self.num_focus, self.sequence_length * self.num_tokens, D)
        key_value = key_value.unsqueeze(1).expand(B * self.num_focus, self.sequence_length, self.sequence_length * self.num_tokens, D)
        key_value = key_value.reshape(B * self.num_focus * self.sequence_length, self.sequence_length * self.num_tokens, D)
        key_padding_mask = None
        if temporal_mask is not None:
            if temporal_mask.shape != (B, self.sequence_length):
                raise ValueError(f"Expected temporal_mask [{B},{self.sequence_length}], got {tuple(temporal_mask.shape)}")
            invalid = (~temporal_mask.bool()).unsqueeze(1).expand(B, self.num_focus, self.sequence_length)
            invalid = invalid.unsqueeze(-1).expand(B, self.num_focus, self.sequence_length, self.num_tokens)
            invalid = invalid.reshape(B * self.num_focus, self.sequence_length * self.num_tokens)
            key_padding_mask = invalid.unsqueeze(1).expand(B * self.num_focus, self.sequence_length, self.sequence_length * self.num_tokens)
            key_padding_mask = key_padding_mask.reshape(B * self.num_focus * self.sequence_length, self.sequence_length * self.num_tokens)
        fused = query
        for layer in self.layers:
            fused = layer(fused, key_value, key_padding_mask=key_padding_mask)
        fused = self.final_norm(fused)
        fused = fused.view(B, self.num_focus, self.sequence_length, self.num_tokens, D).permute(0, 2, 1, 3, 4).contiguous()
        token_score = self.token_score(fused).squeeze(-1).reshape(B, self.sequence_length, self.num_focus * self.num_tokens)
        token_weight = torch.softmax(token_score, dim=-1)
        frame_features = (fused.reshape(B, self.sequence_length, self.num_focus * self.num_tokens, D) * token_weight.unsqueeze(-1)).sum(dim=2)
        frame_score = self.frame_score(frame_features).squeeze(-1)
        if temporal_mask is not None:
            frame_score = frame_score.masked_fill(~temporal_mask.bool(), float("-inf"))
        frame_weight = torch.softmax(frame_score, dim=1)
        sample_feature = (frame_features * frame_weight.unsqueeze(-1)).sum(dim=1)
        return frame_features, sample_feature, token_weight, frame_weight

if __name__ == "__main__":
    model = TF_fusion(depth=2)
    spatial = torch.randn(10, 343, 512)
    temporal = torch.randn(14, 245, 512)
    mask = torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        frame, sample, token_weight, frame_weight = model(spatial, temporal, mask)
    print("frame :", frame.shape)
    print("sample:", sample.shape)