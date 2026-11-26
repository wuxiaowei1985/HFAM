import torch
import torch.nn as nn

class FocusAttentionBlock(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.2, mlp_ratio=4.0):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)
        hidden_dim = int(feature_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        normalized = self.norm1(x)
        x = x + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x

class FocusAttention(nn.Module):
    """
    Input: [B,T,343,512]
    Output:
        sequence: [B,T,343,512]
        focus_feature: [B,T,49,512]
        token_weight: [B,T,343]
        focus_weight: [B,T,7]
    """
    def __init__(self, feature_dim=512, num_heads=8, depth=2, num_focus=7, query_h=7, query_w=7, dropout=0.2):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.feature_dim = feature_dim
        self.num_focus = num_focus
        self.query_h = query_h
        self.query_w = query_w
        self.num_tokens = num_focus * query_h * query_w
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, feature_dim, num_focus, query_h, query_w))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder = nn.ModuleList([
            FocusAttentionBlock(feature_dim=feature_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(feature_dim)
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, x, return_sequence=False):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,T,N,D], got {tuple(x.shape)}")
        B, T, N, D = x.shape
        if N != self.num_tokens:
            raise ValueError(f"Expected N={self.num_tokens}, got {N}")
        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got {D}")
        x = x.view(B, T, self.num_focus, self.query_h, self.query_w, D)
        position = self.pos_embed.permute(0, 1, 3, 4, 5, 2)
        x = x + position
        x = x.reshape(B * T, self.num_tokens, D)
        for layer in self.encoder:
            x = layer(x)
        x = self.final_norm(x)
        token_score = self.score(x).squeeze(-1)
        token_weight = torch.softmax(token_score, dim=-1)
        token_weight_3d = token_weight.view(B * T, self.num_focus, self.query_h * self.query_w)
        focus_weight = token_weight_3d.sum(dim=-1)
        focus_weight = focus_weight / focus_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sequence = x.view(B, T, self.num_tokens, D)
        sequence_3d = sequence.view(B * T, self.num_focus, self.query_h * self.query_w, D)
        focus_weight_spatial = focus_weight.view(B * T, self.num_focus, 1, 1)
        focus_feature = torch.sum(sequence_3d * focus_weight_spatial, dim=1)
        focus_feature = focus_feature.view(B, T, self.query_h * self.query_w, D)
        focus_weight = focus_weight.view(B, T, self.num_focus)
        if return_sequence:
            return sequence, focus_feature, token_weight.view(B, T, self.num_tokens), focus_weight
        return focus_feature, focus_weight

if __name__ == "__main__":
    model = FocusAttention()
    x = torch.randn(2, 5, 343, 512)
    with torch.no_grad():
        sequence, feature, token_weight, focus_weight = model(x, return_sequence=True)
    print("sequence:", sequence.shape)
    print("feature:", feature.shape)
    print("token_weight:", token_weight.shape)
    print("focus_weight:", focus_weight.shape)