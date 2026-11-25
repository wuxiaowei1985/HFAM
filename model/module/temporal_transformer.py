import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    """
    Input: [B,T,49,D]
    Processing:
        [B,T,49,D] -> temporal position embedding + spatial position embedding -> [B,T*49,D] -> global self-attention -> [B,T,49,D]
    """
    def __init__(self,feature_dim=512, num_heads=8, depth=2, dropout=0.2, sequence_length=5, num_tokens=49):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.num_tokens = num_tokens
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, sequence_length, 1, feature_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, num_tokens, feature_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, norm=nn.LayerNorm(feature_dim))

    def forward(self, x, temporal_mask=None):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,T,N,D], got {tuple(x.shape)}")
        B, T, N, D = x.shape
        if T != self.sequence_length:
            raise ValueError(f"Expected T={self.sequence_length}, got T={T}")
        if N != self.num_tokens:
            raise ValueError(f"Expected N={self.num_tokens}, got N={N}")
        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got D={D}")
        x = x + self.temporal_pos_embed[:, :T] + self.spatial_pos_embed[:, :, :N]
        x = x.reshape(B, T * N, D)
        key_padding_mask = None
        if temporal_mask is not None:
            if temporal_mask.shape != (B, T):
                raise ValueError(f"Expected temporal_mask [{B},{T}], got {tuple(temporal_mask.shape)}")
            key_padding_mask = ~temporal_mask.bool()
            key_padding_mask = (key_padding_mask.unsqueeze(-1).expand(B, T, N).reshape(B, T * N))
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x.view(B, T, N, D)

if __name__ == "__main__":
    model = TemporalTransformer()
    x = torch.randn(2, 5, 49, 512)
    mask = torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        y = model(x, mask)
    print("input :", x.shape)
    print("flatten:", (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
    print("output:", y.shape)