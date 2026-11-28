import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    """
    Input: [7B,5*49,512]
    Output: [7B,5*49,512]
    """
    def __init__(self, feature_dim=512, num_heads=4, depth=1, dropout=0.1, sequence_length=5, num_tokens=49):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.num_tokens = num_tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 2, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, norm=nn.LayerNorm(feature_dim))

    def forward(self, x, temporal_mask=None):
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T*N,D], got {tuple(x.shape)}")
        B, L, D = x.shape
        expected_length = self.sequence_length * self.num_tokens
        if L != expected_length:
            raise ValueError(f"Expected sequence length {expected_length}, got {L}")
        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got {D}")
        key_padding_mask = None
        if temporal_mask is not None:
            if temporal_mask.ndim != 2:
                raise ValueError(f"Expected temporal_mask [B_original,T], got {tuple(temporal_mask.shape)}")
            key_padding_mask = (~temporal_mask.bool()).unsqueeze(-1).expand(-1, -1, self.num_tokens).reshape(B, expected_length)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

if __name__ == "__main__":
    model = TemporalTransformer()
    x = torch.randn(14, 245, 512)
    mask = torch.ones(14, 5, dtype=torch.bool)
    with torch.no_grad():
        y = model(x, mask)
    print("input :", x.shape)
    print("output:", y.shape)