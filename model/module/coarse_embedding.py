import torch
import torch.nn as nn

class CoarseEmbedding(nn.Module):
    """
    Temporal embedding: [7B,5,49,512]
    Spatial embedding: [5B,7,49,512]
    """
    def __init__(self, sequence_length=5, num_focus=7, num_tokens=49, feature_dim=512, dropout=0.0):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_focus = num_focus
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self.temporal_embedding = nn.Parameter(torch.empty(1, sequence_length, num_tokens, feature_dim))
        self.spatial_embedding = nn.Parameter(torch.empty(1, num_focus, num_tokens, feature_dim))
        self.temporal_norm = nn.LayerNorm(feature_dim)
        self.spatial_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)

    def forward(self, batch_size, device=None, dtype=None):
        temporal_embedding = self.temporal_embedding.expand(batch_size * self.num_focus, -1, -1, -1)
        spatial_embedding = self.spatial_embedding.expand(batch_size * self.sequence_length, -1, -1, -1)
        if device is not None:
            temporal_embedding = temporal_embedding.to(device=device)
            spatial_embedding = spatial_embedding.to(device=device)
        if dtype is not None:
            temporal_embedding = temporal_embedding.to(dtype=dtype)
            spatial_embedding = spatial_embedding.to(dtype=dtype)
        temporal_embedding = self.dropout(self.temporal_norm(temporal_embedding))
        spatial_embedding = self.dropout(self.spatial_norm(spatial_embedding))
        return temporal_embedding, spatial_embedding