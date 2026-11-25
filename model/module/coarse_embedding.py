import torch
import torch.nn as nn

class CoarseEmbedding(nn.Module):
    """
    Input: coarse_probs [B,T,3]
    Learnable embedding: [3,T,49,512]
    Output: [B,T,49,512]
    """
    def __init__(self, num_classes=3, sequence_length=5, num_tokens=49, feature_dim=512, dropout=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self.class_embedding = nn.Parameter(torch.empty(num_classes, sequence_length, num_tokens, feature_dim))
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.class_embedding, std=0.02)

    def forward(self, coarse_probs):
        if coarse_probs.ndim != 3:
            raise ValueError(f"Expected [B,T,C], got {tuple(coarse_probs.shape)}")
        B, T, C = coarse_probs.shape
        if T != self.sequence_length:
            raise ValueError(f"Expected T={self.sequence_length}, got T={T}")
        if C != self.num_classes:
            raise ValueError(f"Expected C={self.num_classes}, got C={C}")
        embedding = torch.einsum("btc,ctnd->btnd", coarse_probs, self.class_embedding)
        embedding = self.norm(embedding)
        embedding = self.dropout(embedding)
        return embedding

if __name__ == "__main__":
    model = CoarseEmbedding()
    x = torch.softmax(torch.randn(2, 5, 3), dim=-1)
    with torch.no_grad():
        y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)