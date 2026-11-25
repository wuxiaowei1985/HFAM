import torch
import torch.nn as nn
from efficient_kan import KAN

class ClassificationHead(nn.Module):
    """
    Input: [...,512]
    Output: [...,16]
    """
    def __init__(self, in_features=512, hidden_features=256, kan_features=128, num_classes=16, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, kan_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.kan = KAN(
            [kan_features, kan_features, num_classes],
            grid_size=5,
            spline_order=3
        )

    def forward(self, x):
        x = self.mlp(x)
        return self.kan(x)

if __name__ == "__main__":
    head = ClassificationHead()
    x = torch.randn(4, 512)
    with torch.no_grad():
        y = head(x)
    print("input :", x.shape)
    print("output:", y.shape)
    print(f"Parameters: {sum(p.numel() for p in head.parameters()) / 1e6:.3f} M")