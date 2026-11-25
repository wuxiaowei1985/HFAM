import torch
import torch.nn as nn
from torchnd import ConvNd

class ConvBNAct4D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = tuple(k // 2 for k in kernel_size) if isinstance(kernel_size, tuple) else kernel_size // 2
        self.conv = ConvNd(in_channels, out_channels, kernel_size, dim=(-4, -3, -2, -1), stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(32, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResidualBlock4D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 2, 2)):
        super().__init__()
        self.conv1 = ConvBNAct4D(in_channels, out_channels, kernel_size=(3, 3, 3, 3), stride=stride, padding=(1, 1, 1, 1))
        self.conv2 = ConvNd(out_channels, out_channels, (3, 3, 3, 3), dim=(-4, -3, -2, -1), padding=(1, 1, 1, 1), bias=False)
        self.norm2 = nn.GroupNorm(32, out_channels)
        if in_channels != out_channels or stride != (1, 1, 1, 1):
            self.shortcut = nn.Sequential(
                ConvNd(in_channels, out_channels, 1, dim=(-4, -3, -2, -1), stride=stride, bias=False),
                nn.GroupNorm(32, out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm2(self.conv2(x))
        return self.act(x + identity)

class Focal4DBackbone(nn.Module):
    """
    Input: [B,5,7,1,224,224]
    Internal: [B,1,5,7,224,224]
    Outputs:
        p2:     [B,5,512,7,56,56]
        p3:     [B,5,512,7,28,28]
        p4:     [B,5,512,7,14,14]
        tokens: [B,5,343,512]
    """
    def __init__(self, num_focus=7, embed_dim=512, image_size=224, sequence_length=5):
        super().__init__()
        if image_size != 224:
            raise ValueError("Focal4DBackbone requires image_size=224.")
        if num_focus != 7:
            raise ValueError("Focal4DBackbone requires exactly 7 focal planes.")
        if sequence_length != 5:
            raise ValueError("Focal4DBackbone requires sequence_length=5.")
        self.num_focus = num_focus
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.stem = ConvBNAct4D(1, 64, kernel_size=(3, 3, 7, 7), stride=(1, 1, 2, 2), padding=(1, 1, 3, 3))
        self.stage1 = nn.Sequential(
            ResidualBlock4D(64, 128, stride=(1, 1, 2, 2)),
            ResidualBlock4D(128, 128, stride=(1, 1, 1, 1))
        )
        self.p2_projection = ConvNd(128, embed_dim, 1, dim=(-4, -3, -2, -1), bias=False)
        self.p2_norm = nn.GroupNorm(32, embed_dim)
        self.stage2 = nn.Sequential(
            ResidualBlock4D(128, 256, stride=(1, 1, 2, 2)),
            ResidualBlock4D(256, 256, stride=(1, 1, 1, 1))
        )
        self.p3_projection = ConvNd(256, embed_dim, 1, dim=(-4, -3, -2, -1), bias=False)
        self.p3_norm = nn.GroupNorm(32, embed_dim)
        self.stage3 = nn.Sequential(
            ResidualBlock4D(256, 512, stride=(1, 1, 2, 2)),
            ResidualBlock4D(512, 512, stride=(1, 1, 1, 1))
        )
        self.p4_projection = ConvNd(512, embed_dim, 1, dim=(-4, -3, -2, -1), bias=False)
        self.p4_norm = nn.GroupNorm(32, embed_dim)
        self.patch_embed = ConvNd(embed_dim, embed_dim, kernel=(1, 1, 2, 2), dim=(-4, -3, -2, -1), stride=(1, 1, 2, 2), bias=False)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, ConvNd):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 6:
            raise ValueError(f"Expected [B,T,F,C,H,W], got {tuple(x.shape)}")
        B, T, F, C, H, W = x.shape
        if (T, F, C, H, W) != (5, 7, 1, 224, 224):
            raise ValueError(f"Expected [B,5,7,1,224,224], got {tuple(x.shape)}")
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()
        x = self.stem(x)
        x1 = self.stage1(x)
        p2 = self.p2_norm(self.p2_projection(x1))
        x2 = self.stage2(x1)
        p3 = self.p3_norm(self.p3_projection(x2))
        x3 = self.stage3(x2)
        p4 = self.p4_norm(self.p4_projection(x3))
        patch = self.patch_embed(p4)
        tokens = patch.permute(0, 2, 3, 4, 5, 1).contiguous()
        tokens = tokens.view(B, T, self.num_focus * 7 * 7, self.embed_dim)
        tokens = self.patch_norm(tokens)
        expected_shapes = {
            "p2": (B, T, self.embed_dim, 7, 56, 56),
            "p3": (B, T, self.embed_dim, 7, 28, 28),
            "p4": (B, T, self.embed_dim, 7, 14, 14),
            "tokens": (B, T, 343, self.embed_dim)
        }
        actual_shapes = {
            "p2": tuple(p2.permute(0, 2, 1, 3, 4, 5).shape),
            "p3": tuple(p3.permute(0, 2, 1, 3, 4, 5).shape),
            "p4": tuple(p4.permute(0, 2, 1, 3, 4, 5).shape),
            "tokens": tuple(tokens.shape)
        }
        for name, expected in expected_shapes.items():
            if actual_shapes[name] != expected:
                raise RuntimeError(f"{name}: expected {expected}, got {actual_shapes[name]}")
        return {
            "p2": p2.permute(0, 2, 1, 3, 4, 5).contiguous(),
            "p3": p3.permute(0, 2, 1, 3, 4, 5).contiguous(),
            "p4": p4.permute(0, 2, 1, 3, 4, 5).contiguous(),
            "tokens": tokens
        }

if __name__ == "__main__":
    model = Focal4DBackbone()
    x = torch.randn(1, 5, 7, 1, 224, 224)
    with torch.no_grad():
        output = model(x)
    for key, value in output.items():
        print(f"{key:8s}: {tuple(value.shape)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")