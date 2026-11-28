import torch
import torch.nn as nn

class ConvBNAct3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=None):
        super().__init__()
        if padding is None:
            padding = tuple(k // 2 for k in kernel_size)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DualPath3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        self.spatial_conv = ConvBNAct3D(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.temporal_conv = ConvBNAct3D(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.spatial_norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.temporal_norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        B, T, F, C, H, W = x.shape
        # spatial 分支: [B*T, C, F, H, W] -> C 为 channel, F 为 depth
        spatial = x.reshape(B * T, C, F, H, W)
        spatial = self.spatial_conv(spatial)
        spatial = self.spatial_norm(spatial)
        Fo, Ho, Wo = spatial.shape[2:]
        spatial = spatial.permute(0, 2, 1, 3, 4).reshape(B, T, Fo, -1, Ho, Wo)
        # temporal 分支: [B*F, C, T, H, W] -> C 为 channel, T 为 depth
        temporal = x.permute(0, 2, 1, 3, 4, 5).reshape(B * F, C, T, H, W)
        temporal = self.temporal_conv(temporal)
        temporal = self.temporal_norm(temporal)
        To, Ho2, Wo2 = temporal.shape[2:]
        temporal = temporal.permute(0, 2, 1, 3, 4).reshape(B, F, To, -1, Ho2, Wo2).permute(0, 2, 1, 3, 4, 5)
        return self.act(spatial + temporal)

class DualPathResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), kernel_size=(3, 3, 3)):
        super().__init__()
        self.conv1 = DualPath3DConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = DualPath3DConv(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        if in_channels != out_channels or stride != (1, 1, 1):
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.shortcut_norm = nn.GroupNorm(min(32, out_channels), out_channels)
        else:
            self.shortcut = None
            self.shortcut_norm = None
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            B, T, F, C, H, W = x.shape
            identity = x.reshape(B * T, C, F, H, W)
            identity = self.shortcut(identity)
            identity = self.shortcut_norm(identity)
            Fo, Ho, Wo = identity.shape[2:]
            identity = identity.permute(0, 2, 1, 3, 4).reshape(B, T, Fo, -1, Ho, Wo)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + identity)

class Focal3DBackbone(nn.Module):
    """
    Input: [B,T,7,1,224,224]
    Output:
        p2:    [B,T,7,512,56,56]
        p3:    [B,T,7,512,28,28]
        p4:    [B,T,7,512,14,14]
        tokens: [B,T,7,49,512]
    """
    def __init__(self, num_focus=7, embed_dim=512, image_size=224, sequence_length=5):
        super().__init__()
        if image_size != 224:
            raise ValueError("Focal3DBackbone requires image_size=224.")
        if num_focus != 7:
            raise ValueError("Focal3DBackbone requires exactly 7 focal planes.")
        if sequence_length != 5:
            raise ValueError("Focal3DBackbone requires sequence_length=5.")
        self.num_focus = num_focus
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.stem = DualPath3DConv(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2))
        self.stage1 = nn.Sequential(
            DualPathResidualBlock(64, 128, stride=(1, 2, 2)),
            DualPathResidualBlock(128, 128)
        )
        self.p2_projection = nn.Conv3d(128, embed_dim, kernel_size=1, bias=False)
        self.p2_norm = nn.GroupNorm(32, embed_dim)
        self.stage2 = nn.Sequential(
            DualPathResidualBlock(128, 256, stride=(1, 2, 2)),
            DualPathResidualBlock(256, 256)
        )
        self.p3_projection = nn.Conv3d(256, embed_dim, kernel_size=1, bias=False)
        self.p3_norm = nn.GroupNorm(32, embed_dim)
        self.stage3 = nn.Sequential(
            DualPathResidualBlock(256, 512, stride=(1, 2, 2)),
            DualPathResidualBlock(512, 512)
        )
        self.p4_projection = nn.Conv3d(512, embed_dim, kernel_size=1, bias=False)
        self.p4_norm = nn.GroupNorm(32, embed_dim)
        self.patch_embed = nn.Conv3d(embed_dim, embed_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _to_3d(x):
        B, T, F, C, H, W = x.shape
        return x.reshape(B * T, C, F, H, W)

    @staticmethod
    def _from_3d(x, B, T, F):
        _, C, Fd, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B, T, Fd, C, H, W)

    def forward(self, x):
        if x.ndim != 6:
            raise ValueError(f"Expected [B,T,F,C,H,W], got {tuple(x.shape)}")
        B, T, F, C, H, W = x.shape
        if (T, F, C, H, W) != (self.sequence_length, self.num_focus, 1, 224, 224):
            raise ValueError(f"Expected [B,{self.sequence_length},7,1,224,224], got {tuple(x.shape)}")
        x = self.stem(x)
        x1 = self.stage1(x)
        p2 = self.p2_norm(self.p2_projection(self._to_3d(x1)))
        p2 = self._from_3d(p2, B, T, F)
        x2 = self.stage2(x1)
        p3 = self.p3_norm(self.p3_projection(self._to_3d(x2)))
        p3 = self._from_3d(p3, B, T, F)
        x3 = self.stage3(x2)
        p4 = self.p4_norm(self.p4_projection(self._to_3d(x3)))
        p4 = self._from_3d(p4, B, T, F)
        patch = self.patch_embed(self._to_3d(x3))
        patch = self._from_3d(patch, B, T, F)
        tokens = patch.permute(0, 1, 2, 4, 5, 3).contiguous()
        tokens = tokens.reshape(B, T, F, -1, self.embed_dim)
        tokens = self.patch_norm(tokens)
        if p2.shape != (B, T, F, self.embed_dim, 56, 56):
            raise RuntimeError(f"Invalid p2 shape: {tuple(p2.shape)}")
        if p3.shape != (B, T, F, self.embed_dim, 28, 28):
            raise RuntimeError(f"Invalid p3 shape: {tuple(p3.shape)}")
        if p4.shape != (B, T, F, self.embed_dim, 14, 14):
            raise RuntimeError(f"Invalid p4 shape: {tuple(p4.shape)}")
        if tokens.shape != (B, T, F, 49, self.embed_dim):
            raise RuntimeError(f"Invalid token shape: {tuple(tokens.shape)}")
        return {"p2": p2, "p3": p3, "p4": p4, "tokens": tokens}

if __name__ == "__main__":
    model = Focal3DBackbone()
    x = torch.randn(1, 5, 7, 1, 224, 224)
    with torch.no_grad():
        output = model(x)
    for key, value in output.items():
        print(f"{key:8s}: {tuple(value.shape)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")