import torch
import torch.nn as nn
from model.module.Backbone import Focal3DBackbone
from model.module.TF_fusion import TF_fusion
from model.module.temporal_transformer import TemporalTransformer
from model.module.MSFD_Attention import MSFDAttention
from model.module.classifier import ClassificationHead
from model.module.coarse_embedding import CoarseEmbedding

class HierarchicalFocusAttentionModel(nn.Module):
    """
    Input: [B,5,7,1,224,224]
    Backbone: [B,5,7,49,512]
    Spatial branch: [5B,7,49,512]
    Temporal branch: [7B,5,49,512]
    TF_fusion:
        frame feature  [B,5,512]
        sample feature [B,512]
    Classification:
        pronuclear  : 3 classes
        cleavage    : 8 classes
        blastocyst  : 5 classes
    Final:
        sample_logits       [B,16]
        frame_logits        [B,5,16]
    """
    def __init__(self, pretrained=False, feature_dim=512, pronuclear=3, cleavage=8, blastocyst=5, num_layers=2, dropout=0.2, image_size=224, sequence_length=5):
        super().__init__()
        if pretrained:
            raise ValueError("Pretrained weights are disabled.")
        if feature_dim != 512:
            raise ValueError("feature_dim must be 512.")
        if image_size != 224:
            raise ValueError("image_size must be 224.")
        if sequence_length != 5:
            raise ValueError("sequence_length must be 5.")
        if pronuclear + cleavage + blastocyst != 16:
            raise ValueError("Expected 3 + 8 + 5 = 16 classes.")
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.num_focus = 7
        self.num_tokens = 49
        self.num_classes = pronuclear + cleavage + blastocyst
        self.encoder = Focal3DBackbone(num_focus=7, embed_dim=feature_dim, image_size=image_size, sequence_length=sequence_length)
        self.coarse_embedding = CoarseEmbedding(sequence_length=sequence_length, num_focus=7, num_tokens=49, feature_dim=feature_dim, dropout=dropout)
        self.msfd_attention = MSFDAttention(feature_dim=feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, num_focus=7, dropout=dropout)
        self.temporal_transformer = TemporalTransformer(feature_dim=feature_dim, num_heads=8, depth=num_layers, dropout=dropout, sequence_length=sequence_length, num_tokens=49)
        self.tf_fusion = TF_fusion(feature_dim=feature_dim, num_heads=8, sequence_length=sequence_length, num_focus=7, num_tokens=49, dropout=dropout)
        self.fine_heads = nn.ModuleDict({
            "pronuclear": ClassificationHead(in_features=feature_dim, hidden_features=256, kan_features=128, num_classes=pronuclear, dropout=dropout),
            "cleavage": ClassificationHead(in_features=feature_dim, hidden_features=256, kan_features=128, num_classes=cleavage, dropout=dropout),
            "blastocyst": ClassificationHead(in_features=feature_dim, hidden_features=256, kan_features=128, num_classes=blastocyst, dropout=dropout)
        })

    @staticmethod
    def _get_last_valid_index(frame_mask):
        if frame_mask.ndim != 2:
            raise ValueError("frame_mask must have shape [B,T].")
        frame_mask = frame_mask.bool()
        if not frame_mask.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one valid frame.")
        indices = torch.arange(frame_mask.size(1), device=frame_mask.device).unsqueeze(0).expand_as(frame_mask)
        return indices.masked_fill(~frame_mask, -1).max(dim=1, keepdim=True)[0]

    @staticmethod
    def _select_last_valid(x, last_index):
        index = last_index.view(x.size(0), 1, *([1] * (x.ndim - 2))).expand(x.size(0), 1, *x.shape[2:])
        return x.gather(dim=1, index=index).squeeze(1)

    @staticmethod
    def _build_pyramid(backbone, B, T):
        pyramid = []
        for key in ("p2", "p3", "p4"):
            feature = backbone[key]
            _, _, F, D, H, W = feature.shape
            feature = feature.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B * T, D, F, H, W)
            pyramid.append(feature)
        return pyramid

    @staticmethod
    def _merge_head_logits(pronuclear_logits, cleavage_logits, blastocyst_logits):
        return torch.cat([pronuclear_logits, cleavage_logits, blastocyst_logits], dim=-1)

    def forward(self, batch, stage=None, return_dict=False):
        if isinstance(batch, dict):
            images = batch["images"]
            frame_mask = batch.get("frame_mask", batch.get("temporal_mask"))
        else:
            images = batch
            frame_mask = None
        if images.ndim != 6:
            raise ValueError(f"Expected [B,T,7,1,224,224], got {tuple(images.shape)}")
        B, T, F, C, H, W = images.shape
        if (T, F, C, H, W) != (
            self.sequence_length,7, 1, 224, 224):
            raise ValueError(f"Expected [B,{self.sequence_length},7,1,224,224], " f"got {tuple(images.shape)}")
        if frame_mask is None:
            frame_mask = torch.ones(B, T, dtype=torch.bool, device=images.device)
        else:
            frame_mask = frame_mask.to(device=images.device, dtype=torch.bool)
        if frame_mask.shape != (B, T):
            raise ValueError(f"Expected frame_mask [{B},{T}], got {tuple(frame_mask.shape)}")
        last_index = self._get_last_valid_index(frame_mask)
        # --------------------------------------------------
        # 1. Backbone
        # --------------------------------------------------
        backbone = self.encoder(images)
        pyramid = self._build_pyramid(backbone, B, T)
        tokens = backbone["tokens"]
        if tokens.shape != (B, T, self.num_focus, self.num_tokens, self.feature_dim):
            raise RuntimeError(f"Expected backbone tokens " f"[{B},{T},7,49,512], got {tuple(tokens.shape)}")
        # --------------------------------------------------
        # 2. Direct temporal / spatial embedding
        # --------------------------------------------------
        temporal_embedding, spatial_embedding = self.coarse_embedding(B, device=tokens.device, dtype=tokens.dtype)
        # --------------------------------------------------
        # 3. Spatial branch
        # --------------------------------------------------
        spatial_tokens = tokens.reshape(B * T, self.num_focus, self.num_tokens, self.feature_dim)
        spatial_tokens = spatial_tokens + spatial_embedding
        spatial_query = spatial_tokens.reshape(B * T, self.num_focus * self.num_tokens, self.feature_dim)
        # --------------------------------------------------
        # 4. Temporal branch
        # --------------------------------------------------
        temporal_tokens = tokens.permute(0, 2, 1, 3, 4).contiguous()
        temporal_tokens = temporal_tokens.reshape(B * self.num_focus, T, self.num_tokens, self.feature_dim)
        temporal_tokens = temporal_tokens + temporal_embedding
        temporal_query = temporal_tokens.reshape(B * self.num_focus, T * self.num_tokens, self.feature_dim)
        temporal_mask = frame_mask.unsqueeze(1).expand(B, self.num_focus, T).reshape(B * self.num_focus, T)
        # --------------------------------------------------
        # 5. MSFD
        # --------------------------------------------------
        spatial_feature = self.msfd_attention(spatial_query, pyramid)
        # --------------------------------------------------
        # 6. Temporal Transformer
        # --------------------------------------------------
        temporal_feature = self.temporal_transformer(temporal_query, temporal_mask=temporal_mask)
        # --------------------------------------------------
        # 7. TF Fusion
        # --------------------------------------------------
        frame_feature, sample_feature, token_weight, frame_weight = self.tf_fusion(spatial_feature, temporal_feature, temporal_mask=frame_mask)
        # --------------------------------------------------
        # 8. Three classification heads
        # --------------------------------------------------
        # frame-level 仍来自 frame_feature
        pronuclear_frame_logits = self.fine_heads["pronuclear"](frame_feature)
        cleavage_frame_logits = self.fine_heads["cleavage"](frame_feature)
        blastocyst_frame_logits = self.fine_heads["blastocyst"](frame_feature)
        frame_logits = self._merge_head_logits(pronuclear_frame_logits, cleavage_frame_logits, blastocyst_frame_logits)
        # sample-level 改为来自 sample_feature
        pronuclear_logits = self.fine_heads["pronuclear"](sample_feature)
        cleavage_logits = self.fine_heads["cleavage"](sample_feature)
        blastocyst_logits = self.fine_heads["blastocyst"](sample_feature)
        sample_logits = self._merge_head_logits(pronuclear_logits, cleavage_logits, blastocyst_logits)
        frame_log_probs = torch.log_softmax(frame_logits.float(), dim=-1)
        log_probs = torch.log_softmax(sample_logits.float(), dim=-1)
        output = {
            "final_logits": sample_logits,
            "final_log_probs": log_probs,
            "final_probs": torch.exp(log_probs),
            "final_pred": sample_logits.argmax(dim=-1),
            "final_frame_logits": frame_logits,
            "final_frame_log_probs": frame_log_probs,
            "final_frame_probs": torch.exp(frame_log_probs),
            "final_frame_pred": frame_logits.argmax(dim=-1),
            "pronuclear_logits": pronuclear_logits,
            "pronuclear_frame_logits": pronuclear_frame_logits,
            "pronuclear_frame_probs": torch.softmax(pronuclear_frame_logits.float(), dim=-1),
            "cleavage_logits": cleavage_logits,
            "cleavage_frame_logits": cleavage_frame_logits,
            "cleavage_frame_probs": torch.softmax(cleavage_frame_logits.float(), dim=-1),
            "blastocyst_logits": blastocyst_logits,
            "blastocyst_frame_logits": blastocyst_frame_logits,
            "blastocyst_frame_probs": torch.softmax(blastocyst_frame_logits.float(), dim=-1),
            "frame_features": frame_feature,
            "sample_feature": sample_feature,
            "tf_token_weight": token_weight.view(B, T, self.num_focus, self.num_tokens),
            "frame_weight": frame_weight,
            "temporal_embedding": temporal_embedding,
            "spatial_embedding": spatial_embedding,
            "spatial_feature": spatial_feature,
            "temporal_feature": temporal_feature,
            "pyramid": pyramid,
            "frame_mask": frame_mask,
            "last_index": last_index
        }
        return output if return_dict else log_probs

if __name__ == "__main__":
    model = HierarchicalFocusAttentionModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")