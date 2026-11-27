import torch
import torch.nn as nn
from model.module.Backbone import Focal4DBackbone
from model.module.focus_attention import FocusAttention
from model.module.temporal_transformer import TemporalTransformer
from model.module.MSFD_Attention import MSFDAttention
from model.module.classifier import ClassificationHead
from model.module.coarse_embedding import CoarseEmbedding

class HierarchicalFocusAttentionModel(nn.Module):
    """
    Input: images: [B, T, 7, 1, 224, 224]
    Experts:
        Coarse expert: 3 classes
        Pronuclear expert: 3 classes
        Cleavage expert: 8 classes
        Blastocyst expert: 5 classes
    Output:
        final_log_probs: [B, 16]
        final_frame_log_probs: [B, T, 16]
    """
    def __init__(self, pretrained=False,feature_dim=512, coarse=3, pronuclear=3, cleavage=8, blastocyst=5, num_layers=2, dropout=0.2, image_size=224, sequence_length=5):
        super().__init__()
        if pretrained:
            raise ValueError("Pretrained weights are disabled.")
        if feature_dim != 512:
            raise ValueError("feature_dim must be 512.")
        if image_size != 224:
            raise ValueError("image_size must be 224.")
        if coarse != 3 or pronuclear != 3 or cleavage != 8 or blastocyst != 5:
            raise ValueError("Expected expert class numbers: 3, 3, 8, 5.")
        if pronuclear + cleavage + blastocyst != 16:
            raise ValueError("Invalid hierarchical classification configuration.")
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.num_classes = pronuclear + cleavage + blastocyst
        # --------------------------------------------------
        # 1. 4D Backbone
        # --------------------------------------------------
        self.encoder = Focal4DBackbone(num_focus=7, embed_dim=feature_dim, image_size=image_size, sequence_length=sequence_length)
        # --------------------------------------------------
        # 2. Focus Attention
        # --------------------------------------------------
        self.focus_attention = FocusAttention(feature_dim=feature_dim, num_heads=8, depth=num_layers, num_focus=7, query_h=7, query_w=7, dropout=dropout)
        # --------------------------------------------------
        # 3. Coarse Expert
        # --------------------------------------------------
        self.coarse_temporal_transformer = TemporalTransformer(feature_dim=feature_dim, num_heads=8, depth=num_layers, dropout=dropout, sequence_length=sequence_length, num_tokens=49)
        self.coarse_head = ClassificationHead(in_features=feature_dim, hidden_features=256, kan_features=128, num_classes=coarse, dropout=dropout)
        # --------------------------------------------------
        # 4. Coarse Embedding
        # --------------------------------------------------
        self.coarse_embedding = CoarseEmbedding(num_classes=coarse, sequence_length=sequence_length, num_tokens=49, feature_dim=feature_dim, dropout=dropout)
        # --------------------------------------------------
        # 5. Three Independent Fine Experts
        # --------------------------------------------------
        self.msfd_attention = nn.ModuleDict({
            "pronuclear": MSFDAttention(feature_dim=feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, num_focus=7, dropout=dropout),
            "cleavage": MSFDAttention(feature_dim=feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, num_focus=7, dropout=dropout),
            "blastocyst": MSFDAttention(feature_dim=feature_dim, num_heads=8, depth=num_layers, num_levels=3, num_points=4, num_focus=7, dropout=dropout)
        })
        self.fine_temporal_transformer = nn.ModuleDict({
            "pronuclear": TemporalTransformer(feature_dim=feature_dim, num_heads=8, depth=num_layers, dropout=dropout, sequence_length=sequence_length, num_tokens=49),
            "cleavage": TemporalTransformer(feature_dim=feature_dim, num_heads=8, depth=num_layers, dropout=dropout, sequence_length=sequence_length, num_tokens=49),
            "blastocyst": TemporalTransformer(feature_dim=feature_dim, num_heads=8, depth=num_layers, dropout=dropout, sequence_length=sequence_length, num_tokens=49)
        })
        self.fine_heads = nn.ModuleDict({
            "pronuclear": ClassificationHead(in_features=feature_dim, hidden_features=256, kan_features=128, num_classes=pronuclear, dropout=dropout),
            "cleavage": ClassificationHead(in_features=feature_dim, hidden_features=256, kan_features=128, num_classes=cleavage, dropout=dropout),
            "blastocyst": ClassificationHead(in_features=feature_dim, hidden_features=256, kan_features=128, num_classes=blastocyst, dropout=dropout)
        })

    @staticmethod
    def _get_last_valid_index(frame_mask):
        """
        frame_mask: [B, T]
        Return: [B, 1]
        """
        if frame_mask.ndim != 2:
            raise ValueError("frame_mask must have shape [B, T].")
        frame_mask = frame_mask.bool()
        if not frame_mask.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one valid frame.")
        indices = torch.arange(frame_mask.size(1), device=frame_mask.device).unsqueeze(0).expand_as(frame_mask)
        return indices.masked_fill(~frame_mask, -1).max(dim=1, keepdim=True)[0]

    @staticmethod
    def _select_last_valid(x, last_index):
        """
        x: [B, T, ...]
        last_index: [B, 1]
        """
        index = last_index.view(x.size(0), 1, *([1] * (x.ndim - 2))).expand(x.size(0), 1, *x.shape[2:])
        return x.gather(dim=1, index=index).squeeze(1)

    def forward(self, batch, stage=None, return_dict=False):
        # --------------------------------------------------
        # 1. Input
        # --------------------------------------------------
        if isinstance(batch, dict):
            images = batch["images"]
            frame_mask = batch.get("frame_mask", batch.get("temporal_mask"))
        else:
            images = batch
            frame_mask = None
        if images.ndim != 6:
            raise ValueError(f"Expected [B,T,7,1,224,224], got {tuple(images.shape)}")
        B, T, F, C, H, W = images.shape
        if (T, F, C, H, W) != (self.sequence_length, 7, 1, 224, 224):
            raise ValueError(f"Expected [B,{self.sequence_length},7,1,224,224], "f"got {tuple(images.shape)}")
        if frame_mask is None:
            frame_mask = torch.ones(B, T, dtype=torch.bool, device=images.device)
        else:
            frame_mask = frame_mask.to(device=images.device, dtype=torch.bool)
        if frame_mask.shape != (B, T):
            raise ValueError(f"Expected frame_mask [{B},{T}], got {tuple(frame_mask.shape)}")
        last_index = self._get_last_valid_index(frame_mask)
        # --------------------------------------------------
        # 2. 4D Backbone
        # --------------------------------------------------
        backbone = self.encoder(images)
        pyramid = [backbone["p2"], backbone["p3"], backbone["p4"]]
        # --------------------------------------------------
        # 3. Focus Attention
        # --------------------------------------------------
        focus_tokens, focus_feature, focus_token_weight, focus_weight = self.focus_attention(backbone["tokens"], return_sequence=True)
        # --------------------------------------------------
        # 4. Coarse Expert
        # --------------------------------------------------
        coarse_sequence = self.coarse_temporal_transformer(focus_feature, temporal_mask=frame_mask)
        coarse_feature = coarse_sequence.mean(dim=2)
        coarse_frame_logits = self.coarse_head(coarse_feature)
        coarse_frame_log_probs = torch.log_softmax(coarse_frame_logits.float(), dim=-1)
        coarse_frame_probs = torch.exp(coarse_frame_log_probs)
        # --------------------------------------------------
        # 5. Coarse Embedding
        # --------------------------------------------------
        coarse_embedding = self.coarse_embedding(coarse_frame_probs)
        fine_query_tokens = focus_feature + coarse_embedding
        # --------------------------------------------------
        # 6. Three Independent Fine Experts
        # --------------------------------------------------
        expert_features = {}
        expert_frame_logits = {}
        expert_frame_log_probs = {}
        expert_frame_probs = {}
        for expert_name in ("pronuclear", "cleavage", "blastocyst"):
            # Independent MSFD Attention
            expert_feature = self.msfd_attention[expert_name](fine_query_tokens, pyramid)
            # Independent Temporal Transformer
            expert_feature = self.fine_temporal_transformer[expert_name](expert_feature, temporal_mask=frame_mask)
            # Spatial Pooling
            expert_feature = expert_feature.mean(dim=2)
            # Independent MLP + KAN Head
            expert_logits = self.fine_heads[expert_name](expert_feature)
            expert_log_probs = torch.log_softmax(expert_logits.float(), dim=-1)
            expert_features[expert_name] = expert_feature
            expert_frame_logits[expert_name] = expert_logits
            expert_frame_log_probs[expert_name] = expert_log_probs
            expert_frame_probs[expert_name] = torch.exp(expert_log_probs)
        # --------------------------------------------------
        # 7. Hierarchical Probability Fusion
        # --------------------------------------------------
        pronuclear_log_probs = coarse_frame_log_probs[:, :, 0:1] + expert_frame_log_probs["pronuclear"]
        cleavage_log_probs = coarse_frame_log_probs[:, :, 1:2] + expert_frame_log_probs["cleavage"]
        blastocyst_log_probs = coarse_frame_log_probs[:, :, 2:3] + expert_frame_log_probs["blastocyst"]
        final_frame_log_probs = torch.cat([pronuclear_log_probs, cleavage_log_probs, blastocyst_log_probs], dim=-1)
        if final_frame_log_probs.shape != (B, T, self.num_classes):
            raise RuntimeError(f"Expected final frame output [{B},{T},{self.num_classes}], "f"got {tuple(final_frame_log_probs.shape)}")
        # --------------------------------------------------
        # 8. Select Last Valid Frame
        # --------------------------------------------------
        final_log_probs = self._select_last_valid(final_frame_log_probs, last_index)
        coarse_logits = self._select_last_valid(coarse_frame_logits, last_index)
        pronuclear_logits = self._select_last_valid(expert_frame_logits["pronuclear"],last_index)
        cleavage_logits = self._select_last_valid(expert_frame_logits["cleavage"], last_index)
        blastocyst_logits = self._select_last_valid(expert_frame_logits["blastocyst"], last_index)
        # --------------------------------------------------
        # 9. Output
        # --------------------------------------------------
        output = {
            "final_logits": final_log_probs,
            "final_log_probs": final_log_probs,
            "final_probs": torch.exp(final_log_probs),
            "final_pred": final_log_probs.argmax(dim=-1),
            "final_frame_log_probs": final_frame_log_probs,
            "final_frame_probs": torch.exp(final_frame_log_probs),
            "coarse_logits": coarse_logits,
            "coarse_probs": torch.softmax(coarse_logits.float(), dim=-1),
            "coarse_pred": coarse_logits.argmax(dim=-1),
            "coarse_frame_logits": coarse_frame_logits,
            "coarse_frame_probs": coarse_frame_probs,
            "pronuclear_logits": pronuclear_logits,
            "pronuclear_frame_logits": expert_frame_logits["pronuclear"],
            "pronuclear_frame_probs": expert_frame_probs["pronuclear"],
            "cleavage_logits": cleavage_logits,
            "cleavage_frame_logits": expert_frame_logits["cleavage"],
            "cleavage_frame_probs": expert_frame_probs["cleavage"],
            "blastocyst_logits": blastocyst_logits,
            "blastocyst_frame_logits": expert_frame_logits["blastocyst"],
            "blastocyst_frame_probs": expert_frame_probs["blastocyst"],
            "focus_attention": focus_token_weight,
            "focus_weight": focus_weight,
            "focus_tokens": focus_tokens,
            "focus_feature": focus_feature,
            "coarse_sequence": coarse_sequence,
            "coarse_embedding": coarse_embedding,
            "pronuclear_feature": expert_features["pronuclear"],
            "cleavage_feature": expert_features["cleavage"],
            "blastocyst_feature": expert_features["blastocyst"],
            "pyramid": pyramid,
            "query_tokens": fine_query_tokens,
            "frame_mask": frame_mask,
            "last_index": last_index
        }
        return output if return_dict else final_log_probs

if __name__ == "__main__":
    model = HierarchicalFocusAttentionModel(sequence_length=5, num_layers=2, dropout=0.2)
    model.eval()
    images = torch.randn(1, 5, 7, 1, 224, 224)
    frame_mask = torch.tensor([[True, True, True, True, True]])
    with torch.no_grad():
        output = model({"images": images, "frame_mask": frame_mask}, return_dict=True)
    print("final_logits:", output["final_logits"].shape)
    print("final_frame_logits:", output["final_frame_log_probs"].shape)
    print("coarse_logits:", output["coarse_logits"].shape)
    print("coarse_frame_logits:", output["coarse_frame_logits"].shape)
    print("pronuclear_logits:", output["pronuclear_logits"].shape)
    print("cleavage_logits:", output["cleavage_logits"].shape)
    print("blastocyst_logits:", output["blastocyst_logits"].shape)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")