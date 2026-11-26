import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset.labels import get_coarse_label
from configs import config as cfg

class Validator:
    def __init__(self, model, criterion, device):
        self.model = model.to(device)
        self.device = device
        self.use_amp = device.type == "cuda"

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc="Validation")
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            temporal_mask = batch["temporal_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).long()
            frame_labels = batch["frame_labels"].to(self.device, non_blocking=True).long()
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model({"images": images, "temporal_mask": temporal_mask, "label": labels}, return_dict=True)
                coarse_labels = torch.tensor([get_coarse_label(int(label)) for label in labels], device=self.device, dtype=torch.long)
                coarse_frame_labels = torch.full_like(frame_labels, -100)
                valid_mask = frame_labels != -100
                if valid_mask.any():
                    coarse_frame_labels[valid_mask] = torch.tensor([get_coarse_label(int(label)) for label in frame_labels[valid_mask]], device=self.device, dtype=torch.long)
                final_loss = F.nll_loss(output["final_log_probs"].float(), labels)
                coarse_loss = F.cross_entropy(output["coarse_logits"].float(), coarse_labels, label_smoothing=cfg.LABEL_SMOOTHING)
                frame_loss = F.nll_loss(output["final_frame_log_probs"].float().reshape(-1, 16), frame_labels.reshape(-1), ignore_index=-100)
                coarse_frame_loss = F.cross_entropy(output["coarse_frame_logits"].float().reshape(-1, 3), coarse_frame_labels.reshape(-1), ignore_index=-100, label_smoothing=cfg.LABEL_SMOOTHING)
                loss = final_loss + 0.5 * coarse_loss + frame_loss + 0.5 * coarse_frame_loss
            preds = output["final_log_probs"].float().argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")
        return total_loss / max(total, 1), correct / max(total, 1)