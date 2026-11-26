import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset.labels import get_coarse_label
from configs import config as cfg

class Trainer:
    def __init__(self, model, criterion, optimizer, device, final_loss_weight=1.0, coarse_loss_weight=0.5, frame_loss_weight=1.0, coarse_frame_loss_weight=0.5):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.final_loss_weight = final_loss_weight
        self.coarse_loss_weight = coarse_loss_weight
        self.frame_loss_weight = frame_loss_weight
        self.coarse_frame_loss_weight = coarse_frame_loss_weight
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    @staticmethod
    def build_coarse_frame_labels(frame_labels):
        coarse_frame_labels = torch.full_like(frame_labels, -100)
        valid_mask = frame_labels != -100
        if valid_mask.any():
            valid_labels = frame_labels[valid_mask]
            coarse_frame_labels[valid_mask] = torch.tensor([get_coarse_label(int(label)) for label in valid_labels], device=frame_labels.device, dtype=torch.long)
        return coarse_frame_labels

    def _compute_loss(self, output, labels, frame_labels):
        coarse_labels = torch.tensor([get_coarse_label(int(label)) for label in labels], device=self.device, dtype=torch.long)
        coarse_frame_labels = self.build_coarse_frame_labels(frame_labels)
        final_loss = F.nll_loss(output["final_log_probs"].float(), labels)
        coarse_loss = F.cross_entropy(output["coarse_logits"].float(), coarse_labels, label_smoothing=cfg.LABEL_SMOOTHING)
        frame_loss = F.nll_loss(output["final_frame_log_probs"].float().reshape(-1, 16), frame_labels.reshape(-1), ignore_index=-100)
        coarse_frame_loss = F.cross_entropy(output["coarse_frame_logits"].float().reshape(-1, 3), coarse_frame_labels.reshape(-1), ignore_index=-100, label_smoothing=cfg.LABEL_SMOOTHING)
        loss = self.final_loss_weight * final_loss + self.coarse_loss_weight * coarse_loss + self.frame_loss_weight * frame_loss + self.coarse_frame_loss_weight * coarse_frame_loss
        return loss, final_loss, coarse_loss, frame_loss, coarse_frame_loss, coarse_labels

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_final_loss = 0.0
        total_coarse_loss = 0.0
        total_frame_loss = 0.0
        total_coarse_frame_loss = 0.0
        correct = 0
        coarse_correct = 0
        total = 0
        progress = tqdm(loader, desc="Training")
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            temporal_mask = batch["temporal_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).long()
            frame_labels = batch["frame_labels"].to(self.device, non_blocking=True).long()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model({"images": images, "temporal_mask": temporal_mask, "label": labels}, return_dict=True)
                loss, final_loss, coarse_loss, frame_loss, coarse_frame_loss, coarse_labels = self._compute_loss(output, labels, frame_labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_size = labels.size(0)
            preds = output["final_log_probs"].float().argmax(dim=1)
            coarse_preds = output["coarse_logits"].float().argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_final_loss += final_loss.item() * batch_size
            total_coarse_loss += coarse_loss.item() * batch_size
            total_frame_loss += frame_loss.item() * batch_size
            total_coarse_frame_loss += coarse_frame_loss.item() * batch_size
            correct += (preds == labels).sum().item()
            coarse_correct += (coarse_preds == coarse_labels).sum().item()
            total += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}", final=f"{final_loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")
        return {
            "loss": total_loss / max(total, 1),
            "final_loss": total_final_loss / max(total, 1),
            "coarse_loss": total_coarse_loss / max(total, 1),
            "frame_loss": total_frame_loss / max(total, 1),
            "coarse_frame_loss": total_coarse_frame_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "coarse_accuracy": coarse_correct / max(total, 1)
        }

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_final_loss = 0.0
        total_coarse_loss = 0.0
        total_frame_loss = 0.0
        total_coarse_frame_loss = 0.0
        correct = 0
        coarse_correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        progress = tqdm(loader, desc="Validation")
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            temporal_mask = batch["temporal_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).long()
            frame_labels = batch["frame_labels"].to(self.device, non_blocking=True).long()
            output = self.model({"images": images, "temporal_mask": temporal_mask, "label": labels}, return_dict=True)
            loss, final_loss, coarse_loss, frame_loss, coarse_frame_loss, coarse_labels = self._compute_loss(output, labels, frame_labels)
            batch_size = labels.size(0)
            preds = output["final_log_probs"].float().argmax(dim=1)
            coarse_preds = output["coarse_logits"].float().argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_final_loss += final_loss.item() * batch_size
            total_coarse_loss += coarse_loss.item() * batch_size
            total_frame_loss += frame_loss.item() * batch_size
            total_coarse_frame_loss += coarse_frame_loss.item() * batch_size
            correct += (preds == labels).sum().item()
            coarse_correct += (coarse_preds == coarse_labels).sum().item()
            total += batch_size
            all_predictions.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")
        return {
            "loss": total_loss / max(total, 1),
            "final_loss": total_final_loss / max(total, 1),
            "coarse_loss": total_coarse_loss / max(total, 1),
            "frame_loss": total_frame_loss / max(total, 1),
            "coarse_frame_loss": total_coarse_frame_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "coarse_accuracy": coarse_correct / max(total, 1),
            "predictions": all_predictions,
            "labels": all_labels
        }