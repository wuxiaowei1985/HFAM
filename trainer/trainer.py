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
        total_loss = (self.final_loss_weight * final_loss + self.coarse_loss_weight * coarse_loss + self.frame_loss_weight * frame_loss + self.coarse_frame_loss_weight * coarse_frame_loss)
        return total_loss, final_loss, coarse_loss, frame_loss, coarse_frame_loss

    @staticmethod
    def _update_metrics(output, labels, frame_labels, batch, metrics):
        sample_preds = output["final_log_probs"].float().argmax(dim=1)
        frame_probs = output["final_frame_log_probs"].float().exp().detach().cpu()
        frame_labels = frame_labels.detach().cpu()
        temporal_mask = batch["temporal_mask"].detach().cpu()
        metrics["sample_correct"] += (sample_preds == labels).sum().item()
        metrics["sample_total"] += labels.size(0)
        for batch_index, embryo in enumerate(batch["embryo"]):
            runs = batch["runs"][batch_index]
            for time_index, run_id in enumerate(runs):
                if time_index >= frame_labels.size(1):
                    continue
                if not temporal_mask[batch_index, time_index]:
                    continue
                frame_label = int(frame_labels[batch_index, time_index])
                if frame_label < 0:
                    continue
                key = (embryo, int(run_id))
                probability = frame_probs[batch_index, time_index]
                if key not in metrics["frame_predictions"]:
                    metrics["frame_predictions"][key] = {"prob_sum": probability.clone(), "count": 1, "label": frame_label}
                else:
                    item = metrics["frame_predictions"][key]
                    if item["label"] != frame_label:
                        raise ValueError(f"Inconsistent labels for frame {key}.")
                    item["prob_sum"] += probability
                    item["count"] += 1

    @staticmethod
    def _calculate_frame_metrics(frame_predictions):
        frame_correct = 0
        for item in frame_predictions.values():
            mean_probability = item["prob_sum"] / item["count"]
            pred_label = int(mean_probability.argmax())
            if pred_label == item["label"]:
                frame_correct += 1
        frame_total = len(frame_predictions)
        frame_accuracy = frame_correct / max(frame_total, 1)
        return frame_correct, frame_total, frame_accuracy
    def train_one_epoch(self, loader):
        self.model.train()
        loss_names = ("loss", "final_loss", "coarse_loss", "frame_loss", "coarse_frame_loss")
        loss_totals = {name: 0.0 for name in loss_names}
        metrics = {"sample_correct": 0, "sample_total": 0, "frame_predictions": {}}
        progress = tqdm(loader, desc="Training")
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            temporal_mask = batch["temporal_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).long()
            frame_labels = batch["frame_labels"].to(self.device, non_blocking=True).long()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model({"images": images, "temporal_mask": temporal_mask, "label": labels}, return_dict=True)
                losses = self._compute_loss(output, labels, frame_labels)
                total_loss, final_loss, coarse_loss, frame_loss, coarse_frame_loss = losses
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_size = labels.size(0)
            for name, value in zip(loss_names, losses):
                loss_totals[name] += value.item() * batch_size
            self._update_metrics(output, labels, frame_labels, batch, metrics)
            sample_acc = metrics["sample_correct"] / max(metrics["sample_total"], 1)
            frame_correct, frame_total, frame_acc = self._calculate_frame_metrics(metrics["frame_predictions"])
            progress.set_postfix(
                loss=f"{total_loss.item():.4f}",
                final=f"{final_loss.item():.4f}",
                coarse=f"{coarse_loss.item():.4f}",
                frame=f"{frame_loss.item():.4f}",
                coarse_frame=f"{coarse_frame_loss.item():.4f}",
                sample_acc=f"{sample_acc:.4f}",
                frame_acc=f"{frame_acc:.4f}"
            )
        sample_total = max(metrics["sample_total"], 1)
        frame_correct, frame_total, frame_accuracy = self._calculate_frame_metrics(metrics["frame_predictions"])
        return {
            **{name: value / sample_total for name, value in loss_totals.items()},
            "sample_accuracy": metrics["sample_correct"] / sample_total,
            "frame_accuracy": frame_accuracy,
            "sample_correct": metrics["sample_correct"],
            "sample_total": metrics["sample_total"],
            "frame_correct": frame_correct,
            "frame_total": frame_total
        }