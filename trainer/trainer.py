import torch
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, criterion, optimizer, device, sample_loss_weight=1.0, frame_loss_weight=1.0):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.sample_loss_weight = sample_loss_weight
        self.frame_loss_weight = frame_loss_weight
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _compute_loss(self, output, labels, frame_labels):
        sample_loss = F.cross_entropy(output["final_logits"].float(), labels)
        frame_loss = F.cross_entropy(output["final_frame_logits"].float().reshape(-1, 16), frame_labels.reshape(-1), ignore_index=-100,)
        total_loss = self.sample_loss_weight * sample_loss + self.frame_loss_weight * frame_loss
        return total_loss, sample_loss, frame_loss

    @staticmethod
    def _update_metrics(output, labels, frame_labels, temporal_mask, metrics):
        sample_preds = output["final_logits"].float().argmax(dim=-1)
        metrics["sample_correct"] += (sample_preds == labels).sum().item()
        metrics["sample_total"] += labels.size(0)
        frame_preds = output["final_frame_logits"].float().argmax(dim=-1)
        valid_mask = temporal_mask.bool() & (frame_labels != -100)
        if valid_mask.any():
            valid_preds = frame_preds[valid_mask]
            valid_labels = frame_labels[valid_mask]
            metrics["frame_correct"] += (valid_preds == valid_labels).sum().item()
            metrics["frame_total"] += valid_labels.numel()

    @staticmethod
    def _calculate_accuracy(correct, total):
        return correct / max(total, 1)

    def train_one_epoch(self, loader):
        self.model.train()
        loss_names = ("loss", "final_loss", "frame_loss")
        loss_totals = {name: 0.0 for name in loss_names}
        sample_correct = 0
        sample_total = 0
        frame_predictions = {}  # 跨 batch 聚合
        progress = tqdm(loader, desc="Training")
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            temporal_mask = batch["temporal_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).long()
            frame_labels = batch["frame_labels"].to(self.device, non_blocking=True).long()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model({"images": images, "temporal_mask": temporal_mask}, return_dict=True)
                total_loss, sample_loss, frame_loss = self._compute_loss(output, labels, frame_labels)
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_size = labels.size(0)
            for name, value in zip(loss_names, (total_loss, sample_loss, frame_loss)):
                loss_totals[name] += value.item() * batch_size
            sample_preds = output["final_logits"].float().argmax(dim=-1)
            sample_correct += (sample_preds == labels).sum().item()
            sample_total += batch_size
            frame_probs = output["final_frame_log_probs"].float().exp().detach().cpu()
            frame_labels_cpu = frame_labels.detach().cpu()
            temporal_mask_cpu = temporal_mask.detach().cpu()
            for batch_index, embryo in enumerate(batch["embryo"]):
                runs = batch["runs"][batch_index]
                for time_index, run_id in enumerate(runs):
                    if time_index >= frame_labels_cpu.size(1) or not temporal_mask_cpu[batch_index, time_index]:
                        continue
                    frame_label = int(frame_labels_cpu[batch_index, time_index])
                    if frame_label < 0:
                        continue
                    key = (embryo, int(run_id))
                    probability = frame_probs[batch_index, time_index]
                    if key not in frame_predictions:
                        frame_predictions[key] = {"prob_sum": probability.clone(), "count": 1, "label": frame_label}
                    else:
                        if frame_predictions[key]["label"] != frame_label:
                            raise ValueError(f"Inconsistent labels for frame {key}.")
                        frame_predictions[key]["prob_sum"] += probability
                        frame_predictions[key]["count"] += 1
            progress.set_postfix(
                loss=f"{total_loss.item():.4f}",
                final_loss=f"{sample_loss.item():.4f}",
                frame_loss=f"{frame_loss.item():.4f}",
                sample_acc=f"{sample_correct / max(sample_total, 1):.4f}",
            )
        frame_correct = sum(
            int((item["prob_sum"] / item["count"]).argmax().item() == item["label"])
            for item in frame_predictions.values()
        )
        frame_total = len(frame_predictions)
        sample_total = max(sample_total, 1)
        return {
            "loss": loss_totals["loss"] / sample_total,
            "final_loss": loss_totals["final_loss"] / sample_total,
            "frame_loss": loss_totals["frame_loss"] / sample_total,
            "sample_accuracy": sample_correct / sample_total,
            "frame_accuracy": frame_correct / max(frame_total, 1),
            "sample_correct": sample_correct,
            "sample_total": sample_total,
            "frame_correct": frame_correct,
            "frame_total": frame_total,
        }