import torch
import torch.nn.functional as F
from tqdm import tqdm

class Validator:
    def __init__(self, model, criterion, device, final_loss_weight=1.0, frame_loss_weight=1.0, label_smoothing=0.0):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.final_loss_weight = final_loss_weight
        self.frame_loss_weight = frame_loss_weight
        self.label_smoothing = label_smoothing
        self.use_amp = device.type == "cuda"

    def _compute_loss(self, output, labels, frame_labels):
        final_loss = F.cross_entropy(output["final_logits"].float(), labels, label_smoothing=self.label_smoothing)
        frame_loss = F.cross_entropy(output["final_frame_logits"].float().reshape(-1, 16), frame_labels.reshape(-1), ignore_index=-100, label_smoothing=self.label_smoothing,)
        total_loss = self.final_loss_weight * final_loss + self.frame_loss_weight * frame_loss
        return total_loss, final_loss, frame_loss

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        loss_names = ("loss", "final_loss", "frame_loss")
        loss_totals = {name: 0.0 for name in loss_names}
        sample_correct = 0
        sample_total = 0
        frame_predictions = {}
        progress = tqdm(loader, desc="Validation")
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            temporal_mask = batch["temporal_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).long()
            frame_labels = batch["frame_labels"].to(self.device, non_blocking=True).long()
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model({"images": images, "temporal_mask": temporal_mask}, return_dict=True)
                losses = self._compute_loss(output, labels, frame_labels)
            total_loss, final_loss, frame_loss = losses
            batch_size = labels.size(0)
            for name, value in zip(loss_names, losses):
                loss_totals[name] += value.item() * batch_size
            sample_preds = output["final_logits"].float().argmax(dim=1)
            sample_correct += (sample_preds == labels).sum().item()
            sample_total += batch_size
            frame_probs = output["final_frame_log_probs"].float().exp().cpu()
            frame_labels_cpu = frame_labels.cpu()
            temporal_mask_cpu = temporal_mask.cpu()
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
                final=f"{final_loss.item():.4f}",
                frame=f"{frame_loss.item():.4f}",
                sample_acc=f"{sample_correct / max(sample_total, 1):.4f}",
            )
        frame_correct = sum(
            int((item["prob_sum"] / item["count"]).argmax().item() == item["label"])
            for item in frame_predictions.values()
        )
        sample_total = max(sample_total, 1)
        frame_total = len(frame_predictions)
        return {
            **{name: value / sample_total for name, value in loss_totals.items()},
            "sample_accuracy": sample_correct / sample_total,
            "frame_accuracy": frame_correct / max(frame_total, 1),
            "sample_correct": sample_correct,
            "sample_total": sample_total,
            "frame_correct": frame_correct,
            "frame_total": frame_total,
        }