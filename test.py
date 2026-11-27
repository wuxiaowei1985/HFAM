import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from configs import config as cfg
from dataset.loader import test_loader
from utils.get_model import build_model
from dataset.labels import LABEL_NAMES

CLASSES_NUM = len(LABEL_NAMES)

def calculate_class_accuracy(labels, predictions, metric_name):
    result = []
    for class_index, class_name in enumerate(LABEL_NAMES):
        class_indices = [index for index, label in enumerate(labels) if label == class_index]
        total = len(class_indices)
        correct = sum(predictions[index] == class_index for index in class_indices)
        accuracy = correct / total * 100 if total else 0.0
        result.append({
            "Stage": class_name,
            "Correct": correct,
            "Total": total,
            "Accuracy(%)": round(accuracy, 2)
        })
    return pd.DataFrame(result)

def save_metrics(labels, predictions, prefix, class_accuracy_path=None, report_path=None, cm_path=None):
    if not labels:
        raise ValueError(f"No valid {prefix} labels were collected.")
    class_accuracy = calculate_class_accuracy(labels, predictions, prefix)
    report_dict = classification_report(labels, predictions, labels=list(range(CLASSES_NUM)), target_names=LABEL_NAMES, digits=4, zero_division=0, output_dict=True)
    report_text = classification_report(labels, predictions, labels=list(range(CLASSES_NUM)), target_names=LABEL_NAMES, digits=4, zero_division=0)
    confusion = confusion_matrix(labels, predictions, labels=list(range(CLASSES_NUM)))
    print(f"\n{prefix} Classification Report\n")
    print(report_text)
    print(f"{prefix} Accuracy: {(sum(int(gt == pred) for gt, pred in zip(labels, predictions)) / len(labels)) * 100:.2f}%")
    if class_accuracy_path is not None:
        class_accuracy.to_csv(class_accuracy_path, index=False, encoding="utf-8-sig")
    if report_path is not None:
        rows = []
        for class_name in LABEL_NAMES:
            metrics = report_dict[class_name]
            rows.append({
                "class": class_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
                "support": metrics["support"]
            })
        for avg_type in ("macro avg", "weighted avg"):
            metrics = report_dict[avg_type]
            rows.append({
                "class": avg_type,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
                "support": metrics["support"]
            })
        pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig")
    if cm_path is not None:
        cm_df = pd.DataFrame(confusion, index=LABEL_NAMES, columns=LABEL_NAMES)
        cm_df.to_csv(cm_path, encoding="utf-8-sig")
    return sum(int(gt == pred) for gt, pred in zip(labels, predictions)) / len(labels)

def main():
    cfg.RUN_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg.CURRENT_MODEL)
    print("=" * 70)
    print("Loading model")
    print("=" * 70)
    checkpoint = torch.load(cfg.TEST_MODEL_DIR, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.to(cfg.DEVICE)
    model.eval()
    print("Model loaded successfully.")
    use_amp = cfg.DEVICE.type == "cuda"
    sample_labels = []
    sample_predictions = []
    sample_records = []
    frame_predictions = {}
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(cfg.DEVICE, non_blocking=True)
            temporal_mask = batch["temporal_mask"].to(cfg.DEVICE, non_blocking=True)
            labels = batch["label"].to(cfg.DEVICE, non_blocking=True).long()
            with torch.autocast(device_type=cfg.DEVICE.type, dtype=torch.float16, enabled=use_amp):
                output = model({"images": images, "temporal_mask": temporal_mask, "label": labels}, return_dict=True)
            sample_probs = output["final_log_probs"].float().exp()
            sample_preds = sample_probs.argmax(dim=1)
            frame_probs = output["final_frame_log_probs"].float().exp().cpu()
            frame_labels = batch["frame_labels"].cpu()
            temporal_mask_cpu = batch["temporal_mask"].cpu()
            for batch_index, (embryo, runs) in enumerate(zip(batch["embryo"], batch["runs"])):
                true_label = int(labels[batch_index])
                pred_label = int(sample_preds[batch_index])
                target_run = int(batch["target_run"][batch_index])
                sample_labels.append(true_label)
                sample_predictions.append(pred_label)
                sample_records.append({
                    "Embryo": embryo,
                    "Target_RUN": target_run,
                    "True_Label": LABEL_NAMES[true_label],
                    "True_Label_ID": true_label,
                    "Pred_Label": LABEL_NAMES[pred_label],
                    "Pred_Label_ID": pred_label,
                    "Correct": int(true_label == pred_label)
                })
                for time_index, run_id in enumerate(runs):
                    if time_index >= frame_labels.size(1) or not temporal_mask_cpu[batch_index, time_index]:
                        continue
                    frame_label = int(frame_labels[batch_index, time_index])
                    if frame_label < 0:
                        continue
                    key = (embryo, int(run_id))
                    probability = frame_probs[batch_index, time_index]
                    if key not in frame_predictions:
                        frame_predictions[key] = {
                            "prob_sum": probability.clone(),
                            "count": 1,
                            "label": frame_label
                        }
                    else:
                        if frame_predictions[key]["label"] != frame_label:
                            raise ValueError(f"Inconsistent labels for frame {key}.")
                        frame_predictions[key]["prob_sum"] += probability
                        frame_predictions[key]["count"] += 1
    frame_labels_list = []
    frame_predictions_list = []
    frame_records = []
    for (embryo, run_id), item in frame_predictions.items():
        mean_probability = item["prob_sum"] / item["count"]
        pred_label = int(mean_probability.argmax())
        true_label = item["label"]
        frame_labels_list.append(true_label)
        frame_predictions_list.append(pred_label)
        frame_records.append({
            "Embryo": embryo,
            "RUN": run_id,
            "True_Label": LABEL_NAMES[true_label],
            "True_Label_ID": true_label,
            "Pred_Label": LABEL_NAMES[pred_label],
            "Pred_Label_ID": pred_label,
            "Window_Count": item["count"],
            "Correct": int(true_label == pred_label)
        })
    sample_accuracy = save_metrics(
        sample_labels,
        sample_predictions,
        "Sample",
        class_accuracy_path=cfg.SAVE_RESULT_DIR,
        report_path=cfg.SAVE_REPORT_DIR,
        cm_path=cfg.SAVE_CM_DIR
    )
    frame_accuracy = save_metrics(
        frame_labels_list,
        frame_predictions_list,
        "Frame",
        class_accuracy_path=cfg.RUN_DIR / "frame_class_accuracy.csv",
        report_path=cfg.RUN_DIR / "frame_classification_report.csv",
        cm_path=cfg.RUN_DIR / "frame_confusion_matrix.csv"
    )
    pd.DataFrame(sample_records).to_csv(cfg.RUN_DIR / "sample_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(frame_records).to_csv(cfg.RUN_DIR / "frame_predictions.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 70)
    print(f"Sample Accuracy : {sample_accuracy * 100:.2f}% ({len(sample_labels)} samples)")
    print(f"Frame Accuracy  : {frame_accuracy * 100:.2f}% ({len(frame_labels_list)} unique frames)")
    print("=" * 70)
    print("Testing completed.")

if __name__ == "__main__":
    main()