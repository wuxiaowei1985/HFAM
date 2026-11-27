
import pandas as pd


class History:
    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_sample_acc": [],
            "train_frame_acc": [],
            "val_sample_acc": [],
            "val_frame_acc": [],
            "lr": []
        }

    def update(self, epoch, train_metrics, val_metrics, lr):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_metrics["loss"])
        self.history["val_loss"].append(val_metrics["loss"])
        self.history["train_sample_acc"].append(train_metrics["sample_accuracy"])
        self.history["train_frame_acc"].append(train_metrics["frame_accuracy"])
        self.history["val_sample_acc"].append(val_metrics["sample_accuracy"])
        self.history["val_frame_acc"].append(val_metrics["frame_accuracy"])
        self.history["lr"].append(lr)

    def save(self, path):
        pd.DataFrame(self.history).to_csv(path, index=False)