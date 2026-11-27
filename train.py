import torch
import torch.nn as nn
from utils.seed import seed_everything
from configs import config as cfg
from trainer.trainer import Trainer
from trainer.validate import Validator
from dataset.loader import train_loader, val_loader
from utils.early_stopping import EarlyStopping
from utils.history import History
from utils.plot import plot_training_curve
from utils.get_model import build_model

def print_metrics(name, metrics):
    print(f"{name} Total Loss       : {metrics['loss']:.4f}")
    print(f"{name} Final Loss       : {metrics['final_loss']:.4f}")
    print(f"{name} Coarse Loss      : {metrics['coarse_loss']:.4f}")
    print(f"{name} Frame Loss       : {metrics['frame_loss']:.4f}")
    print(f"{name} Coarse Frame Loss: {metrics['coarse_frame_loss']:.4f}")
    print(f"{name} Sample Accuracy  : {metrics['sample_accuracy']:.4f}")
    print(f"{name} Frame Accuracy   : {metrics['frame_accuracy']:.4f}")

def main():
    seed_everything(cfg.SEED)
    cfg.RUN_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TRAIN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg.CURRENT_MODEL).to(cfg.DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR)
    criterion = nn.NLLLoss()
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=cfg.DEVICE)
    validator = Validator(model=model, criterion=criterion, device=cfg.DEVICE)
    early_stopping = EarlyStopping(patience=cfg.EARLY_STOPPING_PATIENCE, min_delta=cfg.MIN_DELTA, save_path=cfg.BEST_MODEL)
    history = History()
    for epoch in range(cfg.EPOCHS):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{cfg.EPOCHS}")
        train_metrics = trainer.train_one_epoch(train_loader)
        val_metrics = validator.validate(val_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print("\n" + "-" * 70)
        print_metrics("Train", train_metrics)
        print("-" * 70)
        print_metrics("Val", val_metrics)
        print(f"Learning Rate        : {lr:.8f}")
        print("=" * 70)
        history.update(epoch=epoch + 1, train_metrics=train_metrics, val_metrics=val_metrics, lr=lr)
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }, cfg.LAST_MODEL)
        stop = early_stopping(val_acc=val_metrics["sample_accuracy"], model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
        if stop:
            print("Early stopping triggered.")
            break
    history.save(cfg.TRAIN_SAVE_DIR / "history.csv")
    plot_training_curve(cfg.TRAIN_SAVE_DIR / "history.csv", cfg.TRAIN_SAVE_DIR)
    print("Training completed.")

if __name__ == "__main__":
    main()