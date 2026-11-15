import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.embryo_dataset import TemporalEmbryoDataset
from dataset.utils.collate_fn import temporal_embryo_collate_fn
from dataset.utils.split import split_embryos
from dataset.utils.transforms import TemporalFocusTransform, TemporalFocusValTransform
from configs import config as cfg

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(cfg.SEED)

train_embryos, val_embryos, test_embryos = split_embryos(cfg.DATA_ROOT, seed=cfg.SEED)

train_dataset = TemporalEmbryoDataset(
    root=cfg.DATA_ROOT,
    drop_non_contiguous=False,
    transform=TemporalFocusTransform(),
    embryo_list=train_embryos
)

val_dataset = TemporalEmbryoDataset(
    root=cfg.DATA_ROOT,
    drop_non_contiguous=False,
    transform=TemporalFocusValTransform(),
    embryo_list=val_embryos
)

test_dataset = TemporalEmbryoDataset(
    root=cfg.DATA_ROOT,
    drop_non_contiguous=False,
    transform=TemporalFocusValTransform(),
    embryo_list=test_embryos
)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.NUM_WORKERS,
    collate_fn=temporal_embryo_collate_fn,
    generator=g,
    worker_init_fn=seed_worker,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    collate_fn=temporal_embryo_collate_fn,
    generator=g,
    worker_init_fn=seed_worker,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    collate_fn=temporal_embryo_collate_fn,
    generator=g,
    worker_init_fn=seed_worker,
    pin_memory=torch.cuda.is_available()
)

print(f"Train temporal samples: {len(train_dataset)}")
print(f"Train dropped non-contiguous windows: {train_dataset.dropped_windows}")
print(f"Train padded samples: {train_dataset.padded_samples}")
print(f"Val temporal samples: {len(val_dataset)}")
print(f"Val dropped non-contiguous windows: {val_dataset.dropped_windows}")
print(f"Val padded samples: {val_dataset.padded_samples}")
print(f"Test temporal samples: {len(test_dataset)}")
print(f"Test dropped non-contiguous windows: {test_dataset.dropped_windows}")
print(f"Test padded samples: {test_dataset.padded_samples}")