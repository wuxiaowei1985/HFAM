import torch

def temporal_embryo_collate_fn(batch):
    images = []
    temporal_masks = []
    for sample in batch:
        sample_images = sample["images"]
        frame_mask = torch.tensor(sample["frame_mask"], dtype=torch.bool)
        if sample["padding_length"] > 0:
            pad_shape = (sample["padding_length"],) + tuple(sample_images.shape[1:])
            padding = torch.zeros(pad_shape, dtype=sample_images.dtype)
            sample_images = torch.cat([sample_images, padding], dim=0)
            frame_mask = torch.cat([frame_mask, torch.zeros(sample["padding_length"], dtype=torch.bool)], dim=0)
        images.append(sample_images)
        temporal_masks.append(frame_mask)
    images = torch.stack(images, dim=0)
    temporal_mask = torch.stack(temporal_masks, dim=0)
    embryos = [sample["embryo"] for sample in batch]
    image_names = [sample["image_names"] for sample in batch]
    runs = [sample["runs"] for sample in batch]
    target_runs = torch.tensor([sample["target_run"] for sample in batch], dtype=torch.long)
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    return {
        "images": images,
        "temporal_mask": temporal_mask,
        "embryo": embryos,
        "image_names": image_names,
        "runs": runs,
        "target_run": target_runs,
        "label": labels
    }