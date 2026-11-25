import torch

def temporal_embryo_collate_fn(batch):
    images = []
    temporal_masks = []
    frame_labels = []
    for sample in batch:
        sample_images = sample["images"]
        frame_mask = torch.tensor(sample["frame_mask"], dtype=torch.bool)
        sample_frame_labels = torch.tensor(sample["frame_labels"], dtype=torch.long)
        if sample["padding_length"] > 0:
            pad_shape = (sample["padding_length"],) + tuple(sample_images.shape[1:])
            padding = torch.zeros(pad_shape, dtype=sample_images.dtype)
            sample_images = torch.cat([sample_images, padding], dim=0)
            frame_mask = torch.cat([frame_mask, torch.zeros(sample["padding_length"], dtype=torch.bool)], dim=0)
            sample_frame_labels = torch.cat([sample_frame_labels, torch.full((sample["padding_length"],), -100, dtype=torch.long)], dim=0)
        sample_frame_labels = torch.where(frame_mask, sample_frame_labels, torch.full_like(sample_frame_labels, -100))
        images.append(sample_images)
        temporal_masks.append(frame_mask)
        frame_labels.append(sample_frame_labels)
    return {
        "images": torch.stack(images, dim=0),
        "temporal_mask": torch.stack(temporal_masks, dim=0),
        "frame_labels": torch.stack(frame_labels, dim=0),
        "embryo": [sample["embryo"] for sample in batch],
        "image_names": [sample["image_names"] for sample in batch],
        "runs": [sample["runs"] for sample in batch],
        "target_run": torch.tensor([sample["target_run"] for sample in batch], dtype=torch.long),
        "label": torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    }