import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class FocusTransform:
    def __init__(self, image_size=224):
        if image_size < 128:
            raise ValueError("image_size must be >= 128.")
        self.image_size = image_size

    def _transform_stack(self, images):
        if len(images) != 7:
            raise ValueError(f"Expected 7 focal planes, got {len(images)}")
        output = []
        for img in images:
            if img.size != (500, 500):
                raise ValueError(f"Expected native 500x500 images, got {img.size}")
            img = TF.resize(img, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
            tensor = TF.to_tensor(img)
            tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])
            output.append(tensor)
        return torch.stack(output, dim=0)

    def __call__(self, images):
        return self._transform_stack(images)

class TemporalFocusTransform:
    def __init__(self, image_size=224):
        if image_size < 128:
            raise ValueError("image_size must be >= 128.")
        self.image_size = image_size
        self.focus_transform = FocusTransform(image_size=image_size)

    def __call__(self, sequence_images):
        if len(sequence_images) == 0:
            raise ValueError("Temporal sequence cannot be empty.")
        output = [self.focus_transform(images) for images in sequence_images]
        return torch.stack(output, dim=0)

class FocusValTransform(FocusTransform):
    pass

class TemporalFocusValTransform(TemporalFocusTransform):
    pass