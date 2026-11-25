import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class FocusTransform:
    def __init__(self, image_size=224):
        if image_size < 128:
            raise ValueError("image_size must be >= 128.")
        self.image_size = image_size

    def _resize(self, image):
        return TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)

    def _transform_stack(self, images):
        if len(images) != 7:
            raise ValueError(f"Expected 7 focal planes, got {len(images)}.")
        output = []
        for image in images:
            if image.size != (500, 500):
                raise ValueError(f"Expected native 500x500 images, got {image.size}.")
            image = self._resize(image)
            tensor = TF.to_tensor(image)
            tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])
            output.append(tensor)
        return torch.stack(output, dim=0)

    def __call__(self, images):
        return self._transform_stack(images)

class TemporalFocusTransform:
    def __init__(self, image_size=224, horizontal_flip_prob=0.5, vertical_flip_prob=0.1, rotation_degree=10.0, brightness=0.1, contrast=0.1, noise_std=0.01):
        if image_size < 128:
            raise ValueError("image_size must be >= 128.")
        self.image_size = image_size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_degree = rotation_degree
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def _sample_parameters(self):
        horizontal_flip = random.random() < self.horizontal_flip_prob
        vertical_flip = random.random() < self.vertical_flip_prob
        angle = random.uniform(-self.rotation_degree, self.rotation_degree)
        brightness_factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
        contrast_factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
        return {
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
            "angle": angle,
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor
        }

    def _transform_image(self, image, parameters, add_noise=True):
        if image.size != (500, 500):
            raise ValueError(f"Expected native 500x500 images, got {image.size}.")
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        if parameters["horizontal_flip"]:
            image = TF.hflip(image)
        if parameters["vertical_flip"]:
            image = TF.vflip(image)
        image = TF.rotate(image, parameters["angle"], interpolation=InterpolationMode.BILINEAR, fill=0)
        image = TF.adjust_brightness(image, parameters["brightness_factor"])
        image = TF.adjust_contrast(image, parameters["contrast_factor"])
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])
        if add_noise and self.noise_std > 0:
            tensor = tensor + torch.randn_like(tensor) * self.noise_std
            tensor = tensor.clamp(-1.0, 1.0)
        return tensor

    def __call__(self, sequence_images):
        if len(sequence_images) == 0:
            raise ValueError("Temporal sequence cannot be empty.")
        parameters = self._sample_parameters()
        output = []
        for images in sequence_images:
            if images is None:
                output.append(torch.zeros( 7, 1, self.image_size, self.image_size, dtype=torch.float32))
                continue
            if len(images) != 7:
                raise ValueError(f"Expected 7 focal planes, got {len(images)}.")
            frame_output = []
            for image in images:
                frame_output.append(self._transform_image(image, parameters, add_noise=True))
            output.append(torch.stack(frame_output, dim=0))
        return torch.stack(output, dim=0)

class FocusValTransform(FocusTransform):
    pass

class TemporalFocusValTransform:
    def __init__(self, image_size=224):
        self.image_size = image_size

    def __call__(self, sequence_images):
        if len(sequence_images) == 0:
            raise ValueError("Temporal sequence cannot be empty.")
        output = []
        for images in sequence_images:
            if images is None:
                output.append(torch.zeros(7, 1, self.image_size, self.image_size, dtype=torch.float32))
                continue
            if len(images) != 7:
                raise ValueError(f"Expected 7 focal planes, got {len(images)}.")
            frame_output = []
            for image in images:
                if image.size != (500, 500):
                    raise ValueError(f"Expected native 500x500 images, got {image.size}.")
                image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
                tensor = TF.to_tensor(image)
                tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])
                frame_output.append(tensor)
            output.append(torch.stack(frame_output, dim=0))
        return torch.stack(output, dim=0)