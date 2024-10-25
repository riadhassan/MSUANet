import torch
import numpy as np
import torchvision.transforms as transforms
import random
import math

def augment_image_label(x, y, imsize=256, trans_threshold=0.0, horizontal_flip=None, rotation_range=None,
                        height_shift_range=None, width_shift_range=None, shear_range=None, zoom_range=None,
                        elastic=None, add_noise=None):
    # Convert numpy arrays to PyTorch tensors
    x = torch.from_numpy(x).float().unsqueeze(0)
    y = torch.from_numpy(y).float().unsqueeze(0)

    # Resize to (1, imsize, imsize)
    x = x.view(1, imsize, imsize)
    y = y.view(1, imsize, imsize)

    if horizontal_flip is not None and random.random() < trans_threshold:
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])

    if random.random() < trans_threshold:
        # Rotation
        if rotation_range is not None:
            angle = (-rotation_range, rotation_range)  # Provide as a tuple for the valid range
        else:
            angle = (0, 0)  # No rotation if rotation_range is None

        # Shifts
        if height_shift_range is not None:
            height_shift = random.uniform(-height_shift_range, height_shift_range) * imsize
        else:
            height_shift = 0

        if width_shift_range is not None:
            width_shift = random.uniform(-width_shift_range, width_shift_range) * imsize
        else:
            width_shift = 0

        # Shear
        if shear_range is not None:
            shear = random.uniform(-shear_range, shear_range)
        else:
            shear = 0

        # Zoom
        if zoom_range is not None and zoom_range[0] != 1 and zoom_range[1] != 1:
            scale = random.uniform(zoom_range[0], zoom_range[1])
        else:
            scale = 1.0

        # Apply the transformations
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=angle, translate=(width_shift_range, height_shift_range), scale=(scale, scale), shear=shear)
        ])
        
        x = transform(x)
        y = transform(y)

    # Add noise
    if add_noise is not None and random.random() < trans_threshold:
        noise = torch.randn_like(x) * 0.15 * x.std()
        x = x + noise

    # Return reshaped images
    return x.view(1, imsize, imsize), y.view(1, imsize, imsize)
