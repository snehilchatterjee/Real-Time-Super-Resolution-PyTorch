import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

HR_SIZE = 128
SCALE = 4
LR_SIZE = HR_SIZE // SCALE
BATCH_SIZE = 8

# Random Compressions
def random_compression(example):
    hr = example['hr']
    hr_shape = hr.shape
    compression_idx = np.random.randint(0, 7)
    
    if compression_idx == 0 or compression_idx == 1:
        # bicubic
        lr = F.interpolate(hr.unsqueeze(0), size=(hr_shape[0] // SCALE, hr_shape[1] // SCALE), mode='bicubic', align_corners=False).squeeze(0)
    elif compression_idx == 2 or compression_idx == 3:
        # bilinear
        lr = F.interpolate(hr.unsqueeze(0), size=(hr_shape[0] // SCALE, hr_shape[1] // SCALE), mode='bilinear', align_corners=False).squeeze(0)
    elif compression_idx == 4 or compression_idx == 5:
        # nearest
        lr = F.interpolate(hr.unsqueeze(0), size=(hr_shape[0] // SCALE, hr_shape[1] // SCALE), mode='nearest').squeeze(0)
    else:
        # default
        lr = example['lr']
    
    lr = torch.clamp(lr, 0, 255).round().byte()
    return lr, hr

# Spatial Random Augmentations

def random_crop(lr, hr):
    lr_shape = lr.shape[-2:]
    lr_w = np.random.randint(0, lr_shape[1] - LR_SIZE + 1)
    lr_h = np.random.randint(0, lr_shape[0] - LR_SIZE + 1)

    hr_w = lr_w * SCALE
    hr_h = lr_h * SCALE

    lr_cropped = lr[..., lr_h:lr_h + LR_SIZE, lr_w:lr_w + LR_SIZE]
    hr_cropped = hr[..., hr_h:hr_h + HR_SIZE, hr_w:hr_w + HR_SIZE]

    return lr_cropped, hr_cropped

def random_rotate(lr, hr):
    rn = np.random.randint(0, 4)
    lr_rot = torch.rot90(lr, rn, [1, 2])
    hr_rot = torch.rot90(hr, rn, [1, 2])
    return lr_rot, hr_rot

def random_spatial_augmentation(lrs, hrs):
    if np.random.rand() < 0.5:
        lrs, hrs = random_rotate(lrs, hrs)

    return lrs.float(), hrs.float()

def visualize_samples(images_lists, titles=None, size=(12, 12)):
    assert len(images_lists) == len(titles)
    
    cols = len(images_lists)
    
    for images in zip(*images_lists):
        plt.figure(figsize=size)
        for idx, image in enumerate(images):
            plt.subplot(1, cols, idx + 1)
            plt.imshow(torch.clamp(image, 0, 255).round().byte().permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            if titles:
                plt.title(titles[idx])
        plt.show()
