import torch
import os
from data import *
from torch import nn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, high_res_path, low_res_path=None):
        self.high_res_path = high_res_path
        self.low_res_path = low_res_path
        
        self.high_dataset = os.listdir(high_res_path)
        if low_res_path is not None:
            self.low_dataset = os.listdir(low_res_path)
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        high_res = self.high_dataset[idx]
        if self.low_res_path is not None:
            low_res = self.low_dataset[idx]
        else:
            # Bicubic interpolation for low resolution
            hr_shape = high_res.shape
            low_res = F.interpolate(high_res.unsqueeze(0), size=(hr_shape[1] // SCALE, hr_shape[2] // SCALE), mode='bicubic', align_corners=False).squeeze(0)
        
        image_path = {'lr': low_res, 'hr': high_res}
        
        lr, hr = random_compression(image_path)
        lr, hr = random_crop(lr, hr)
        lr, hr = random_spatial_augmentation(lr, hr)
        
        return lr, hr