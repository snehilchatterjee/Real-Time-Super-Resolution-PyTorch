import torch
import cv2
import os
from data import *
from torch import nn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, high_res_path, low_res_path=None):
        self.high_res_path = high_res_path
        self.low_res_path = low_res_path
        
        self.high_dataset = os.listdir(high_res_path)
        self.high_dataset.sort()
        if low_res_path is not None:
            self.low_dataset = os.listdir(low_res_path)
            self.low_dataset.sort()
        
    def __len__(self):
        return len(self.high_dataset)

    def __getitem__(self, idx):
        high_res_filename = self.high_dataset[idx] 
        high_res_path = os.path.join(self.high_res_path, high_res_filename)
        high_res = cv2.imread(high_res_path)
        high_res = torch.tensor(high_res).permute(2,0,1)
        
        if self.low_res_path is not None:
            low_res_filename = self.low_dataset[idx]
            low_res_path = os.path.join(self.low_res_path, low_res_filename)
            low_res = cv2.imread(low_res_path)
            low_res = torch.tensor(low_res).permute(2,0,1)     
        else:
            # Bicubic interpolation for low resolution
            hr_shape = high_res.shape
            low_res = F.interpolate(high_res.unsqueeze(0), size=(hr_shape[1] // SCALE, hr_shape[2] // SCALE), mode='bicubic', align_corners=False).squeeze(0)
        
        images = {'lr': low_res, 'hr': high_res}
        
        lr, hr = random_compression(images)
        lr, hr = random_crop(lr, hr)
        lr, hr = random_spatial_augmentation(lr, hr)
        
        return lr, hr