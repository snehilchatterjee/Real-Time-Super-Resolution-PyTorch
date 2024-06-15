import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models import *
from dataset import *
#from callbacks import *
from losses import *

train_dataset = Dataset('../../../DIV2K_Complete/DIV2K_train', '../../../DIV2K_Complete/DIV2K_train_LR_bicubic/X4')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

# Sample data inspection
for lrs, hrs in train_loader:
    break

print(lrs.shape, hrs.shape)
print(lrs.dtype, hrs.dtype)
print(torch.min(lrs), torch.max(lrs))
print(torch.min(hrs), torch.max(hrs))

visualize_samples(images_lists=(lrs[:15], hrs[:15]), titles=('Low Resolution', 'High Resolution'), size=(8, 8))

