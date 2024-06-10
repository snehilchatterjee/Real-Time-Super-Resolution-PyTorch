from layers import *
import torch
from torch import nn
from data import HR_SIZE

GEN_FILTERS = 64
DISC_FILTERS = 64

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = Conv2DBlock(in_channels = 3,out_channels = GEN_FILTERS, kernel_size = 3, stride = 1, padding = 'same', batchnorm = False)
        self.conv2 = Conv2DBlock(in_channels = GEN_FILTERS, out_channels = GEN_FILTERS, kernel_size = 1, stride = 1, padding = 'valid', batchnorm = False)
        
        self.rrdb1 = RRDBlock(in_channels = GEN_FILTERS,filters = GEN_FILTERS)
        self.rrdb2 = RRDBlock(in_channels = GEN_FILTERS,filters = GEN_FILTERS)
        self.rrdb3 = RRDBlock(in_channels = GEN_FILTERS,filters = GEN_FILTERS)
        self.rrdb4 = RRDBlock(in_channels = GEN_FILTERS,filters = GEN_FILTERS)
        
        self.upsample1 = PixelShuffleUpsampling(in_channels = GEN_FILTERS, out_channels = GEN_FILTERS * 4, upscale_factor = 2)
        self.upsample2 = PixelShuffleUpsampling(in_channels = GEN_FILTERS, out_channels = GEN_FILTERS * 4, upscale_factor = 2)
        
        self.conv3 = Conv2DBlock(in_channels = GEN_FILTERS, out_channels = GEN_FILTERS, batchnorm = False)
        self.conv4 = Conv2DBlock(in_channels = GEN_FILTERS, out_channels = 3,kernel_size = 3 , batchnorm = False, activate = False)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x1 = self.rrdb1(x)
        x2 = self.rrdb2(x1)
        x3 = self.rrdb3(x2)
        x4 = self.rrdb4(x3)
        
        up_x1 = self.upsample1(x4)
        up_x2 = self.upsample2(up_x1)
        
        x = self.conv3(up_x2)
        x = self.conv4(x)
        x = self.tanh(x)
        
        return x
    
    
model=Generator()

print(model(torch.rand(1,3,28,28)).shape)
