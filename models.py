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
        x = x / 255.0
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
        
        x = (x + 1) * 127.5
        return x
    

#generator=Generator()
#print(generator(torch.rand(1,3,HR_SIZE,HR_SIZE)).shape)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2)    
        self.bnorm1 = nn.BatchNorm2d(num_features = DISC_FILTERS//2)
        self.bnorm2 = nn.BatchNorm2d(num_features = DISC_FILTERS)
        self.bnorm3 = nn.BatchNorm2d(num_features = DISC_FILTERS*2)
        self.bnorm4 = nn.BatchNorm2d(num_features = DISC_FILTERS*4)
        
        self.conv1 = Conv2DBlock(in_channels = 3, out_channels = DISC_FILTERS//2)
        
        self.conv2 = Conv2D(in_channels = DISC_FILTERS//2, out_channels = DISC_FILTERS//2, kernel_size = 3, stride = 2)
        self.conv3 = Conv2D(in_channels = DISC_FILTERS//2, out_channels = DISC_FILTERS, kernel_size = 3)
        self.conv4 = Conv2D(in_channels = DISC_FILTERS, out_channels = DISC_FILTERS, kernel_size = 3, stride = 2)
        self.conv5 = Conv2D(in_channels = DISC_FILTERS, out_channels = DISC_FILTERS * 2, kernel_size = 3, stride = 1)
        self.conv6 = Conv2D(in_channels = DISC_FILTERS * 2, out_channels = DISC_FILTERS * 2, kernel_size = 3, stride = 2)
        self.conv7 = Conv2D(in_channels = DISC_FILTERS * 2, out_channels = DISC_FILTERS * 4, kernel_size = 3, stride = 1)
        self.conv8 = Conv2D(in_channels = DISC_FILTERS * 4, out_channels = DISC_FILTERS * 4, kernel_size = 3, stride = 2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(DISC_FILTERS * 4 * (HR_SIZE // 16) * (HR_SIZE // 16), 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = x / 127.5 - 1
        x = self.conv1(x)
        x = self.lrelu(x)
        
        x = self.conv2(x)
        x = self.bnorm1(x)
        x = self.lrelu(x)
        
        x = self.conv3(x)
        x = self.bnorm2(x)
        x = self.lrelu(x)
        
        x = self.conv4(x)
        x = self.bnorm2(x)
        x = self.lrelu(x)
        
        x = self.conv5(x)
        x = self.bnorm3(x)
        x = self.lrelu(x)
        
        x = self.conv6(x)
        x = self.bnorm3(x)
        x = self.lrelu(x)
        
        x = self.conv7(x)
        x = self.bnorm4(x)
        x = self.lrelu(x)
        
        x = self.conv8(x)
        x = self.bnorm4(x)
        x = self.lrelu(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = self.lrelu(x)
        logits = self.fc3(x)
        return logits
    
#discriminator=Discriminator()
#print(discriminator(torch.rand(16,3,128,128)).shape)