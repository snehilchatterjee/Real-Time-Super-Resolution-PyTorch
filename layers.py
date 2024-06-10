import torch
from torch import nn

class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=True, **kwargs):
        super(DepthwiseSeparableConv2D, self).__init__()

        #if padding == 'same':
        #    padding = kernel_size // 2

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

        nn.init.xavier_normal_(self.depthwise_conv.weight)
        nn.init.xavier_normal_(self.pointwise_conv.weight)
        nn.init.zeros_(self.depthwise_conv.bias)
        nn.init.zeros_(self.pointwise_conv.bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class Conv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same',bias=True, **kwargs):
        if padding == 'same':
            padding = kernel_size // 2
        super(Conv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            **kwargs
        )
        
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)
        


class Conv2DBlock(nn.Module):
    def __init__(self,in_channels,out_channels,batchnorm=True,activate=True,**kwargs):
        super(Conv2DBlock, self).__init__()
        
        self.conv=Conv2D(in_channels=in_channels,out_channels=out_channels,**kwargs)
        self.batchnorm=nn.BatchNorm2d(out_channels) if batchnorm else None
        self.activate=nn.PReLU(num_parameters=1,init=0) if activate else None           # IN PAPER LReLU IS USED
    
    def forward(self,x):
        x=self.conv(x)
        if self.batchnorm is not None:
            x=self.batchnorm(x)
        if self.activate is not None:
            x=self.activate(x)
            
        return x

#conv_block = Conv2DBlock(in_channels=3,out_channels=64, batchnorm=True, activate=True, kernel_size=3, stride=1, padding=1)

#print(conv_block(torch.rand(1,3,224,224)).shape)

class PixelShuffleUpsampling(nn.Module):
    def __init__(self,in_channels,out_channels,upscale_factor=2):
        super(PixelShuffleUpsampling, self).__init__()
        
        self.conv=Conv2DBlock(in_channels=in_channels,out_channels=out_channels,batchnorm=False,activate=False)
        self.upsample=nn.PixelShuffle(upscale_factor)
        self.prelu=nn.PReLU(num_parameters=1,init=0)
        
    def forward(self,x):
        x=self.conv(x)
        x=self.upsample(x)
        x=self.prelu(x)
    
        return x
    
class ResidualDenseBlock(nn.Module):            # NOT ENTIRELY DENSE ALSO LESS NUMBER OF CONV2D BLOCKS ARE USED
    def __init__(self,in_channels,filters=64):
        super(ResidualDenseBlock, self).__init__()     
        self.conv1=Conv2DBlock(in_channels,filters//2)
        self.conv2=Conv2DBlock(in_channels+filters//2,filters//2)
        self.conv3=Conv2DBlock(filters,filters,activate=False) 
        
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(torch.concat([x1,x],dim=1))
        x3=self.conv3(torch.concat([x2,x1],dim=1))  
        
        return x3+x
    
class RRDBlock(nn.Module):
    def __init__(self, in_channels, filters, **kwargs):
        super(RRDBlock, self).__init__()

        self.rdb_1 = ResidualDenseBlock(in_channels, filters)
        self.rdb_2 = ResidualDenseBlock(filters, filters)
        self.rdb_3 = ResidualDenseBlock(filters, filters)

        self.beta_input = nn.Parameter(torch.tensor(1.0))                  # IN THE PAPER IT IS ALWAYS 1 (NOT LEARNABLE)
        self.rrdb_inputs_scales = torch.ones(1, filters, 1, 1) * self.beta_input
        
        self.beta_output = nn.Parameter(torch.tensor(0.5))
        self.rrdb_outputs_scales = torch.ones(1, filters, 1, 1) * self.beta_output

    def forward(self, x):
        x1 = self.rdb_1(x)
        x2 = self.rdb_2(x1)
        out = self.rdb_3(x2)
        return (self.rrdb_inputs_scales.to(x.device) * x) + (self.rrdb_outputs_scales.to(x.device) * out)
