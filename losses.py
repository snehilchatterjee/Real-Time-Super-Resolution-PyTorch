import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

class PixelLossTraining(nn.Module):
    def __init__(self, pixel_loss):
        if pixel_loss == 'l1':
            self.pixel_loss_type = nn.L1Loss()
        elif pixel_loss == 'l2':
            self.pixel_loss_type = nn.MSELoss()
            
    def __call__(self, srs, hrs):
        return self.pixel_loss_type(hrs, srs)


class VGGContentTraining:
    def setup_content_loss(self, content_loss):
        if content_loss == 'l1':
            self.content_loss_type = nn.L1Loss()
        elif content_loss == 'l2':
            self.content_loss_type = nn.MSELoss()
        
        vgg = models.vgg19(pretrained=True).features
        
        # Remove activations for specified layers
        vgg[8] = nn.Identity()
        vgg[17] = nn.Identity()
        vgg[35] = nn.Identity()
        
        self.feature_extractor = vgg
        
        self.outputs = {}

        def get_hook(name):
            def hook(model, input, output):
                self.outputs[name] = output
            return hook
        
        vgg[8].register_forward_hook(get_hook('layer8'))
        vgg[17].register_forward_hook(get_hook('layer17'))
        vgg[35].register_forward_hook(get_hook('layer35'))
        
        self.feature_extractor.eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def content_loss(self, srs, hrs):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224))])
        
        srs = preprocess(srs)
        hrs = preprocess(hrs)
        
        
        srs_features = self.feature_extractor(srs)
        val1 = self.outputs['layer8']
        val2= self.outputs['layer17']
        val3 = self.outputs['layer35']
        hrs_features = self.feature_extractor(hrs)
        val4 = self.outputs['layer8']
        val5= self.outputs['layer17']
        val6 = self.outputs['layer35']
        
        #print(torch.sum(val1)==torch.sum(val4), torch.sum(val2)==torch.sum(val5), torch.sum(val3)==torch.sum(val6))
        
        loss = 0.0
        loss += self.content_loss_type(val4 / 12.75, val1 / 12.75)
        loss += self.content_loss_type(val5 / 12.75, val2 / 12.75)
        loss += self.content_loss_type(val6 / 12.75, val3 / 12.75)
        
        return loss
    