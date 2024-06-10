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
        
        outputs = {}

        def get_hook(name):
            def hook(model, input, output):
                outputs[name] = output
            return hook
        
        vgg[8].register_forward_hook(get_hook('layer8'))
        vgg[17].register_forward_hook(get_hook('layer17'))
        vgg[35].register_forward_hook(get_hook('layer35'))
        
        self.feature_extractor = vgg

        '''
        self.feature_extractor = nn.Sequential(
            *list(vgg.children())[:20]
        )
        self.feature_extractor.eval()
        '''
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def content_loss(self, srs, hrs):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        srs = preprocess(srs)
        hrs = preprocess(hrs)
        
        srs = srs.unsqueeze(0)  # Add batch dimension
        hrs = hrs.unsqueeze(0)  # Add batch dimension
        
        srs_features = self.feature_extractor(srs)
        hrs_features = self.feature_extractor(hrs)

        loss = 0.0
        for srs_feature, hrs_feature in zip(srs_features, hrs_features):
            loss += self.content_loss_type(hrs_feature / 12.75, srs_feature / 12.75)

        return lo
    