import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models import *
from dataset import *
#from callbacks import *
from losses import *

torch.autograd.set_detect_anomaly(True)

train_dataset = Dataset('../../DIV2K_Complete/DIV2K_train', '../../DIV2K_Complete/DIV2K_train_LR_bicubic/X4')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

# Sample data inspection
for lrs, hrs in train_loader:
    break

print(lrs.shape, hrs.shape)
print(lrs.dtype, hrs.dtype)
print(torch.min(lrs), torch.max(lrs))
print(torch.min(hrs), torch.max(hrs))

#visualize_samples(images_lists=(lrs[:15], hrs[:15]), titles=('Low Resolution', 'High Resolution'), size=(8, 8))

class SRGAN(nn.Module):
    
    def __init__(self,
                generator,
                discriminator,
                generator_optimizer, 
                discriminator_optimizer,
                perceptual_finetune,
                pixel_loss,
                content_loss,
                adv_loss,
                loss_weights):
        
        super().__init__()
        
        self.generator = generator
        self.discriminator = discriminator
        
        self.optimizer_G = generator_optimizer
        self.optimizer_D = discriminator_optimizer
        
        self.perceptual_finetune = perceptual_finetune

        self.pixel_loss = PixelLossTraining(pixel_loss)
        self.content_loss = VGGContentTraining(content_loss)
        self.adv_loss = AdversarialTraining(adv_loss)        

        if self.perceptual_finetune:
            self.loss_weights = loss_weights

    def train_step(self, batch):
        self.lrs = batch[0]  # Assuming batch[0] is the low-resolution images
        self.hrs = batch[1]  # Assuming batch[1] is the high-resolution images

        if self.perceptual_finetune:
            
            self.discriminator.train()
            self.generator.train()

            self.srs = self.generator(self.lrs)

            ###################
            # (1) Update D network
            ###################
            
            self.optimizer_D.zero_grad()
            
            # train with real
            real_logits = self.discriminator(self.hrs)
            
            # train with fake
            fake_logits = self.discriminator(self.srs.detach())

            # Combined Loss            
            disc_adv_loss = self.adv_loss.disc_adv_loss(fake_logits, real_logits)
            disc_adv_loss.backward()
            self.optimizer_D.step()
            
            ###################
            # (2) Update G network
            ###################
            
            #self.generator.train()
            
            self.optimizer_G.zero_grad()
            
            fake_logits = self.discriminator(self.srs)
            real_logits = self.discriminator(self.hrs)
                       
            content_loss = self.loss_weights['content_loss'] * self.content_loss(self.srs, self.hrs.type(torch.float32))
            gen_adv_loss = self.loss_weights['adv_loss'] * self.adv_loss.gen_adv_loss(fake_logits, real_logits) # MIGHT CAUSE ERROR HERE
           
            # Combined Loss
            perceptual_loss = content_loss + gen_adv_loss
            gen_loss = perceptual_loss
            

            gen_loss.backward()
            self.optimizer_G.step()

            return {
                'Perceptual Loss': perceptual_loss.item(),
                'Content Loss': content_loss.item(),
                'Generator Adv Loss': gen_adv_loss.item(),
                'Discriminator Adv Loss': disc_adv_loss.item(),
            }
        
        else:
            # [=================== Training Generator Only ===================]

            self.generator.train()
            self.optimizer_G.zero_grad()

            self.srs = self.generator(self.lrs)

            pixel_loss = self.pixel_loss(self.srs, self.hrs)

            pixel_loss.backward()
            self.optimizer_G.step()

            return {
                'Pixel Loss': pixel_loss.item(),
            }

EPOCHS = 1000
LR = 0.002
BETA_1 = 0.9
BETA_2 = 0.999

PERCEPTUAL_FINETUNE = False
# first train the model for pixel loss
# once pixel loss is saturated, set perceptual finetune to True

PIXEL_LOSS = 'l1'
CONTENT_LOSS = 'l1'
ADV_LOSS = 'ragan'

LOSS_WEIGHTS = {
    'content_loss': 1.0,
    'adv_loss': 0.09,
}

# checkpoint ??

generator = Generator()
discriminator = Discriminator()

generator_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(BETA_1, BETA_2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA_1, BETA_2))

model = SRGAN(generator, discriminator, generator_optimizer, discriminator_optimizer, PERCEPTUAL_FINETUNE, PIXEL_LOSS, CONTENT_LOSS, ADV_LOSS, LOSS_WEIGHTS)

model.train()

temp_batch = next(iter(train_loader))
result = model.train_step(temp_batch)
print(result)


'''

for epoch in range(EPOCHS):
    for batch in train_loader:
        model.train_step(batch)
        
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}/{EPOCHS}')
        # save model checkpoint
        
'''