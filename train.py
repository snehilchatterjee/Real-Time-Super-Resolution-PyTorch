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

class SRGAN(nn.Module,
            PixelLossTraining,
            VGGContentTraining,
            AdversarialTraining):
    
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
        
        super(SRGAN, self).__init__(self)
        
        self.generator = generator
        self.discriminator = discriminator
        
        self.generator.optimizer = generator_optimizer
        self.discriminator.optimizer = discriminator_optimizer
        
        self.perceptual_finetune = perceptual_finetune

        self.setup_pixel_loss(pixel_loss)
        # self.setup_gram_style_loss(style_loss)
        # uncomment this to utilize style loss function
        self.setup_content_loss(content_loss)
        self.setup_adversarial_loss(adv_loss)

        if self.perceptual_finetune:
            self.loss_weights = loss_weights
            
            
    import torch

def train_step(self, batch):
    self.lrs = batch[0].to(self.device)  # Assuming batch[0] is the low-resolution images
    self.hrs = batch[1].to(self.device)  # Assuming batch[1] is the high-resolution images

    if self.perceptual_finetune:
        # [=================== Training Discriminator ===================]

        self.generator.train()
        self.discriminator.train()

        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        self.srs = self.generator(self.lrs)

        real_logits = self.discriminator(self.hrs)
        fake_logits = self.discriminator(self.srs.detach())

        content_loss = self.loss_weights['content_loss'] * self.content_loss(self.srs, self.hrs)
        gen_adv_loss = self.loss_weights['adv_loss'] * self.gen_adv_loss(fake_logits, real_logits)
        perceptual_loss = content_loss + gen_adv_loss
        
        # style_loss = self.loss_weights['style_loss'] * self.gram_style_loss(self.srs, self.hrs)
        # Uncomment and add to gen_loss to utilize style loss function

        gen_loss = perceptual_loss
        disc_adv_loss = self.disc_adv_loss(fake_logits, real_logits)

        disc_adv_loss.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        gen_loss.backward()
        self.optimizer_G.step()

        return {
            'Perceptual Loss': perceptual_loss.item(),
            # 'Style Loss': style_loss.item(),
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

    '''
            
    def train_step(self, batch):
        self.lrs = batch[0]
        self.hrs = batch[1]

        if self.perceptual_finetune:
            # [=================== Training Discriminator ===================]

            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                self.srs = self.generator(self.lrs, training = True)

                real_logits = self.discriminator(self.hrs, training = True)
                fake_logits = self.discriminator(self.srs, training = True)

                content_loss = self.loss_weights['content_loss'] * self.content_loss(self.srs, self.hrs)
                gen_adv_loss = self.loss_weights['adv_loss'] * self.gen_adv_loss(fake_logits, real_logits)
                perceptual_loss = content_loss + gen_adv_loss
                
                # style_loss = self.loss_weights['style_loss'] * self.gram_style_loss(self.srs, self.hrs)
                # uncomment this and add it to gen_loss to utilize style loss function

                gen_loss = perceptual_loss

                disc_adv_loss = self.disc_adv_loss(fake_logits, real_logits)
            
            discriminator_gradients = disc_tape.gradient(disc_adv_loss, self.discriminator.trainable_variables)
            generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            
            self.discriminator.optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
            self.generator.optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

            return {
                'Perceptual Loss': perceptual_loss,
                # 'Style Loss': style_loss,
                'Generator Adv Loss': gen_adv_loss,
                'Discriminator Adv Loss': disc_adv_loss,
            }
        
        else:
            with tf.GradientTape() as gen_tape:
                self.srs = self.generator(self.lrs, training = True)

                pixel_loss = self.pixel_loss(self.srs, self.hrs)

            generator_gradients = gen_tape.gradient(pixel_loss, self.generator.trainable_variables)
            self.generator.optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

            return {
                'Pixel Loss': pixel_loss,
            }

    '''
    