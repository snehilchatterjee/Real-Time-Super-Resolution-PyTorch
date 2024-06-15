import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data import *
from models import *
#from callbacks import *
from losses import *

# Define the dataset and data transformations
class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self,path , scale=4, transform=None):
        self.dataset = os.listdir(path)
        #self.dataset = tfds.load(f'div2k/bicubic_x{scale}', split='train', shuffle_files=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        lrs = sample['lr']
        hrs = sample['hr']
        if self.transform:
            lrs = self.transform(lrs)
            hrs = self.transform(hrs)
        return lrs, hrs

# Define data transformations
'''
transform = transforms.Compose([
    transforms.Lambda(random_compression),
    transforms.Lambda(random_crop),
    transforms.Lambda(random_spatial_augmentation),
    transforms.ToTensor()
])
'''

train_dataset = DIV2KDataset(scale=SCALE, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Sample data inspection
for lrs, hrs in train_loader:
    break

print(lrs.shape, hrs.shape)
print(lrs.dtype, hrs.dtype)
print(torch.min(lrs), torch.max(lrs))
print(torch.min(hrs), torch.max(hrs))

visualize_samples(images_lists=(lrs[:15], hrs[:15]), titles=('Low Resolution', 'High Resolution'), size=(8, 8))

# Define the SRGAN model
class SRGAN(nn.Module, PixelLossTraining, GramStyleTraining, VGGContentTraining, AdversarialTraining):
    def __init__(self, generator, discriminator):
        super(SRGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, generator_optimizer, discriminator_optimizer, perceptual_finetune, pixel_loss, style_loss, content_loss, adv_loss, loss_weights):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.perceptual_finetune = perceptual_finetune
        self.setup_pixel_loss(pixel_loss)
        self.setup_content_loss(content_loss)
        self.setup_adversarial_loss(adv_loss)
        if self.perceptual_finetune:
            self.loss_weights = loss_weights

    def train_step(self, batch):
        lrs, hrs = batch
        if self.perceptual_finetune:
            self.generator.train()
            self.discriminator.train()

            # Training Discriminator
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            srs = self.generator(lrs)

            real_logits = self.discriminator(hrs)
            fake_logits = self.discriminator(srs)

            content_loss = self.loss_weights['content_loss'] * self.content_loss(srs, hrs)
            gen_adv_loss = self.loss_weights['adv_loss'] * self.gen_adv_loss(fake_logits, real_logits)
            perceptual_loss = content_loss + gen_adv_loss

            gen_loss = perceptual_loss
            disc_adv_loss = self.disc_adv_loss(fake_logits, real_logits)

            disc_adv_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            gen_loss.backward()
            self.generator_optimizer.step()

            return {
                'Perceptual Loss': perceptual_loss.item(),
                'Generator Adv Loss': gen_adv_loss.item(),
                'Discriminator Adv Loss': disc_adv_loss.item(),
            }

        else:
            self.generator_optimizer.zero_grad()
            srs = self.generator(lrs)
            pixel_loss = self.pixel_loss(srs, hrs)
            pixel_loss.backward()
            self.generator_optimizer.step()

            return {
                'Pixel Loss': pixel_loss.item(),
            }

EPOCHS = 1000
LR = 0.002
BETA_1 = 0.9
BETA_2 = 0.999

PERCEPTUAL_FINETUNE = False

PIXEL_LOSS = nn.L1Loss()
STYLE_LOSS = nn.L1Loss()
CONTENT_LOSS = nn.L1Loss()
ADV_LOSS = 'ragan'

LOSS_WEIGHTS = {'content_loss': 1.0, 'adv_loss': 0.09, 'style_loss': 1.0}

CHECKPOINT_DIR = os.path.join('drive', 'MyDrive', 'Model-Checkpoints', 'Super Resolution')

generator_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(BETA_1, BETA_2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA_1, BETA_2))

generator = Generator()
print(generator)

discriminator = Discriminator()
print(discriminator)

srgan = SRGAN(generator, discriminator)
srgan.compile(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    perceptual_finetune=PERCEPTUAL_FINETUNE,
    pixel_loss=PIXEL_LOSS,
    style_loss=STYLE_LOSS,
    content_loss=CONTENT_LOSS,
    adv_loss=ADV_LOSS,
    loss_weights=LOSS_WEIGHTS
)

ckpt_callback = CheckpointCallback(
    checkpoint_dir=CHECKPOINT_DIR,
    resume=True,
    epoch_step=4
)
ckpt_callback.set_model(srgan)
ckpt_callback.setup_checkpoint(srgan)
ckpt_callback.set_lr(LR, BETA_1)

# Training loop
for epoch in range(EPOCHS):
    for batch in train_loader:
        loss_dict = srgan.train_step(batch)
        print(f"Epoch {epoch}/{EPOCHS}, {loss_dict}")
    
    if epoch % 10 == 0:
        ckpt_callback.save_checkpoint(epoch)

    # Callbacks
    for callback in [
        ProgressCallback(logs_step=0.2, generator_step=2)
    ]:
        callback.on_epoch_end(epoch, srgan)

