from src.dataloader import get_dataloader
from src.models.generator import Generator
from src.models.discriminator import Discriminator
import torch
# import torch.optim as optim
import torch.nn as nn
import yaml
import sys
import os
from torch.optim import Adam
from src.losses.gan_loss import GANLoss
from src.dataloader import get_dataloader
import wandb

def train():
        
    # Loading the configuration file -> Done successfully
        config_path = "../configs/config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    # print(config)

    # GPU Readiness line -> Done successfully
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

#     # Making GAn READY -> done successfully
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)

#     # Load dataloader -> ?
        dataloader = get_dataloader(config_path)

#     # Optimizers and loss setting -> done successfully
        optimizer_g = Adam(generator.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
        optimizer_d = Adam(discriminator.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
        # loss_fn = nn.MSELoss()
        loss_fn = GANLoss()
#     # Training loop -> done successfully
        num_epochs = config['train']['num_epochs']

        #epoch running -> done successfully
        for epoch in range(num_epochs):
            for i, (real_imgs, target_imgs) in enumerate(dataloader):
                real_imgs, target_imgs = real_imgs.to(device), target_imgs.to(device)

            # Generator and discriminator training logic
            # ...
            # Train Discriminator
            optimizer_d.zero_grad()
            real_validity = discriminator(target_imgs)
            fake_imgs = generator(real_imgs).detach()
            fake_validity = discriminator(fake_imgs)
            d_loss = loss_fn.discriminator_loss(real_validity, fake_validity)
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            fake_imgs = generator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            g_loss = loss_fn.generator_loss(fake_validity, fake_imgs, target_imgs)
            g_loss.backward()
            optimizer_g.step()

            # if i % 10 == 0:
                # wandb.log({"D_loss": d_loss.item(), "G_loss": g_loss.item()})
            print(f"Epoch {epoch},  Loss: ...")

if __name__ == '__main__':
    train()
