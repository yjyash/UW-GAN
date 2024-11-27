from src.dataloader import get_dataloader
from src.models.generator import Generator
from src.models.discriminator import Discriminator
import torch
# import torch.optim as optim
import torch.nn as nn
import torchvision
import yaml
import sys
import os
from torch.optim import Adam
from src.losses.gan_loss import GANLoss
from src.dataloader import get_dataloader
import wandb
from utils import log_losses, save_checkpoint, save_generated_images, calculate_metrics, display_metrics

def save_checkpoint(generator, discriminator, epoch, optimizer_g, optimizer_d, checkpoint_dir="../checkpoints/"):
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the model weights and optimizer states
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
    }, os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))

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
        loss_fn = nn.MSELoss()
        loss_fn = GANLoss()
#     # Training loop -> done successfully
        num_epochs = config['train']['num_epochs']

        #epoch running -> done successfully
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            psnr_vals = []
            ssim_vals = []
            for i, (real_imgs, target_imgs) in enumerate(dataloader):
            
                real_imgs, target_imgs = real_imgs.to(device), target_imgs.to(device)
                # print(f"Batch {i} - Input Shape: {real_imgs.shape}, Target Shape: {target_imgs.shape}")
            
            # Train Discriminator
                optimizer_d.zero_grad()
                real_validity = discriminator(target_imgs)
                fake_imgs = generator(real_imgs).detach()

                # print(f"Fake Image Shape: {fake_imgs.shape}")

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

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

            # if i % 10 == 0:
                # wandb.log({"D_loss": d_loss.item(), "G_loss": g_loss.item()})
                torchvision.utils.save_image(fake_imgs, f"../outputs/epoch_{epoch}_iter_{i}.png")

                print(f"Epoch {epoch}, Iter {i}, G_Loss: {g_loss.item()}, D_Loss: {d_loss.item()}")
            psnr, ssim = calculate_metrics(target_imgs, fake_imgs)
            display_metrics(epoch, psnr, ssim)
            log_losses(epoch, i, epoch_g_loss / len(dataloader), epoch_d_loss / len(dataloader))

            save_checkpoint(generator, discriminator, epoch, optimizer_g, optimizer_d)

            # discriminator.save("discriminatorTrained.h5")
            # generator.save("generatorTrained.h5")


if __name__ == '__main__':
    train()