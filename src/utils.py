import os
import torch
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Logging GAN losses
def log_losses(epoch, iteration, g_loss, d_loss):
    """Logs generator and discriminator losses."""
    print(f"Epoch [{epoch}] | Iteration [{iteration}] | G_Loss: {g_loss:.4f} | D_Loss: {d_loss:.4f}")

# Save model checkpoints
def save_checkpoint(model, optimizer, path, epoch):
    """Saves model and optimizer state to a checkpoint file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Checkpoint saved at {path}")

# Load model checkpoints
def load_checkpoint(path, model, optimizer=None):
    """Loads model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path}")
    return checkpoint['epoch']

# Save generated images during training or testing
def save_generated_images(images, path, prefix="generated"):
    """Saves a batch of generated images."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, f"{path}/{prefix}_{i}.png")
    print(f"Generated images saved at {path}")

# Calculate metrics: PSNR and SSIM
def calculate_metrics(real_images, fake_images):
    """Calculates PSNR and SSIM metrics for quality evaluation."""
    real_np = real_images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to NHWC
    fake_np = fake_images.cpu().detach().numpy().transpose(0, 2, 3, 1)
    
    psnr_vals, ssim_vals = [], []
    for real, fake in zip(real_np, fake_np):
        mse = np.mean((real - fake) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Using a smaller window size for SSIM
        ssim_val = ssim(real, fake, multichannel=True, data_range=1.0, win_size=3)
        
        psnr_vals.append(psnr)
        ssim_vals.append(ssim_val)

    return np.mean(psnr_vals), np.mean(ssim_vals)

# Display metrics
def display_metrics(epoch, psnr, ssim):
    """Logs evaluation metrics for an epoch."""
    print(f"Epoch [{epoch}] | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")
