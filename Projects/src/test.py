import torch
from src.models.generator import Generator
from torchvision.utils import save_image
from src.dataloader import get_test_loader

# Load the generator
generator = Generator().to("cuda")
generator.load_state_dict(torch.load("outputs/generator.pth"))

# Load test data
test_loader = get_test_loader("configs/config.yaml")

# Inference
for i, real_imgs in enumerate(test_loader):
    real_imgs = real_imgs.to("cuda")
    enhanced_imgs = generator(real_imgs)
    save_image(enhanced_imgs, f"outputs/enhanced_{i}.png")
