import torch
import torch.nn as nn

class GANLoss:
    def __init__(self):
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

    def generator_loss(self, fake_validity, fake_imgs, real_imgs, lambda_l1=100.0):
        adv_loss = self.adversarial_loss(fake_validity, torch.ones_like(fake_validity))
        l1_loss = self.l1_loss(fake_imgs, real_imgs)
        return adv_loss + lambda_l1 * l1_loss

    def discriminator_loss(self, real_validity, fake_validity):
        real_loss = self.adversarial_loss(real_validity, torch.ones_like(real_validity))
        fake_loss = self.adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
        return (real_loss + fake_loss) / 2
