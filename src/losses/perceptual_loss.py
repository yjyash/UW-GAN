import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, layers=None, use_gpu=True):
        """
        Initialize the perceptual loss module.

        Args:
            layers (list): VGG layers to use for feature comparison. Default: ['conv1_2', 'conv2_2', 'conv3_3']
            use_gpu (bool): If True, use GPU for feature extraction.
        """
        super(PerceptualLoss, self).__init__()
        self.use_gpu = use_gpu
        self.vgg = models.vgg19(pretrained=True).features
        self.layers = layers or [4, 9, 18]  # Corresponds to 'conv1_2', 'conv2_2', 'conv3_3' in VGG
        self.loss_fn = nn.MSELoss()

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        if self.use_gpu:
            self.vgg = self.vgg.cuda()

    def forward(self, pred, target):
        """
        Compute the perceptual loss between the predicted and target images.

        Args:
            pred (torch.Tensor): Predicted images.
            target (torch.Tensor): Target (ground truth) images.

        Returns:
            torch.Tensor: Calculated perceptual loss.
        """
        loss = 0.0
        pred_features = pred
        target_features = target

        for i, layer in enumerate(self.vgg):
            pred_features = layer(pred_features)
            target_features = layer(target_features)

            if i in self.layers:
                loss += self.loss_fn(pred_features, target_features)

        return loss
