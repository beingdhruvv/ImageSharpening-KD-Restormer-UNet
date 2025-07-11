import torch
import torch.nn as nn
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True, layer='relu2_2'):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).features[:9].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.layer = layer
        self.resize = resize
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        if self.resize:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = nn.functional.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)

        x_feat = self.vgg(x)
        y_feat = self.vgg(y)
        return self.criterion(x_feat, y_feat)

