
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.args = args
        num_groups = 8

        self.disc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_channels=128, num_groups=num_groups),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_channels=128, num_groups=num_groups),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_channels=256, num_groups=num_groups),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, x):
        logit = self.disc(x).squeeze()
        return logit





class DEnsemble(nn.Module):
    def __init__(self, Discriminator):
        super(DEnsemble, self).__init__()
        self.Discriminator = Discriminator
    def forward(self, x):
        D_out = self.Discriminator(x)
        return D_out
