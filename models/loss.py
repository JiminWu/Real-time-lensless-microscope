import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def label_like(label: int, x: "Tensor") -> "Tensor":
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v

def soft_zeros_like(x: "Tensor") -> "Tensor":
    zeros = label_like(0, x)
    return torch.rand_like(zeros)


def soft_ones_like(x: "Tensor") -> "Tensor":
    ones = label_like(1, x)
    return ones * 0.7 + torch.rand_like(ones) * 0.5

class DLoss(nn.Module):
    def __init__(self, args):
        super(DLoss, self).__init__()
        self.args = args

    def forward(self, real_logit, fake_logit):
        self.real_loss = F.binary_cross_entropy_with_logits(
            real_logit, soft_ones_like(real_logit)
        )
        self.fake_loss = F.binary_cross_entropy_with_logits(
            fake_logit, soft_zeros_like(fake_logit)
        )

        self.total_loss = (self.real_loss + self.fake_loss) / 2.0

        return self.total_loss

class GLoss(nn.Module):
    def __init__(self, args):
        super(GLoss, self).__init__()
        self.args = args

    def forward(self, output: "Tensor", target: "Tensor", fake_logit: "Tensor"):
        device = output.device

        self.total_loss = torch.tensor(0.0).to(device)
        self.adversarial_loss = torch.tensor(0.0).to(device)
        self.L1_loss = torch.tensor(0.0).to(device)
        self.SSIM_loss = torch.tensor(0.0).to(device)
        
        if self.args.lambda_L1:
            l1_loss = torch.nn.L1Loss()
            self.L1_loss += (l1_loss(output, target) * self.args.lambda_L1)
            
        if self.args.lambda_SSIM:
            self.SSIM_loss += ((1 - ms_ssim(output, target)) * self.args.lambda_SSIM)

        if self.args.lambda_adversarial:
            self.adversarial_loss += (
                F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
                * self.args.lambda_adversarial
            )

        self.total_loss += (
            self.adversarial_loss
            + self.L1_loss
            + self.SSIM_loss
        )

        return self.total_loss
