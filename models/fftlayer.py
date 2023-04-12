import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class fft_conv (nn.Module):
    
    def __init__(self, psfs, gamma):
        super(fft_conv, self).__init__()
        psfs = torch.tensor(psfs, dtype=torch.float32)
        gamma = torch.tensor(gamma, dtype=torch.float32)
        
        self.psfs = nn.Parameter(psfs, requires_grad =True)
        self.gamma = nn.Parameter(gamma, requires_grad =True)
        
    def forward(self, img, padding_fft_x=0, padding_fft_y=0):
        
        a,b,h,w = img.shape
        img = img.type(torch.complex64)
        
        if padding_fft_x == 0 & padding_fft_y == 0:
            img_pad = img
            psfs_pad = self.psfs
            center_x = h // 2
            center_y = w // 2
     
        else:
            padding = (padding_fft_x, padding_fft_x, padding_fft_y, padding_fft_y)
            center_x = (h + 2 * padding_fft_x) // 2
            center_y = (w + 2 * padding_fft_y) // 2
        
            img_pad = F.pad(img, padding)
            psfs_pad = F.pad(self.psfs, padding)
            
        Y = torch.fft.fft2(img_pad)
        W = torch.fft.fft2(psfs_pad)

        X = (torch.conj(W)*Y)/ (torch.square(torch.abs(W))+100*self.gamma)
    
        img_pad = torch.real((torch.fft.ifftshift(torch.fft.ifft2(X), dim = (-2, -1))))
        img = img_pad[:, :, center_x-h//2:center_x+h//2, center_y-w//2:center_y+w//2]

        return img
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'psfs': self.psfs.numpy(),
            'gamma': self.gamma.numpy()
        })
        return config
    
class MyEnsemble(nn.Module):
    def __init__(self, fftlayer, unet_model):
        super(MyEnsemble, self).__init__()
        self.fftlayer = fftlayer
        self.unet_model = unet_model
    def forward(self, x):
        fft_out = self.fftlayer(x)
        final_output = self.unet_model(fft_out)
        return final_output

