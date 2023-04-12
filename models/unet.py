import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, norm='batch', residual=True, activation='leakyrelu', transpose=False):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose

        if self.dropout:
            self.dropout1 = nn.Dropout2d(p=0.05)
            self.dropout2 = nn.Dropout2d(p=0.05)

        self.norm1 = None
        self.norm2 = None
        if norm == 'batch':
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'mixed':
            self.norm1 = nn.BatchNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        if self.transpose:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if self.activation == 'relu':
            self.actfun1 = nn.ReLU()
            self.actfun2 = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.actfun1 = nn.LeakyReLU()
            self.actfun2 = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.actfun1 = nn.ELU()
            self.actfun2 = nn.ELU()
        elif self.activation == 'selu':
            self.actfun1 = nn.SELU()
            self.actfun2 = nn.SELU()

    def forward(self, x):
        ox = x

        x = self.conv1(x)

        if self.dropout:
            x = self.dropout1(x)

        if self.norm1:
            x = self.norm1(x)

        x = self.actfun1(x)

        x = self.conv2(x)

        if self.dropout:
            x = self.dropout2(x)

        if self.norm2:
            x = self.norm2(x)

        if self.residual:
            x[:, 0:min(ox.shape[1], x.shape[1]), :, :] += ox[:, 0:min(ox.shape[1], x.shape[1]), :, :]

        x = self.actfun2(x)

        # print("shapes: x:%s ox:%s " % (x.shape,ox.shape))

        return x


class Unet(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, residual=False, activation='selu'):
        super(Unet, self).__init__()

        self.residual = residual

        self.down1 = nn.Conv2d(32, 32, kernel_size=2, stride=2, groups=32)
        self.down2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, groups=64)
        self.down3 = nn.Conv2d(128, 128, kernel_size=2, stride=2, groups=128)
        self.down4 = nn.Conv2d(256, 256, kernel_size=2, stride=2, groups=256)

        self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
        self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
        self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
        self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25

        self.down1.bias.data = 0.01 * self.down1.bias.data
        self.down2.bias.data = 0.01 * self.down2.bias.data
        self.down3.bias.data = 0.01 * self.down3.bias.data
        self.down4.bias.data = 0.01 * self.down4.bias.data

        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, groups=256)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, groups=128)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, groups=64)
        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, groups=32)

        self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
        self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
        self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
        self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25

        self.up1.bias.data = 0.01 * self.up1.bias.data
        self.up2.bias.data = 0.01 * self.up2.bias.data
        self.up3.bias.data = 0.01 * self.up3.bias.data
        self.up4.bias.data = 0.01 * self.up4.bias.data

        self.conv1 = ConvBlock(n_channel_in, 32, residual=residual, activation=activation)
        self.conv2 = ConvBlock(32, 64, residual=residual, activation=activation)
        self.conv3 = ConvBlock(64, 128, residual=residual, activation=activation)
        self.conv4 = ConvBlock(128, 256, residual=residual, activation=activation)

        self.conv5 = ConvBlock(256, 256, residual=residual, activation=activation)

        self.conv6 = ConvBlock(2 * 256, 128, residual=residual, activation=activation)
        self.conv7 = ConvBlock(2 * 128, 64, residual=residual, activation=activation)
        self.conv8 = ConvBlock(2 * 64, 32, residual=residual, activation=activation)
        self.conv9 = ConvBlock(2 * 32, n_channel_out, residual=residual, activation=activation)

        if self.residual:
            self.convres = ConvBlock(n_channel_in, n_channel_out, residual=residual, activation=activation)

    def forward(self, x):
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        x = torch.cat([x, c4], 1)  # x[:,0:128]*x[:,128:256],
        x = self.conv6(x)
        x = self.up2(x)
        x = torch.cat([x, c3], 1)  # x[:,0:64]*x[:,64:128],
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.cat([x, c2], 1)  # x[:,0:32]*x[:,32:64],
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.cat([x, c1], 1)  # x[:,0:16]*x[:,16:32],
        x = self.conv9(x)
        if self.residual:
            x = torch.add(x, self.convres(c0))

        return x
