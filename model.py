import functools

import torch
import torch.nn as nn


def get_norm_layer(norm_layer_type):
    if norm_layer_type == 'batch':
        layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_layer_type == 'instance':
        layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_layer_type == 'none':
        layer = None
    else:
        raise NotImplementedError('[!] The norm_layer_type {] is not found.'.format(norm_layer_type))
    return layer


def get_non_linear_layer(non_linear_layer_type):
    if non_linear_layer_type == 'relu':
        layer = functools.partial(nn.ReLU, inplace=True)
    elif non_linear_layer_type == 'leaky_relu':
        layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif non_linear_layer_type == 'tanh':
        layer = functools.partial(nn.Tanh)
    elif non_linear_layer_type == 'none':
        layer = None
    else:
        raise NotImplementedError('[!] The non_linear_layer_type {] is not found.'.format(non_linear_layer_type))
    return layer


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ----------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k, s, p, norm_layer_type, non_linear_layer_type):
        super(ConvBlock, self).__init__()

        norm_layer = get_norm_layer(norm_layer_type)
        non_linear_layer = get_non_linear_layer(non_linear_layer_type)

        layers = []
        layers += [nn.Conv2d(in_dim, out_dim,  kernel_size=k, stride=s, padding=p)]
        if norm_layer is not None:
            layers += [norm_layer(out_dim, affine=True)]
        if non_linear_layer is not None:
            layers += [non_linear_layer(inplace=True)]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k, s, p, norm_layer_type, non_linear_layer_type):
        super(DeconvBlock, self).__init__()

        norm_layer = get_norm_layer(norm_layer_type)
        non_linear_layer = get_non_linear_layer(non_linear_layer_type)

        layers = []
        layers += [nn.ConvTranspose2d(in_dim, out_dim,  kernel_size=k, stride=s, padding=p)]
        if norm_layer is not None:
            layers += [norm_layer(out_dim, affine=True)]
        if non_linear_layer is not None:
            layers += [non_linear_layer()]

        self.deconv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_block(x)


def conv3x3(in_dim, out_dim):
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer_type='instance', non_linear_layer_type='leaky_relu'):
        super(ResBlock, self).__init__()

        norm_layer = get_norm_layer(norm_layer_type)
        non_linear_layer = get_non_linear_layer(non_linear_layer_type)

        self.conv = nn.Sequential(
            norm_layer(in_dim, affine=True),
            non_linear_layer(),
            conv3x3(in_dim, in_dim),
            norm_layer(in_dim, affine=True),
            non_linear_layer(),
            conv3x3(in_dim, out_dim),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# ----------------------------------------------------------------
# Discriminator
# ----------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf):
        super(Discriminator, self).__init__()

        # Discriminator with last patch (14x14)
        self.d_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=0, count_include_pad=False),
            ConvBlock(input_nc, ndf // 2, k=4, s=2, p=1, norm_layer_type='none', non_linear_layer_type='leaky_relu'),
            ConvBlock(ndf // 2, ndf, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu'),
            ConvBlock(ndf, ndf * 2, k=4, s=1, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu'),
            ConvBlock(ndf * 2, 1, k=4, s=1, p=1, norm_layer_type='none', non_linear_layer_type='none'),
        )

        # Discriminator with last patch (30x30)
        self.d_2 = nn.Sequential(
            ConvBlock(input_nc, ndf, k=4, s=2, p=1, norm_layer_type='none', non_linear_layer_type='leaky_relu'),
            ConvBlock(ndf, ndf * 2, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu'),
            ConvBlock(ndf * 2, ndf * 4, k=4, s=1, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu'),
            ConvBlock(ndf * 4, 1, k=4, s=1, p=1, norm_layer_type='none', non_linear_layer_type='none'),
        )

    def forward(self, x):
        """
        Args
            x: (half_size, input_nc(args.input_nc + args.output_nc), h, w)

        Return
            out_1: (half_size, 1, 15, 15)
            out_2: (half_size, 1, 30, 30)
        """

        # (half_size, input_nc, ndf(63), ndf(63)) ->
        # (half_size, ndf // 2, ndf // 2(31), ndf // 2(31)) ->
        # (half_size, ndf, ndf // 4(15), ndf // 4(15)) ->
        # (half_size, ndf, ndf // 4(14), ndf // 4(14)) ->
        # (half_size, 1, ndf // 4(13), ndf // 4(13))
        out_1 = self.d_1(x)

        # (half_size, ndf, ndf, ndf) ->
        # (half_size, ndf * 2, ndf // 2, ndf // 2) ->
        # (half_size, ndf * 4, ndf // 4 (31), ndf // 4(31)) ->
        # (half_size, 1, ndf // 4 (30), ndf // 4(30))
        out_2 = self.d_2(x)
        # print(out_2.size())

        return out_1, out_2


# ----------------------------------------------------------------
# Generator
# ----------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, nz=8):
        super(Generator, self).__init__()

        self.downsample_1 = ConvBlock(input_nc + nz, ngf, k=4, s=2, p=1, norm_layer_type='none', non_linear_layer_type='leaky_relu')
        self.downsample_2 = ConvBlock(ngf, ngf * 2, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu')
        self.downsample_3 = ConvBlock(ngf * 2, ngf * 4, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu')
        self.downsample_4 = ConvBlock(ngf * 4, ngf * 8, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu')
        self.downsample_5 = ConvBlock(ngf * 8, ngf * 8, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu')
        self.downsample_6 = ConvBlock(ngf * 8, ngf * 8, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu')
        self.downsample_7 = ConvBlock(ngf * 8, ngf * 8, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu')
        # self.downsample_8 = ConvBlock(ngf * 8, ngf * 8, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='leaky_relu')

        self.upsample_1 = DeconvBlock(ngf * 8, ngf * 8, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='relu')
        self.upsample_2 = DeconvBlock(ngf * 16, ngf * 8, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='relu')
        self.upsample_3 = DeconvBlock(ngf * 16, ngf * 8, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='relu')
        self.upsample_4 = DeconvBlock(ngf * 16, ngf * 4, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='relu')
        self.upsample_5 = DeconvBlock(ngf * 8, ngf * 2, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='relu')
        self.upsample_6 = DeconvBlock(ngf * 4, ngf, k=4, s=2, p=1, norm_layer_type='instance', non_linear_layer_type='relu')
        self.upsample_7 = DeconvBlock(ngf * 2, output_nc, k=4, s=2, p=1, norm_layer_type='none', non_linear_layer_type='tanh')

    def forward(self, x, z):
        """
        Args
            x: (half_size, input_nc, h, w)
            z: (half_size, nz)

        Return
            up_7: (half_size, input_nc, h, w)
        """

        z = z.view(z.size(0), z.size(1), 1, 1)  # (half_size, nz, 1, 1)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3))  # (half_size, nz, h, w)
        x_z = torch.cat([x, z], dim=1)  # (half_size, input_nc + nz, h, w)

        down_1 = self.downsample_1(x_z)  # (half_size, ngf, ngf, ngf)
        down_2 = self.downsample_2(down_1)  # (half_size, ngf * 2, ngf // 2, ngf // 2)
        down_3 = self.downsample_3(down_2)  # (half_size, ngf * 4, ngf // 4, ngf // 4)
        down_4 = self.downsample_4(down_3)  # (half_size, ngf * 8, ngf // 8, ngf // 8)
        down_5 = self.downsample_5(down_4)  # (half_size, ngf * 8, ngf // 16, ngf // 16)
        down_6 = self.downsample_6(down_5)  # (half_size, ngf * 8, ngf // 32, ngf // 32)
        down_7 = self.downsample_7(down_6)  # (half_size, ngf * 8, ngf // 64, ngf // 64)

        up_1 = self.upsample_1(down_7)  # (half_size, ngf * 8, ngf // 32, ngf // 32)
        up_2 = self.upsample_2(torch.cat([up_1, down_6], dim=1))  # (half_size, ngf * 8, ngf // 16, ngf // 16)
        up_3 = self.upsample_3(torch.cat([up_2, down_5], dim=1))  # (half_size, ngf * 8, ngf // 8, ngf // 8)
        up_4 = self.upsample_4(torch.cat([up_3, down_4], dim=1))  # (half_size, ngf * 4, ngf // 4, ngf // 4)
        up_5 = self.upsample_5(torch.cat([up_4, down_3], dim=1))  # (half_size, ngf * 2, ngf // 2, ngf // 2)
        up_6 = self.upsample_6(torch.cat([up_5, down_2], dim=1))  # (half_size, ngf * 1, ngf, ngf)
        up_7 = self.upsample_7(torch.cat([up_6, down_1], dim=1))  # (half_size, output_nc, ngf * 2, ngf * 2)

        return up_7


# ----------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_nc, nef, nz, non_linear_layer_type='leaky_relu'):
        super(Encoder, self).__init__()

        non_linear_layer = get_non_linear_layer(non_linear_layer_type)

        self.conv = nn.Conv2d(input_nc, nef, kernel_size=4, stride=2, padding=1)

        self.res_block = nn.Sequential(
            ResBlock(nef, nef * 2),
            ResBlock(nef * 2, nef * 3),
            ResBlock(nef * 3, nef * 4)
        )

        self.pool_block = nn.Sequential(
            non_linear_layer(),
            nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        )

        self.fc_mu = nn.Linear(nef * 4, nz)
        self.fc_logvar = nn.Linear(nef * 4, nz)

    def forward(self, x):
        """
        Args
            x: (half_size, input_nc, h, w)

        Return
            mu: (half_size, nz)
            logvar: (half_size, nz)
        """

        out = self.conv(x)  # (half_size, ndf, nef, nef)
        # (half_size, nef * 2, nef // 2, nef // 2) ->
        # (half_size, nef * 3, nef // 4, nef // 4) ->
        # (half_size, nef * 4, nef // 8, nef // 8)
        out = self.res_block(out)
        out = self.pool_block(out)  # (half_size, nef * 4, 1, 1)
        out = out.view(x.size(0), -1)  # (half_size, nef * 4)

        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        return mu, logvar
