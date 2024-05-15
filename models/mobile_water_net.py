import torch.nn as nn
import torch
from thop import profile
import time


def build_conv_block(in_chans, out_chans, kernel_size=3, stride=2, padding=1, use_bn=True, bn_momentum=0.8,
                     use_leaky=False):
    layers = []
    layers.append(nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding))
    if use_leaky:
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    else:
        layers.append(nn.ReLU(inplace=True))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_chans, momentum=bn_momentum))
    return nn.Sequential(*layers)


def build_deconv_block(in_chans, out_chans, use_bn=True):
    layers = []
    layers.append(nn.Upsample(scale_factor=2,
                              mode="bilinear", align_corners=True))
    layers.append(nn.Conv2d(in_chans, out_chans, 3, 1, 1))
    layers.append(nn.ReLU(inplace=True))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_chans, momentum=0.8))
    return nn.Sequential(*layers)


class stdConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False):
        super(stdConv, self).__init__()
        self.stdConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.Hardswish()
        )

    def forward(self, x):
        x = self.stdConv(x)
        return x


class SEblock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(SEblock, self).__init__()
        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        x = self.SE(x) * x
        return x


class PatchChannelAttention(nn.Module):
    def __init__(self, in_channels, scale_factor=4, patch_ratio=2):
        super(PatchChannelAttention, self).__init__()
        self.patch_ratio = patch_ratio
        self.SE = SEblock(in_channels * patch_ratio * patch_ratio, in_channels // scale_factor)

    def forward(self, x):
        patches = x.reshape(x.size(0), -1, x.size(2) // self.patch_ratio, x.size(3) // self.patch_ratio).contiguous()
        patches = self.SE(patches)
        patches = patches.reshape(x.size(0), x.size(1), x.size(2), x.size(3)).contiguous()
        #return patches + x
        return patches


class ConBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, act='Linear', kernel_size=1, stride=1, padding=0, groups=1,
                 bias=False):
        super(ConBNAct, self).__init__()
        self.ConBNAct = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=bias),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        )
        if act == 'Relu' or act == 'R':
            self.ConBNAct.add_module('Relu', nn.ReLU(inplace=True))
        elif act == 'Linear' or act == 'L':
            self.ConBNAct.add_module('Linear', nn.Identity())
        elif act == 'Hardswish' or act == 'H':
            self.ConBNAct.add_module("Hardswish", nn.Hardswish())
        elif act == 'Tanh' or act == 'T':
            self.ConBNAct.add_module('Tanh', nn.Tanh())

    def forward(self, x):
        x = self.ConBNAct(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = SEblock(in_channels, in_channels//reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, act, scale_factor=2, kernel_size=3, stride=2, padding=1):
        super(DSConv, self).__init__()
        self.dsconv = nn.Sequential(
            ConBNAct(in_channels, in_channels * scale_factor, act, kernel_size=1),
            ConBNAct(in_channels * scale_factor, in_channels * scale_factor, act, kernel_size=kernel_size,
                     stride=stride, padding=padding, groups=in_channels * scale_factor),
            SEblock(in_channels * scale_factor, in_channels * scale_factor // 2),
            ConBNAct(in_channels * scale_factor, out_channels, act='L', kernel_size=1),
        )

    def forward(self, x):
        return self.dsconv(x)


class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, act, kernel_size=3, stride=2, padding=1, up_dim=1):
        super(DeConv, self).__init__()
        self._layer = []
        self._layer.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        if up_dim > 1:
            self._layer.append(ConBNAct(in_channels, in_channels * up_dim, act, kernel_size=1))

        self._layer.append(
            ConBNAct(in_channels * up_dim, in_channels * up_dim, act, kernel_size=kernel_size, stride=stride,
                     padding=padding,
                     groups=in_channels * up_dim))
        self._layer.append(SEblock(in_channels * up_dim, in_channels * up_dim // 2))
        self._layer.append(ConBNAct(in_channels * up_dim, out_channels, act, kernel_size=1, stride=1))

        self.deconv = nn.Sequential(*self._layer)

    def forward(self, x):
        return self.deconv(x)



#原模型
class Mobile_Water_Net(nn.Module):
    def __init__(self, n_feats=32):
        super(Mobile_Water_Net, self).__init__()

        self.stdconv = stdConv(3, n_feats, kernel_size=5, padding=2, stride=2)
        self.conv1 = DSConv(n_feats, n_feats*4, scale_factor=4, act='R', kernel_size=3, stride=2, padding=1)
        self.conv2 = DSConv(n_feats*4,  n_feats*8, act='R', kernel_size=3, stride=2, padding=1)
        self.conv3 = DSConv(n_feats*8, n_feats*8, act='H', kernel_size=3, stride=2, padding=1)
        self.conv4 = DSConv(n_feats*8, n_feats*8, act='H', kernel_size=3, stride=2, padding=1)

        # self.conv1 = build_conv_block(
        #     3, n_feats, 5, padding=2, use_bn=False)
        # self.conv2 = build_conv_block(n_feats, n_feats*4, 4)
        # self.conv3 = build_conv_block(n_feats*4, n_feats*8, 4)
        # self.conv4 = build_conv_block(n_feats*8, n_feats*8)
        # self.conv5 = build_conv_block(n_feats*8, n_feats*8)

        self.deconv1 = DeConv(n_feats*8, n_feats*8, act='H', kernel_size=3, stride=1, padding=1)
        self.deconv2 = DeConv(n_feats*16, n_feats*8, act='H', kernel_size=3, stride=1, padding=1)
        self.deconv3 = DeConv(n_feats*16, n_feats*4, act='H', kernel_size=3, stride=1, padding=1)
        self.deconv4 = DeConv(n_feats*8, n_feats, act='R', kernel_size=3, stride=1, padding=1)
        self.deconv5 = DeConv(n_feats*2, 3, act='R', kernel_size=3, stride=1, padding=1, up_dim=2)
        self.deconv5.deconv[-1] = ConBNAct(n_feats*4, 3, act='T', kernel_size=1)

        # self.deconv1 = build_deconv_block(n_feats * 8, n_feats * 8)
        # self.deconv2 = build_deconv_block(n_feats * 16, n_feats * 8)
        # self.deconv3 = build_deconv_block(n_feats * 16, n_feats * 4)
        # self.deconv4 = build_deconv_block(n_feats * 8, n_feats * 1)
        # self.deconv5 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        #                              nn.Conv2d(n_feats * 2, 3, 3, 1, 1),
        #                              nn.Tanh())

    def forward(self, x):
        # FUnIE downsample
        # d1 = self.conv1(x)   # (B, 32, 128, 128)
        # d2 = self.conv2(d1)  # (B, 128, 64, 64)
        # d3 = self.conv3(d2)  # (B, 256, 32, 32)
        # d4 = self.conv4(d3)  # (B, 256, 16, 16)
        # d5 = self.conv5(d4)  # (B, 256, 8, 8)

        # our downsample
        d1 = self.stdconv(x)  # (B, 32, 128, 128)
        d2 = self.conv1(d1)  # (B, 128, 64, 64)
        d3 = self.conv2(d2)  # (B, 256, 32, 32)
        d4 = self.conv3(d3)  # (B, 256, 16, 16)
        d5 = self.conv4(d4)  # (B, 256, 8, 8)



        u1 = torch.cat([self.deconv1(d5), d4], dim=1)  # (B, 512, 16, 16)
        u2 = torch.cat([self.deconv2(u1), d3], dim=1)  # (B, 512, 32, 32)
        u3 = torch.cat([self.deconv3(u2), d2], dim=1)  # (B, 256, 64, 64)
        u4 = torch.cat([self.deconv4(u3), d1], dim=1)  # (B, 64, 128, 128)
        u5 = self.deconv5(u4)

        return u5





if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #x = torch.randn((1, 256, 8, 8))
    x = torch.randn((1, 3, 256, 256))

    model = Mobile_Water_Net()
    model.eval()
    model = model.to(device)
    x = x.to(device)
    flops, params = profile(model, (x,))
    print('flops: %.4f G, params: %.4f M' % (flops / 1e9, params / 1e6))

    torch.cuda.synchronize()
    start = time.time()
    result = model(x)
    torch.cuda.synchronize()
    end = time.time()
    print('infer_time:', end - start)
