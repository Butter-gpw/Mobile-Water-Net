import torch.nn as nn
import torch
from thop import profile
import time


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


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, act, scale_factor=2, kernel_size=3, stride=2, padding=1):
        super(DSConv, self).__init__()
        self.dsconv = nn.Sequential(
            ConBNAct(in_channels, in_channels * scale_factor, act, kernel_size=1),
            ConBNAct(in_channels * scale_factor, in_channels * scale_factor, act, kernel_size=kernel_size,
                     stride=stride, padding=padding, groups=in_channels * scale_factor),
            SEblock(in_channels * scale_factor, in_channels * scale_factor // 2),
            ConBNAct(in_channels * scale_factor, out_channels, act='L', kernel_size=1)
        )

    def forward(self, x):
        return self.dsconv(x)


class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, act, kernel_size=3, stride=2, padding=1):
        super(DeConv, self).__init__()
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConBNAct(in_channels, in_channels, act, kernel_size=kernel_size, stride=stride, padding=padding,
                     groups=in_channels),
            SEblock(in_channels, in_channels // 2),
            ConBNAct(in_channels, out_channels, act, kernel_size=1, stride=1)
        )


    def forward(self, x):
        return self.deconv(x)


class Mobile_Water_Net(nn.Module):
    def __init__(self):
        super(Mobile_Water_Net, self).__init__()
        self.stdconv = stdConv(3, 32, kernel_size=5, padding=2, stride=2)
        self.conv1 = DSConv(32, 128, scale_factor=4, act='R', kernel_size=3, stride=2, padding=1)
        self.conv2 = DSConv(128,256, act='R', kernel_size=3, stride=2, padding=1)
        self.conv3 = DSConv(256, 256, act='H', kernel_size=3, stride=2, padding=1)
        self.conv4 = DSConv(256, 256, act='H', kernel_size=3, stride=2, padding=1)

        self.deconv1 = DeConv(256, 256, act='H', kernel_size=3, stride=1, padding=1)
        self.deconv2 = DeConv(512, 256, act='H', kernel_size=3, stride=1, padding=1)
        self.deconv3 = DeConv(512, 128, act='H', kernel_size=3, stride=1, padding=1)
        self.deconv4 = DeConv(256, 32, act='R', kernel_size=3, stride=1, padding=1)
        self.deconv5 = DeConv(64, 3, act='R', kernel_size=3, stride=1, padding=1)
        self.deconv5.deconv[-1] = ConBNAct(64, 3, act='T', kernel_size=1)

    def forward(self, x):
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

    x = torch.randn((1, 3, 256, 256))

    model = Mobile_Water_Net()
    model = model.to(device)
    x = x.to(device)
    flops, params = profile(model, (x,))
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))

    torch.cuda.synchronize()
    start = time.time()
    result = model(x)
    torch.cuda.synchronize()
    end = time.time()
    print('infer_time:', end - start)
