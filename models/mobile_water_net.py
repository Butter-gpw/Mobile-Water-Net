import torch.nn as nn
import torch


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


class ConBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, groups=1, bias=False):
        super(ConBNReLU, self).__init__()
        self.ConBNRe = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=bias),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.ConBNRe(x)
        return x


class ConBNHswish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, groups=1, bias=False):
        super(ConBNHswish, self).__init__()
        self.ConBNHs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=bias),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.Hardswish()
        )

    def forward(self, x):
        x = self.ConBNHs(x)
        return x


class ConBNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False):
        super(ConBNLinear, self).__init__()
        self.ConBNLin = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=bias),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.Identity()
        )

    def forward(self, x):
        x = self.ConBNLin(x)
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


class Mobile_Water_Net(nn.Module):
    def __init__(self):
        super(Mobile_Water_Net, self).__init__()
        self.stdconv = stdConv(3, 32, kernel_size=5, padding=2, stride=2)
        self.conv1 = nn.Sequential(
            ConBNReLU(32, 128, kernel_size=1, stride=1),
            ConBNReLU(128, 128, kernel_size=3, stride=2, groups=128, padding=1),
            SEblock(128, 64),
            ConBNLinear(128, 128, kernel_size=1, stride=1)
        )
        self.conv2 = nn.Sequential(
            ConBNReLU(128, 256, kernel_size=1, stride=1),
            ConBNReLU(256, 256,  kernel_size=3, stride=2, groups=256, padding=1),
            SEblock(256, 128),
            ConBNLinear(256, 256, kernel_size=1, stride=1)
        )

        self.conv3 = nn.Sequential(
            ConBNHswish(256, 512, kernel_size=1, stride=1),
            ConBNHswish(512, 512, kernel_size=3, stride=2, groups=512, padding=1),
            SEblock(512, 256),
            ConBNLinear(512, 256, kernel_size=1, stride=1)
        )

        self.conv4 = nn.Sequential(
            ConBNHswish(256, 512, kernel_size=1, stride=1),
            ConBNHswish(512, 512, kernel_size=3, stride=2, groups=512, padding=1),
            SEblock(512, 256),
            ConBNLinear(512, 256, kernel_size=1, stride=1)
        )

        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConBNHswish(256, 256, kernel_size=3, stride=1, padding=1, groups=256),
            SEblock(256, 128),
            ConBNHswish(256, 256, kernel_size=1, stride=1)
        )

        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConBNHswish(512, 512, kernel_size=3, stride=1, padding=1, groups=512),
            SEblock(512, 256),
            ConBNHswish(512, 256, kernel_size=1, stride=1)
        )

        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConBNHswish(512, 512, kernel_size=3, stride=1, padding=1, groups=512),
            SEblock(512, 256),
            ConBNHswish(512, 128, kernel_size=1, stride=1)
        )

        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConBNHswish(256, 256, kernel_size=3, stride=1, padding=1, groups=256),
            SEblock(256, 128),
            ConBNHswish(256, 32, kernel_size=1, stride=1)
        )

        self.deconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConBNReLU(64, 256, kernel_size=1, stride=1, groups=16),
            SEblock(256, 128),
            ConBNLinear(256, 3, kernel_size=1, stride=1),
            nn.Tanh()
        )

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
    x = torch.randn((1, 3, 256, 256))

    model = Mobile_Water_Net()
    num_params = sum(p.numel() for p in model.parameters())
    # print(model)
    print(f'网络参数量：{num_params}')
    model(x)
