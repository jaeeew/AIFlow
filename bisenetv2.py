import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class BiSeNetV2(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.layer1 = ConvBNReLU(3, 16, 3, 2, 1)   # 256 -> 128
        self.layer2 = ConvBNReLU(16, 32, 3, 2, 1)  # 128 -> 64
        self.layer3 = ConvBNReLU(32, 64, 3, 2, 1)  # 64 -> 32
        self.layer4 = ConvBNReLU(64, 128, 3, 2, 1) # 32 -> 16
        self.layer5 = ConvBNReLU(128, 128, 3, 1, 1)

        # Channel matching for skip connections
        self.match3 = nn.Conv2d(128, 64, kernel_size=1)
        self.match2 = nn.Conv2d(64, 32, kernel_size=1)
        self.match1 = nn.Conv2d(32, 16, kernel_size=1)

        # Upsampling layers
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.final_conv = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        x1 = self.layer1(x)    # 256 -> 128
        x2 = self.layer2(x1)   # 128 -> 64
        x3 = self.layer3(x2)   # 64 -> 32
        x4 = self.layer4(x3)   # 32 -> 16
        x5 = self.layer5(x4)   # 16

        x = self.upsample4(x5)
        x = self.match3(x) + x3

        x = self.upsample3(x)
        x = self.match2(x) + x2

        x = self.upsample2(x)
        x = self.match1(x) + x1

        x = self.upsample1(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)
