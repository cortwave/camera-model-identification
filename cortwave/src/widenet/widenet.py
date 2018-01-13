from torch import nn
import torch


class Wideblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Wideblock, self).__init__()
        out_channels = out_channels // 4
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.conv9 = nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=2, padding=4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x9 = self.conv9(x)
        x = torch.cat([x3, x5, x7, x9], 0)
        x = self.relu(x)
        return x


class Widenet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(Widenet, self).__init__()
        self.layer1 = Wideblock(in_channels, 128)
        self.layer2 = Wideblock(128, 64)
        self.layer3 = Wideblock(64, 32)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        print(x)
        return x
