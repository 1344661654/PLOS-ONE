import torch
import torch.nn as nn

class NAMAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(NAMAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, bottleneck_ratio=0.5):
        super(CSPBlock, self).__init__()
        hidden_channels = int(out_channels * bottleneck_ratio)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(hidden_channels)
        self.conv5 = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.namodule = NAMAttention(out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = nn.SiLU()(y)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.SiLU()(x)

        x = torch.cat((x, y), dim=1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.SiLU()(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.SiLU()(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = self.namodule(x)

        return x
