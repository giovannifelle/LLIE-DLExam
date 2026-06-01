import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two convolution layers used in each UNet stage."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DownBlock(nn.Module):
    """Reduce spatial size and increase the number of features."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    """Upsample and combine decoder features with the encoder skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.convolution = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat((skip, x), dim=1)
        return self.convolution(x)


class UNet(nn.Module):
    """Compact implementation of the classic Ronneberger UNet."""

    def __init__(self, in_channels=3, out_channels=3, base_features=32):
        super().__init__()
        features = base_features

        self.encoder1 = DoubleConv(in_channels, features)
        self.encoder2 = DownBlock(features, features * 2)
        self.encoder3 = DownBlock(features * 2, features * 4)
        self.encoder4 = DownBlock(features * 4, features * 8)
        self.bottleneck = DownBlock(features * 8, features * 16)

        self.decoder4 = UpBlock(features * 16, features * 8, features * 8)
        self.decoder3 = UpBlock(features * 8, features * 4, features * 4)
        self.decoder2 = UpBlock(features * 4, features * 2, features * 2)
        self.decoder1 = UpBlock(features * 2, features, features)

        self.output = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        x = self.bottleneck(skip4)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        # Pixel values are kept in the same range used by the dataset pipeline.
        return torch.sigmoid(self.output(x))
