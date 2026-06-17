import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two-convolution residual block without normalization."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        # Prepare the residual path so it can be added to the main convolutional output.
        self.projection = (
            # If the channel size is already correct, keep the residual unchanged; otherwise adapt it with a 1x1 convolution.
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.layers(x) + self.projection(x))


class ResidualDownBlock(nn.Module):
    """Reduce spatial size and increase residual feature capacity."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualUpBlock(nn.Module):
    """Upsample decoder features and fuse them through a residual block."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.residual = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Skip connection
        x = torch.cat((skip, x), dim=1)
        return self.residual(x)


class ResidualUNet(nn.Module):
    """UNet variant that replaces convolutional blocks with residual blocks."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_features=32,
    ):
        super().__init__()
        features = base_features

        self.encoder1 = ResidualBlock(in_channels, features)
        self.encoder2 = ResidualDownBlock(features, features * 2)
        self.encoder3 = ResidualDownBlock(features * 2, features * 4)
        self.encoder4 = ResidualDownBlock(features * 4, features * 8)
        self.bottleneck = ResidualDownBlock(features * 8, features * 16)

        self.decoder4 = ResidualUpBlock(features * 16, features * 8, features * 8)
        self.decoder3 = ResidualUpBlock(features * 8, features * 4, features * 4)
        self.decoder2 = ResidualUpBlock(features * 4, features * 2, features * 2)
        self.decoder1 = ResidualUpBlock(features * 2, features, features)

        self.output = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        features = self.bottleneck(skip4)

        features = self.decoder4(features, skip4)
        features = self.decoder3(features, skip3)
        features = self.decoder2(features, skip2)
        features = self.decoder1(features, skip1)

        return torch.sigmoid(self.output(features))
