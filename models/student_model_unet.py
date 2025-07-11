import torch
import torch.nn as nn

# Optional residual conv block (disabled by default)
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x) + self.skip(x)

# Standard double conv block with optional dropout
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, base_filters=64, use_dropout=False, depth=3):
        super().__init__()

        f = base_filters
        d = 0.2 if use_dropout else 0.0  # dropout probability

        # Encoder
        self.enc1 = ConvBlock(3, f, dropout=d)
        self.enc2 = ConvBlock(f, f*2, dropout=d)
        self.enc3 = ConvBlock(f*2, f*4, dropout=d) if depth >= 3 else None

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f*4 if depth >= 3 else f*2, f*8, dropout=d)

        # Decoder
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2) if depth >= 3 else None
        self.dec3 = ConvBlock(f*8, f*4, dropout=d) if depth >= 3 else None

        self.up2 = nn.ConvTranspose2d(f*4 if depth >= 3 else f*8, f*2, 2, stride=2)
        self.dec2 = ConvBlock(f*4, f*2, dropout=d)

        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.dec1 = ConvBlock(f*2, f, dropout=d)

        self.final = nn.Conv2d(f, 3, kernel_size=1)

    def forward(self, x):
        # Auto-pad input so H and W are divisible by 16 (for depth=4)
        _, _, h, w = x.size()
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        if self.enc3 is not None:
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        else:
            b = self.bottleneck(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))

        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.final(d1)

        # Remove padding to match original input size
        return out[:, :, :h, :w]

