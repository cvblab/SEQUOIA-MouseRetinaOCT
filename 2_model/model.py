import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


seed = 42

# Python RNG
random.seed(seed)

# Numpy RNG
np.random.seed(seed)

# Torch RNG (CPU)
torch.manual_seed(seed)

# Torch RNG (GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class UNet(nn.Module):
    """
    U-Net architecture for image segmentation tasks.

    This model follows the typical U-shaped structure, consisting of a contracting path (encoder)
    to capture context and a symmetric expanding path (decoder) for precise localization.

    Attributes:
        enc1, enc2, enc3, enc4 (nn.Sequential): Convolutional blocks of the encoder.
        pool (nn.MaxPool2d): Max pooling layer used between encoder blocks.
        bottleneck (nn.Sequential): Middle part of the network with the highest number of filters.
        up1, up2, up3, up4 (nn.ConvTranspose2d): Transposed convolution layers for upsampling.
        dec1, dec2, dec3, dec4 (nn.Sequential): Convolutional blocks of the decoder.
        final (nn.Conv2d): Final 1x1 convolution layer producing the output segmentation map.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale images).
        out_channels (int): Number of output channels/classes for segmentation.
    """

    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            """
            Creates a double convolutional block with BatchNorm and ReLU activation.

            Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.

            Returns:
                nn.Sequential: Sequential block of two Conv2D layers with BatchNorm and ReLU.
            """
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder path
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder path
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where
                N is the batch size,
                C is the number of channels,
                H and W are the height and width of the input image.

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W), representing
            the segmentation map (raw logits).
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        # Output
        return self.final(dec1)
        # Optionally, apply softmax during inference:
        # return torch.softmax(self.final(dec1), dim=1)
