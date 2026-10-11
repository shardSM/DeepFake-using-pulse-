"""PPGNet: 2D-CNN architecture for rPPG deepfake detection."""

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Construct a convolutional block: [Conv2d-BN-ReLU-Conv2d-BN-ReLU-MaxPool2d].

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        Sequential container of layers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class PPGNet(nn.Module):
    """PPGNet model processing (B, 3, 32, 128) PPG maps into window-level logits (B,)."""

    def __init__(self, dropout_rate: float = 0.3) -> None:
        """Initialize PPGNet architecture.

        Args:
            dropout_rate: Dropout probability in the classification head.
        """
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 32, 128).

        Returns:
            Logit tensor of shape (B,).
        """
        feats = self.features(x)
        logits = self.head(feats)
        return logits.squeeze(1)
