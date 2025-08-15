import torch.nn as nn

class CNNClassifier(nn.Module):
    """
    A simple Convolutional Neural Network for MFCC-based audio classification.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Input shape: (batch_size, 1, 13, 51) where 1=channel, 13=n_mfcc, 51=time_frames
        self.conv_stack = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # After L1: (batch_size, 16, 6, 25)

            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # After L2: (batch_size, 32, 3, 12)

            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # After L3: (batch_size, 64, 1, 6)
        )

        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=64 * 1 * 7, out_features=128), # Adjusted for new MFCC shape (448 features)
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout for regularization
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits