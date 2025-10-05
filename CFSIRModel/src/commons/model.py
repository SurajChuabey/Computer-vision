import torch
import torch.nn as nn
import torch.nn.functional as F

class CFSIRBlock(nn.Module):
    def __init__(self, input_channels=3, mid_channels=16, output_channels=32, kernel_size=3, stride=1, padding=1):
        super(CFSIRBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(mid_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

class CFSIRModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CFSIRModel, self).__init__()

        # Block 1: 3 -> 16 -> 32
        self.block1 = CFSIRBlock(input_channels=3, mid_channels=16, output_channels=32)

        # Block 2: 32 -> 64 -> 128
        self.block2 = CFSIRBlock(input_channels=32, mid_channels=64, output_channels=128)

        # Additional conv layers: 128 -> 64 -> 32
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)      
        x = self.block2(x)       
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x