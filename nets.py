import torch
from torch import nn as nn
from torch.nn import functional as F


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 5 - kernel size ... image dims -=4
        self.pool = nn.MaxPool2d(2, 2) # halves the image dims
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120) #32**2 = 1024
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x): # x: [4, 3, 64, 64]  - four images in batch [4 3 32 32]
        x = self.pool(F.relu(self.conv1(x))) # [4, 6, 30, 30] - 6 comes from 6 outputs in conv1 [4 6 14 14]
        x = self.pool(F.relu(self.conv2(x))) # [4, 16, 13, 13] - 16 comes from 16 outputs in conv2 [4 16 5 5]
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @classmethod
    def pth_fname(cls):
        return './cifar_net.pth'