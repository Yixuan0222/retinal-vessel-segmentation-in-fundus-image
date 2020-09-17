import torch
import torch.nn as nn
import torch.nn.functional as F




class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[4, 4], stride=4)
        self.pool3 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        self.pool4 = nn.MaxPool2d(kernel_size=[12, 12], stride=12)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, padding=0)


    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv1(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv2(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv3(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv4(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


