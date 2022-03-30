import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_block=False):
        super(DownBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, in_ch, 3, 1, 1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True)) if first_block else nn.MaxPool2d(2),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.pool = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        x_t = self.pool(x)
        x_in = torch.cat((x_t, y), dim=1)
        out = self.model(x_in)
        return out

class FBPConvNet(nn.Module):
    def __init__(self):
        super(FBPConvNet, self).__init__()
        self.conv1 = DownBlock(64, 64, True)
        self.conv2 = DownBlock(64, 128)
        self.conv3 = DownBlock(128, 256)
        self.conv4 = DownBlock(256, 512)
        self.conv5 = DownBlock(512, 1024)
        self.conv4_t = UpBlock(1024, 512)
        self.conv3_t = UpBlock(512, 256)
        self.conv2_t = UpBlock(256, 128)
        self.conv1_t = UpBlock(128, 64)
        self.conv_last = nn.Conv2d(64, 1, 1, 1, 0)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        # encoder
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)
        x_5 = self.conv5(x_4)

        # decoder
        y_4 = self.conv4_t(x_5, x_4)
        y_3 = self.conv3_t(y_4, x_3)
        y_2 = self.conv2_t(y_3, x_2)
        y_1 = self.conv1_t(y_2, x_1)
        y = self.conv_last(y_1)
        out = y + x
        return out
