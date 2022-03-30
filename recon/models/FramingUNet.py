import torch
import torch.nn as nn

class Harr_wav(nn.Module):
    def __init__(self):
        super(Harr_wav, self).__init__()
        filter = [[[ 0.5,  0.5], [ 0.5, 0.5]], 
                  [[-0.5,  0.5], [-0.5, 0.5]], 
                  [[-0.5, -0.5], [ 0.5, 0.5]], 
                  [[ 0.5, -0.5], [-0.5, 0.5]]]
        weight = torch.tensor(filter).view(4, 1, 1, 2, 2)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return nn.functional.conv3d(x.unsqueeze(1), self.weight, stride=(1,2,2))

class Harr_iwav_cat(nn.Module):
    def __init__(self):
        super(Harr_iwav_cat, self).__init__()
        filter_LL = [[ 0.5,  0.5], [ 0.5, 0.5]]
        filter_LH = [[-0.5,  0.5], [-0.5, 0.5]]
        filter_HL = [[-0.5, -0.5], [ 0.5, 0.5]]
        filter_HH=  [[ 0.5, -0.5], [-0.5, 0.5]]
        weight_LL = torch.tensor(filter_LL).view(1,1,1,2,2)
        weight_LH = torch.tensor(filter_LH).view(1,1,1,2,2)
        weight_HL = torch.tensor(filter_HL).view(1,1,1,2,2)
        weight_HH = torch.tensor(filter_HH).view(1,1,1,2,2)
        self.weight_LL = nn.Parameter(weight_LL, requires_grad=False)
        self.weight_LH = nn.Parameter(weight_LH, requires_grad=False)
        self.weight_HL = nn.Parameter(weight_HL, requires_grad=False)
        self.weight_HH = nn.Parameter(weight_HH, requires_grad=False)

    def forward(self, x_LL, x, y):
        LL = nn.functional.conv_transpose3d(x_LL.unsqueeze(1), self.weight_LL, stride=(1,2,2)).squeeze(1)
        LH = nn.functional.conv_transpose3d(x[:,[1],...], self.weight_LH, stride=(1,2,2)).squeeze(1)
        HL = nn.functional.conv_transpose3d(x[:,[2],...], self.weight_HL, stride=(1,2,2)).squeeze(1)
        HH = nn.functional.conv_transpose3d(x[:,[3],...], self.weight_HH, stride=(1,2,2)).squeeze(1)
        out = torch.cat((LL,LH,HL,HH, y), dim=1)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, hid_ch, 3, 1, 1),
            nn.BatchNorm2d(hid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class FramingUNet(nn.Module):
    def __init__(self):
        super(FramingUNet, self).__init__()
        self.conv1 = ConvBlock(1, 64, 64)
        self.conv2 = ConvBlock(64, 128, 128)
        self.conv3 = ConvBlock(128, 256, 256)
        self.conv4 = ConvBlock(256, 512, 512)
        self.conv5 = ConvBlock(512, 1024, 512)
        self.conv4_t = ConvBlock(2560, 512, 256)
        self.conv3_t = ConvBlock(1280, 256, 128)
        self.conv2_t = ConvBlock(640, 128, 64)
        self.conv1_t = ConvBlock(320, 64, 64)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)
        self.downsample = Harr_wav()
        self.upsample = Harr_iwav_cat()
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x0):
        x1 = self.conv1(x0)
        wav1 = self.downsample(x1)
        x2 = self.conv2(wav1[:,0,...])
        wav2 = self.downsample(x2)
        x3 = self.conv3(wav2[:,0,...])
        wav3 = self.downsample(x3)
        x4 = self.conv4(wav3[:,0,...])
        wav4 = self.downsample(x4)
        x5 = self.conv5(wav4[:,0,...])
        iwav4 = self.upsample(x5, wav4, x4)
        x4_t = self.conv4_t(iwav4)
        iwav3 = self.upsample(x4_t, wav3, x3)
        x3_t = self.conv3_t(iwav3)
        iwav2 = self.upsample(x3_t, wav2, x2)
        x2_t = self.conv2_t(iwav2)
        iwav1 = self.upsample(x2_t, wav1, x1)
        x1_t = self.conv1_t(iwav1)
        out = self.conv_last(x1_t)
        return out