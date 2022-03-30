import torch.nn as nn

class RED_CNN(nn.Module):
    def __init__(self, hid_channels=48, kernel_size=5, padding=2):
        super(RED_CNN, self).__init__()
        self.conv_1 = nn.Conv2d(1, hid_channels, kernel_size=kernel_size, padding=padding)
        self.conv_2 = nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding)
        self.conv_3 = nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding)
        self.conv_4 = nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding)
        self.conv_5 = nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding)
        self.conv_t_1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding)
        self.conv_t_2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding)
        self.conv_t_3 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding)
        self.conv_t_4 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding)
        self.conv_t_5= nn.ConvTranspose2d(hid_channels, 1, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        # encoder
        residual_1 = x.clone()
        out = self.relu(self.conv_1(x))
        out = self.relu(self.conv_2(out))
        residual_2 = out.clone()
        out = self.relu(self.conv_3(out))
        out = self.relu(self.conv_4(out))
        residual_3 = out.clone()
        out = self.relu(self.conv_5(out))

        # decoder
        out = self.conv_t_1(out)
        out = out + residual_3
        out = self.conv_t_2(self.relu(out))
        out = self.conv_t_3(self.relu(out))
        out = out + residual_2
        out = self.conv_t_4(self.relu(out))
        out = self.conv_t_5(self.relu(out))
        out = out + residual_1
        out = self.relu(out)
        return out
