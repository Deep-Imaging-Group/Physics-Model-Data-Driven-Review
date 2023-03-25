import torch.nn as nn

class dCNN(nn.Module):
    def __init__(self):
        super(dCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):        
        return x + self.model(x)
