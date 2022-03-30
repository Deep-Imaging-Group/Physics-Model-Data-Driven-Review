import torch
import torch.nn as nn

class KSAE(nn.Module):
    def __init__(self, imgsize=256, hid_ch=1024, sparsity=100) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(imgsize, hid_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hid_ch, hid_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hid_ch, hid_ch),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hid_ch, hid_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hid_ch, hid_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hid_ch, imgsize),
        )
        self.sparsity = sparsity

    def forward(self, x):
        B, C, Ph, Pw, H, W = x.shape
        x_in = x.view(B*C*Ph*Pw, H*W).contiguous()
        feature = self.encoder(x_in)
        mask = torch.zeros_like(feature)
        _, indices = torch.topk(feature.detach(), self.sparsity, dim=-1, sorted=False)
        mask.scatter_(1, indices, 1.0)
        feature = feature * mask
        res = self.decoder(feature)
        out = res.view(B, C, Ph, Pw, H, W).contiguous()
        return out
