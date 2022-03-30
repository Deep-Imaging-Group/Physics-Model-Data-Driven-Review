import torch
import torch.nn as nn
from .LEARN import projector
from .LEARN import projector_t

class primal_module(nn.Module):
    def __init__(self, n_primal, hid_channels, kernel_size, padding, options):
        super(primal_module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_primal+1, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(hid_channels, n_primal, kernel_size=kernel_size, padding=padding),
        )
        self.options = nn.Parameter(options, requires_grad=False)
        self.projector_t = projector_t()

    def forward(self, x, h):
        t = self.projector_t(h, self.options)
        inputs = torch.cat((x, t), dim=1)
        return x + self.model(inputs)

class dual_module(nn.Module):
    def __init__(self, n_dual, hid_channels, kernel_size, padding, options):
        super(dual_module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_dual+2, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(hid_channels, n_dual, kernel_size=kernel_size, padding=padding),
        )
        self.options = nn.Parameter(options, requires_grad=False)
        self.projector = projector()

    def forward(self, x, y, h):
        t = self.projector(x, self.options)
        inputs = torch.cat((h,t,y), dim=1)
        return h + self.model(inputs)

class Learned_primal_dual(nn.Module):
    def __init__(self, options, n_iter=10, n_primal=5, n_dual=5, hid_channels=32, kernel_size=3, padding=1):
        super(Learned_primal_dual, self).__init__()
        self.primal_models = nn.ModuleList([primal_module(n_primal, hid_channels, kernel_size, padding, options) for i in range(n_iter)])
        self.dual_models = nn.ModuleList([dual_module(n_dual, hid_channels, kernel_size, padding, options) for i in range(n_iter)])        
        self.n_iter = n_iter
        self.n_primal = n_primal
        self.n_dual = n_dual
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x0, y):
        h0 = torch.zeros(y.size(0), self.n_dual, y.size(2), y.size(3), device=y.device)
        x0 = x0.expand(x0.size(0), self.n_primal, x0.size(2), x0.size(3))
        for i in range(self.n_iter):
            h = self.dual_models[i](x0[:,[1],:,:], y, h0)
            x = self.primal_models[i](x0, h[:,[0],:,:])
            x0 = x
            h0 = h
        return x[:,[0],:,:]