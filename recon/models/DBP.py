import torch
import torch.nn as nn
from torch.autograd import Function
import ctlib

class bprj_sv_fun(Function):
    @staticmethod
    def forward(self, proj, options):
        self.save_for_backward(options)
        return ctlib.backprojection_sv(proj, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        temp = grad_output.sum(1, keepdim=True)
        grad_input = ctlib.backprojection_t(temp.contiguous(), options)
        return grad_input, None

class backprojector_sv(nn.Module):
    def __init__(self):
        super(backprojector_sv, self).__init__()
        
    def forward(self, proj, options):
        return bprj_sv_fun.apply(proj, options)

class DBP(nn.Module):
    def __init__(self, options) -> None:
        super().__init__()
        self.options = nn.Parameter(options, requires_grad=False)
        layers = []
        layers.append(nn.Conv2d(512, 64, 3, 1, 1))
        layers.append(nn.ReLU(inplace=True))
        for i in range(15):
            layers.append(nn.Conv2d(64, 64, 3, 1, 1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)
        self.backprojector = backprojector_sv()

    def forward(self, p):
        x = self.backprojector(p, self.options)
        out = self.model(x)
        return out