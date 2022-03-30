import torch
import torch.nn as nn
from torch.autograd import Function
import ctlib
from .LEARN import projector

class bprj_fun(Function):
    @staticmethod
    def forward(self, proj, options):
        self.save_for_backward(options)
        return ctlib.backprojection(proj, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        grad_input = ctlib.backprojection_t(grad_output.contiguous(), options)
        return grad_input, None

class backprojector(nn.Module):
    def __init__(self):
        super(backprojector, self).__init__()
        
    def forward(self, image, options):
        return bprj_fun.apply(image, options)

class FBP(nn.Module):
    def __init__(self, options):
        super(FBP, self).__init__()
        dets = int(options[1])
        dDet = options[5]
        s2r = options[8]
        d2r = options[9]
        virdet = dDet * s2r / (s2r + d2r)
        filter = torch.empty(2 * dets - 1)
        pi = torch.acos(torch.tensor(-1.0))
        for i in range(filter.size(0)):
            x = i - dets + 1
            if abs(x) % 2 == 1:
                filter[i] = -1 / (pi * pi * x * x * virdet * virdet)
            elif x == 0:
                filter[i] = 1 / (4 * virdet * virdet)
            else:
                filter[i] = 0
        filter = filter.view(1,1,1,-1)
        w = torch.arange((-dets / 2 + 0.5) * virdet, dets / 2 * virdet, virdet)
        w = s2r / torch.sqrt(s2r ** 2 + w ** 2)
        w = w.view(1,1,1,-1) * virdet
        self.w = nn.Parameter(w, requires_grad=False)
        self.filter = nn.Parameter(filter, requires_grad=False)
        self.options = nn.Parameter(options, requires_grad=False)
        self.backprojector = backprojector()
        self.dets = dets
        self.coef = pi / options[0]

    def forward(self, projection):
        p = projection * self.w
        p = torch.nn.functional.conv2d(p, self.filter, padding=(0,self.dets-1))
        recon = self.backprojector(p, self.options)
        recon = recon * self.coef
        return recon

class fidelity_module(nn.Module):
    def __init__(self, options):
        super(fidelity_module, self).__init__()        
        self.options = nn.Parameter(options, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(1).squeeze())
        self.projector = projector()
        self.fbp = FBP(options)
        
    def forward(self, input_data, proj):
        p_tmp = self.projector(input_data, self.options)
        y_error = proj - p_tmp
        x_error = self.fbp(y_error)
        out = self.weight * x_error + input_data
        return out

class Iter_block(nn.Module):
    def __init__(self, hid_channels, kernel_size, padding, options):
        super(Iter_block, self).__init__()
        self.block1 = fidelity_module(options)
        self.block2 = nn.Sequential(
            nn.Conv2d(1, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, 1, kernel_size=kernel_size, padding=padding)
        )
        self.relu = nn.ReLU(inplace=True)      

    def forward(self, input_data, proj):
        tmp1 = self.block1(input_data, proj)
        tmp2 = self.block2(input_data)
        output = tmp1 + tmp2
        output = self.relu(output)
        return output

class LEARN_FBP(nn.Module):
    def __init__(self, options, block_num=50, hid_channels=48, kernel_size=5, padding=2):
        super(LEARN_FBP, self).__init__()
        self.model = nn.ModuleList([Iter_block(hid_channels, kernel_size, padding, options) for i in range(block_num)])
        for module in self.modules():
            if isinstance(module, fidelity_module):
                module.weight.data.zero_()
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_data, proj):
        x = input_data
        for index, module in enumerate(self.model):
            x = module(x, proj)
        return x