import torch
import torch.nn as nn
from torch.autograd import Function
import ctlib
from .LEARN import fidelity_module

class adj_weight(nn.Module):
    def __init__(self, k):
        super(adj_weight, self).__init__()
        self.k = k

    def forward(self, x):
        return ctlib.laplacian(x, self.k)

def img2patch(x, patch_size, stride):
    x_size = x.size()
    Ph = x_size[-2]-patch_size+1
    Pw = x_size[-1]-patch_size+1
    y = torch.empty(*x_size[:-2], Ph, Pw, patch_size, patch_size, device=x.device)
    for i in range(patch_size):
        for j in range(patch_size):
            y[...,i,j] = x[...,i:i+Ph,j:j+Ph]
    return y[...,::stride,::stride,:,:]

def patch2img(y, patch_size, stride, x_size):
    Ph = x_size[-2]-patch_size+1
    Pw = x_size[-1]-patch_size+1
    y_tmp = torch.zeros(*x_size[:-2], Ph, Pw, patch_size, patch_size, device=y.device)
    y_tmp[...,::stride,::stride,:,:] = y
    x = torch.zeros(*x_size, device=y.device)
    for i in range(patch_size):
        for j in range(patch_size):
            x[...,i:i+Ph,j:j+Ph] += y_tmp[...,i,j]
    return x

class img2patch_fun(Function):    

    @staticmethod
    def forward(self, x, size):
        self.save_for_backward(size)
        patch_size = size[0]
        stride = size[1]
        p_size = size[5:]
        y = img2patch(x, patch_size, stride)
        out = y.reshape(y.size(0), p_size[1]*p_size[2], p_size[3]*p_size[4])
        return out

    @staticmethod
    def backward(self, grad_output):
        size = self.saved_tensors[0]
        patch_size = size[0]
        stride = size[1]
        x_size = size[2:5]
        p_size = size[5:]
        y = grad_output.view(grad_output.size(0), *p_size)
        grad_input = patch2img(y, patch_size, stride, (grad_output.size(0), *x_size))
        return grad_input, None

class patch2img_fun(Function):

    @staticmethod
    def forward(self, x, size):
        self.save_for_backward(size)
        patch_size = size[0]
        stride = size[1]
        x_size = size[2:5]
        p_size = size[5:]
        y = x.view(x.size(0), *p_size)
        out = patch2img(y, patch_size, stride, (x.size(0), *x_size))
        return out

    @staticmethod
    def backward(self, grad_output):
        size = self.saved_tensors[0]
        patch_size = size[0]
        stride = size[1]
        p_size = size[5:]
        y = img2patch(grad_output, patch_size, stride)
        grad_input = y.reshape(grad_output.size(0), p_size[1]*p_size[2], p_size[3]*p_size[4])
        return grad_input, None

class Im2Patch(nn.Module):
    def __init__(self, patch_size, stride, img_size) -> None:
        super(Im2Patch, self).__init__()
        Ph = (img_size-patch_size) // stride + 1
        Pw = (img_size-patch_size) // stride + 1
        self.size = torch.LongTensor([patch_size, stride, 1, img_size, img_size, 1, Ph, Pw, patch_size, patch_size])

    def forward(self, x):
        return img2patch_fun.apply(x, self.size)

class Patch2Im(nn.Module):
    def __init__(self, patch_size, stride, img_size) -> None:
        super(Patch2Im, self).__init__()
        Ph = (img_size-patch_size) // stride + 1
        Pw = (img_size-patch_size) // stride + 1
        self.size = torch.LongTensor([patch_size, stride, 1, img_size, img_size, 1, Ph, Pw, patch_size, patch_size])
        m = torch.ones(1, Ph * Pw, patch_size ** 2)
        mask = patch2img_fun.apply(m, self.size)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        y = patch2img_fun.apply(x, self.size)
        out = y / self.mask
        return out

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))

    def forward(self, x, adj):
        t = x.view(-1, x.size(2))
        support = torch.mm(t, self.weight)
        support = support.view(x.size(0), x.size(1), -1)
        out = torch.zeros_like(support)
        for i in range(x.size(0)):
            out[i] = torch.mm(adj[i], support[i])
        out = out + self.bias
        return out


class Iter_block(nn.Module):
    def __init__(self, hid_channels, kernel_size, padding, img_size, p_size, stride, gcn_hid_ch, options):
        super(Iter_block, self).__init__()
        self.block1 = fidelity_module(options)
        self.block2 = nn.Sequential(
            nn.Conv2d(1, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, 1, kernel_size=kernel_size, padding=padding)
        )
        self.block3 = GCN(p_size**2, gcn_hid_ch)
        self.block4 = GCN(gcn_hid_ch, p_size**2)
        self.image2patch = Im2Patch(p_size, stride, img_size)
        self.patch2image = Patch2Im(p_size, stride, img_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data, proj, adj):
        tmp1 = self.block1(input_data, proj)
        tmp2 = self.block2(input_data)
        patch = self.image2patch(input_data)
        tmp3 = self.relu(self.block3(patch, adj))
        tmp3 = self.block4(tmp3, adj)
        tmp3 = self.patch2image(tmp3)
        output = tmp1 + tmp2 + tmp3
        output = self.relu(output)
        return output

class MAGIC(nn.Module):
    def __init__(self, options, block_num=50, hid_channels=64, kernel_size=5, padding=2, img_size=256, p_size=6, stride=2, gcn_hid_ch=64, k=9):
        super(MAGIC, self).__init__()
        self.block1 = nn.ModuleList([Iter_block(hid_channels, kernel_size, padding, img_size, p_size, stride, gcn_hid_ch, options) for i in range(block_num//2)])
        self.block2 = nn.ModuleList([Iter_block(hid_channels, kernel_size, padding, img_size, p_size, stride, gcn_hid_ch, options) for i in range(block_num//2)])
        self.adj_weight = adj_weight(k)
        self.image2patch = Im2Patch(p_size, stride, img_size)
        for module in self.modules():
            if isinstance(module, fidelity_module):
                module.weight.data.zero_()
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, GCN):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                module.bias.data.zero_()
    
    def forward(self, input_data, proj):
        x = input_data
        patch1 = self.image2patch(x)
        adj1 = []
        for i in range(input_data.size(0)):
            adj1.append(self.adj_weight(patch1[i]))
        for index, module in enumerate(self.block1):
            x = module(x, proj, adj1)
        adj2 = []
        patch2 = self.image2patch(x)
        for i in range(input_data.size(0)):
            adj2.append(self.adj_weight(patch2[i]))
        for index, module in enumerate(self.block2):
            x = module(x, proj, adj2)
        return x
