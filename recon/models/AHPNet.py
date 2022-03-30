import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import ctlib

def filter_gen():
    h0 = torch.tensor([1/4, 1/2, 1/4]).unsqueeze(-1)
    h1 = torch.tensor([-1/4, 1/2, 1/4]).unsqueeze(-1)
    h2 = torch.tensor([math.sqrt(2)/4, 0, -math.sqrt(2)/4]).unsqueeze(-1)
    h = [h0, h1, h2]
    filter = []
    for i in range(3):
        for j in range(3):
            if i == 0 and j == 0:
                continue
            f = h[i] @ h[j].t()
            f = f.view(1,1,3,3)
            filter.append(f)
    filter = torch.cat(filter, dim=0)
    return filter

class MLP(nn.Module):
    def __init__(self, options) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9,9),
            nn.ReLU(True),
            nn.Linear(9,9),
            nn.ReLU(True),
            nn.Linear(9,8),
            nn.ReLU(True)
        )
        self.filter = nn.Parameter(filter_gen(),requires_grad=False)
        self.options = nn.Parameter(options,requires_grad=False)
    
    def forward(self, x, z, p):
        r0 = p - ctlib.projection(x, self.options)
        rk = z - F.conv2d(x, self.filter, stride=1, padding=1)
        r0_norm = (r0 ** 2).sum(dim=(2,3))
        rk_norm = (rk ** 2).sum(dim=(2,3))
        r_norm = torch.cat((r0_norm, rk_norm), dim=1)
        beta = self.model(r_norm)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return beta

class CNN(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(k, 64, 3, 1, 1))
        layers.append(nn.ReLU(True))
        for i in range(17):
            layers.append(nn.Conv2d(64, 64, 3, 1, 1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(64, 1, 3, 1, 1))
        layers.append(nn.ReLU(True))
        self.model = nn.Sequential(*layers)
        self.filter = nn.Parameter(filter_gen(),requires_grad=False)

    def forward(self, x):
        x_tilde = self.model(x)
        z = F.conv2d(x_tilde, self.filter, stride=1, padding=1)
        return z

class CGModule(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = nn.Parameter(options, requires_grad=False)
        self.filter = nn.Parameter(filter_gen(),requires_grad=False)
        self.filter_t = nn.Parameter(self.filter.flip((2,3)),requires_grad=False)

    def AWx(self,img,mu):
        Ax = ctlib.projection(img, self.options)
        AtAx = ctlib.projection_t(Ax, self.options)
        Ax0 = AtAx + self.Ft(self.F(img), mu)
        return Ax0

    def F(self,x):
        return F.conv2d(x, self.filter, stride=1, padding=1)

    def Ft(self,y, mu):
        Ft = F.conv2d(y, self.filter_t, stride=1, padding=1, groups=8) * mu
        return Ft.sum(dim=1, keepdim=True)

    def pATAp(self,img):
        Ap=ctlib.projection(img, self.options)
        pATApNorm=torch.sum(Ap**2,dim=(1,2,3), keepdim=True)
        return pATApNorm

    def pWTWp(self,img,mu):
        Wp=self.F(img)
        mu_Wp=mu*(Wp**2)
        pWTWpNorm=torch.sum(mu_Wp,dim=(1,2,3), keepdim=True)
        return pWTWpNorm

    def CG_alg(self,x,mu,y,z,CGiter=20):
        Aty = ctlib.projection_t(y, self.options)
        Ftz = self.Ft(z, mu)
        res = Aty + Ftz
        r=res
        p=-res
        for k in range(CGiter):
            pATApNorm = self.pATAp(p)
            mu_pWtWpNorm=self.pWTWp(p,mu)
            rTr=torch.sum(r**2,dim=(1,2,3), keepdim=True)
            alphak = rTr / (mu_pWtWpNorm+pATApNorm)
            x = x+alphak*p
            r = r+alphak*self.AWx(p,mu)
            betak = torch.sum(r**2,dim=(1,2,3), keepdim=True)/ rTr
            p=-r+betak*p

        pATApNorm = self.pATAp(p)
        mu_pWtWpNorm=self.pWTWp(p,mu)
        rTr=torch.sum(r**2,dim=(1,2,3), keepdim=True)
        alphak = rTr/(mu_pWtWpNorm+pATApNorm)
        x = x+alphak*p
        return x

class IterBlock(nn.Module):
    def __init__(self, k, options) -> None:
        super().__init__()
        self.Dcnn = CNN(k)
        self.Pmlp = MLP(options)
        self.CGModule = CGModule(options)
        self.filter = nn.Parameter(filter_gen(),requires_grad=False)

    def forward(self, x, p):
        z = self.Dcnn(x)
        beta = self.Pmlp(x[:,[-1],:,:].detach(), z.detach(), p)
        x_t = self.CGModule.CG_alg(x[:,[-1],:,:], beta, p, z, CGiter=5)
        return x_t


class AHPNet(nn.Module):
    def __init__(self, options, layers=3):
        super(AHPNet,self).__init__()
        self.layers = layers
        self.model = nn.ModuleList([IterBlock(i+1, options) for i in range(self.layers)])

    def forward(self, x, p):
        res = x.clone()
        for model in self.model:
            x_t = model(res, p)
            res = torch.cat((res, x_t), dim=1)
        return res