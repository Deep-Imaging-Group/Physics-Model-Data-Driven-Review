import torch
import torch.nn as nn
import torch.nn.functional as F
from .LEARN import projector
from .LEARN import projector_t


class  BasicBlock(nn.Module):
    """docstring for  BasicBlock"""

    def __init__(self, options, features=32):
        super(BasicBlock, self).__init__()
        self.Sp = nn.Softplus()
        self.conv_D = nn.Conv2d(1, features, (3,3), stride=1, padding=1)
        self.conv_forward = nn.Sequential(
            nn.Conv2d(features, features, (3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, (3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, (3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, (3,3), stride=1, padding=1)
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(features, features, (3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, (3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, (3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, (3,3), stride=1, padding=1)
        )
        self.conv_G = nn.Conv2d(features, 1, (3,3), stride=1, padding=1)
        self.options = nn.Parameter(options, requires_grad=False)
        self.projector = projector()
        self.projector_t = projector_t()

    def forward(self, x, y, W_inv, lambda_step, soft_thr):     
        p_error = self.projector(x, self.options) - y
        x_error = self.projector_t(p_error, self.options) * W_inv
        x = x - self.Sp(lambda_step) * x_error
        x_input = x.clone()        
        x_D = self.conv_D(x_input)
        x_forward = self.conv_forward(x_D)
        x_st = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.Sp(soft_thr)))
        x_backward = self.conv_backward(x_st)
        x_G = self.conv_G(x_backward)
        x_pred = F.relu(x_input + x_G)
        x_D_est = self.conv_backward(x_forward)
        symloss = x_D_est - x_D
        return x_pred, symloss, x_st

class FistaNet(nn.Module):
    def __init__(self, LayerNo, options):
        super(FistaNet, self).__init__()
        self.LayerNo = LayerNo
        W_inv = self.normlize_weight(options)
        self.W_inv = nn.Parameter(W_inv, requires_grad=False)
        self.fcs = nn.ModuleList([BasicBlock(options, features=32) for i in range(LayerNo)])
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)     
        # thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        # gradient step
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))
        self.Sp = nn.Softplus()

    def normlize_weight(self, options):
        height = options[2].int().item()
        width = options[3].int().item()
        x0 = torch.ones(1, 1, height, width)
        p = projector()(x0.double().cuda().contiguous(), options.double().cuda())
        W = projector_t()(p, options.double().cuda())
        W_inv = 1 / (W + 1e-6)
        W_inv[W==0] = 0
        W_inv = W_inv.cpu().float()
        return W_inv
    
    def forward(self, x0, b):
        xold = x0
        y = xold 
        layers_sym = []     
        layers_st = []      
        xnews = []       
        xnews.append(xold)
        for i in range(self.LayerNo):
            theta_ = self.w_theta * i + self.b_theta
            mu_ = self.w_mu * i + self.b_mu
            xnew, layer_sym, layer_st = self.fcs[i](y, b, self.W_inv, mu_, theta_)
            rho_ = (self.Sp(self.w_rho * i + self.b_rho) -  self.Sp(self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)
            y = xnew + rho_ * (xnew - xold)
            xold = xnew
            xnews.append(xnew)
            layers_sym.append(layer_sym)
            layers_st.append(layer_st)
        layers_sym = torch.cat(layers_sym, dim=0)   
        layers_st = torch.cat(layers_st, dim=0)
        return xnew, layers_sym, layers_st