
# sub-parts of the U-Net model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctlib

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch, eps=0.0001, momentum = 0.95, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch, eps=0.0001, momentum = 0.95, track_running_stats=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x_in):
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = x_in+self.outc(x)
        return x

class Wavelet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        D,R=self.GenerateFrameletFilter(frame=1)
        D_tmp=torch.zeros(3,1,3,1)
        for ll in range(3):
            D_tmp[ll,]=torch.from_numpy(np.reshape(D[ll],(-1,1)))
        W=D_tmp
        W2=W.permute(0,1,3,2)
        kernel_dec=np.kron(W.numpy(),W2.numpy())
        kernel_dec=torch.tensor(kernel_dec,requires_grad=False,dtype=torch.float32)
        R_tmp=torch.zeros(3,1,1,3)
        for ll in range(3):
            R_tmp[ll,]=torch.from_numpy(np.reshape(R[ll],(1,-1)))
        R=R_tmp
        R2=R_tmp.permute(0,1,3,2)
        kernel_rec=np.kron(R2.numpy(),R.numpy())
        kernel_rec=torch.tensor(kernel_rec,requires_grad=False,dtype=torch.float32).view(1,9,3,3)
        self.kernel_dec = nn.Parameter(kernel_dec, requires_grad=False)
        self.kernel_rec = nn.Parameter(kernel_rec, requires_grad=False)

    def GenerateFrameletFilter(self, frame):
        # Haar Wavelet
        if frame==0:
            D1=np.array([0.0, 1.0, 1.0] )/2
            D2=np.array([0.0, 1, -1])/2
            D3=('cc')
            R1=np.array([1 , 1 ,0])/2
            R2=np.array([-1, 1, 0])/2
            R3=('cc')
            D=[D1,D2,D3]
            R=[R1,R2,R3]
        # Piecewise Linear Framelet
        elif frame==1:
            D1=np.array([1.0, 2, 1])/4
            D2=np.array([1, 0, -1])/4*np.sqrt(2)
            D3=np.array([-1 ,2 ,-1])/4
            D4='ccc'
            R1=np.array([1, 2, 1])/4
            R2=np.array([-1, 0, 1])/4*np.sqrt(2)
            R3=np.array([-1, 2 ,-1])/4
            R4='ccc'
            D=[D1,D2,D3,D4]
            R=[R1,R2,R3,R4]
        # Piecewise Cubic Framelet
        elif frame==3:
            D1=np.array([1, 4 ,6, 4, 1])/16
            D2=np.array([1 ,2 ,0 ,-2, -1])/8
            D3=np.array([-1, 0 ,2 ,0, -1])/16*np.sqrt(6)
            D4=np.array([-1 ,2 ,0, -2, 1])/8
            D5=np.array([1, -4 ,6, -4, 1])/16
            D6='ccccc'
            R1=np.array([1 ,4, 6, 4 ,1])/16
            R2=np.array([-1, -2, 0, 2, 1])/8
            R3=np.array([-1, 0 ,2, 0, -1])/16*np.sqrt(6)
            R4=np.array([1 ,-2, 0, 2, -1])/8
            R5=np.array([1, -4, 6, -4 ,1])/16
            R6='ccccc'
            D=[D1,D2,D3,D4,D5,D6]
            R=[R1,R2,R3,R4,R5,R6]
        return D,R

    def W(self, img):
        Dec_coeff=F.conv2d(F.pad(img, (1,1,1,1), mode='circular'), self.kernel_dec[1:,...])
        return Dec_coeff

    def Wt(self, Dec_coeff):
        kernel_rec=self.kernel_rec.view(9,1,3,3)
        tem_coeff=F.conv2d(F.pad(Dec_coeff, (1,1,1,1), mode='circular'), kernel_rec[1:,:,...],groups=8)
        rec_img=torch.sum(tem_coeff,dim=1,keepdim=True)
        return rec_img

class MetaInvH(nn.Module):
    '''MetaInvNet with heavy weight CG-Init'''
    def __init__(self, options):
        super(MetaInvH, self).__init__()
        self.CGModule = CGClass(options)

    def forward(self, x, sino, laam, miu, CGInitCNN):
        Wu=self.CGModule.W(x)
        dnz=F.relu(Wu-laam)-F.relu(-Wu-laam)
        PtY=ctlib.projection_t(sino, self.CGModule.options)
        muWtV=self.CGModule.Wt(dnz)
        rhs=PtY+muWtV*miu

        uk0=CGInitCNN(x)
        Ax0=self.CGModule.AWx(uk0,miu)
        res=Ax0-rhs
        img=self.CGModule.CG_alg(uk0, miu, res, CGiter=5)
        return img

class CGClass(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = nn.Parameter(options, requires_grad=False)
        self.Wavelet = Wavelet()

    def AWx(self,img,mu):
        Ax = ctlib.projection(img, self.options)
        AtAx = ctlib.projection_t(Ax, self.options)
        Ax0 = AtAx + self.Wt(self.W(img))*mu
        return Ax0

    def W(self,img):
        return self.Wavelet.W(img)

    def Wt(self,Wu):
        return self.Wavelet.Wt(Wu)

    def pATAp(self,img):
        Ap=ctlib.projection(img, self.options)
        pATApNorm=torch.sum(Ap**2,dim=(1,2,3), keepdim=True)
        return pATApNorm

    def pWTWp(self,img,mu):
        Wp=self.W(img)
        mu_Wp=mu*(Wp**2)
        pWTWpNorm=torch.sum(mu_Wp,dim=(1,2,3), keepdim=True)
        return pWTWpNorm

    def CG_alg(self,x,mu,res,CGiter=20):
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

class MetaInvNet_H(nn.Module):
    def __init__(self, options, layers = 3, InitNet = MetaInvH):
        super(MetaInvNet_H,self).__init__()
        self.layers = layers
        self.net = nn.ModuleList([InitNet(options) for i in range(self.layers+1)])
        self.CGInitCNN=UNet(n_channels=1, n_classes=1)

    def forward(self, fbpu, sino):
        img_list = [None] * (self.layers + 1)      
        laam=0.05
        miu=0.01
        img_list[0] = self.net[0](fbpu.detach(), sino.detach(), laam, miu, self.CGInitCNN)
        inc_lam, inc_miu=0.0008, 0.02 
        for i in range(self.layers):
            laam=laam-inc_lam
            miu=miu+inc_miu
            img_list[i+1] = self.net[i+1](img_list[i], sino.detach(), laam, miu, self.CGInitCNN)
        return img_list