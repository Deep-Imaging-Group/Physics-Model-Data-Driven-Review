import os
import glob
import json
import argparse
import scipy.io as scio
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from recon.models import dCNN
from vis_tools import Visualizer
import ctlib

def setup_parser(arguments, title):
    parser = argparse.ArgumentParser(description=title,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key, val in arguments.items():
        parser.add_argument('--%s' % key,
                            type=eval(val["type"]),
                            help=val["help"],
                            default=val["default"],
                            nargs=val["nargs"] if "nargs" in val else None)
    return parser

def get_parameters(title=None):
    with open("config.json") as data_file:
        data = json.load(data_file)
    parser = setup_parser(data, title)
    parameters = parser.parse_args()
    return parameters

class net(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        options = torch.tensor([args.views, args.dets, args.width, args.height,
                                args.dImg, args.dDet, args.Ang0, args.dAng,
                                args.s2r, args.d2r, args.binshift, args.scan_type])
        self.model = nn.ModuleList([dCNN() for i in range(100)])
        self.epochs = args.epochs
        self.lr = args.lr
        self.is_vis_show = args.is_vis_show
        self.show_win = args.show_win
        self.is_res_save = args.is_res_save
        self.res_dir = args.res_dir     
        self.options = options        
        self.automatic_optimization = False
        if self.is_vis_show:
            self.vis = Visualizer(env='MomentumNet')
        photon_path = os.path.join(args.data_root_dir, 'photoncounting.mat')
        photon = scio.loadmat(photon_path)['photon']
        photon = torch.FloatTensor(photon).squeeze()
        self.photon = photon / photon.max()


    def forward(self, x, idx):
        out = self.model[idx](x)
        return out

    def training_step(self, batch, batch_idx):    
        current_epoch = self.trainer.current_epoch    
        x, p, y = batch
        options = self.options.to(x.device)
        # photon = self.photon.to(x.device)
        W = torch.exp(-p) #* photon
        M_tmp = ctlib.projection(torch.ones_like(x).contiguous(), options) * W
        M = ctlib.projection_t(M_tmp.contiguous(), options)
        svdvals = torch.linalg.svdvals(M)
        gamma = (svdvals[...,0] - svdvals[...,-1]) / 167.64
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        M_tilde = M + gamma
        M_inv = 1 / M_tilde
        x_k = x.clone()
        x_pre = x_k
        t_old = 1.0
        rho = 0.5
        for i in range(100):
            if current_epoch == 0 and i > 0:
                self.model[i].load_state_dict(self.model[i-1].state_dict())
            opt = self.optimizers()[i]
            opt.zero_grad()
            out = self(x_k, i)
            loss_i = F.mse_loss(out, y)
            self.manual_backward(loss_i)
            opt.step()
            with torch.no_grad():
                x_r = self(x_k, i)
            z_k = (1 - rho) * x_k + rho * x_r
            t_new = (1 + math.sqrt(1 + 4 * t_old ** 2)) / 2
            x_t = x_k + (t_old - 1) / t_new * (x_k - x_pre)
            x_pre = x_k
            t_old = t_new
            y_error = (ctlib.projection(x_t.contiguous(), options) - p) * W
            grad_xt = ctlib.projection_t(y_error.contiguous(), options) + gamma * (x_t - z_k)
            z_t = x_t - grad_xt * M_inv
            mask = z_t.abs() < M_inv
            x_k = z_t.clone()
            x_k[mask] = 0
        loss = F.mse_loss(x_k, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        if self.is_vis_show:
            self.vis_show(loss, x.detach(), y.detach(), x_k.detach())
    
    def validation_step(self, batch, batch_idx):
        x, p, y = batch
        options = self.options.to(x.device)
        # photon = self.photon.to(x.device)
        W = torch.exp(-p) #* photon
        M_tmp = ctlib.projection(torch.ones_like(x).contiguous(), options) * W
        M = ctlib.projection_t(M_tmp.contiguous(), options)
        svdvals = torch.linalg.svdvals(M)
        gamma = (svdvals[...,0] - svdvals[...,-1]) / 167.64
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        M_tilde = M + gamma
        M_inv = 1 / M_tilde
        x_k = x.clone()
        x_pre = x_k
        t_old = 1.0
        rho = 0.5
        for i in range(100):
            x_r = self(x_k, i)
            z_k = (1 - rho) * x_k + rho * x_r
            t_new = (1 + math.sqrt(1 + 4 * t_old ** 2)) / 2
            x_t = x_k + (t_old - 1) / t_new * (x_k - x_pre)
            x_pre = x_k
            t_old = t_new
            y_error = (ctlib.projection(x_t.contiguous(), options) - p) * W
            grad_xt = ctlib.projection_t(y_error.contiguous(), options) + gamma * (x_t - z_k)
            z_t = x_t - grad_xt * M_inv
            mask = z_t.abs() < M_inv
            x_k = z_t.clone()
            x_k[mask] = 0
        loss = F.mse_loss(x_k, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, p, y, res_name = batch
        options = self.options.to(x.device)
        # photon = self.photon.to(x.device)
        W = torch.exp(-p) #* photon
        M_tmp = ctlib.projection(torch.ones_like(x).contiguous(), options) * W
        M = ctlib.projection_t(M_tmp.contiguous(), options)
        svdvals = torch.linalg.svdvals(M)
        gamma = (svdvals[...,0] - svdvals[...,-1]) / 167.64
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        M_tilde = M + gamma
        M_inv = 1 / M_tilde
        x_k = x.clone()
        x_pre = x_k
        t_old = 1.0
        rho = 0.5
        for i in range(100):
            x_r = self(x_k, i)
            z_k = (1 - rho) * x_k + rho * x_r
            t_new = (1 + math.sqrt(1 + 4 * t_old ** 2)) / 2
            x_t = x_k + (t_old - 1) / t_new * (x_k - x_pre)
            x_pre = x_k
            t_old = t_new
            y_error = (ctlib.projection(x_t.contiguous(), options) - p) * W
            grad_xt = ctlib.projection_t(y_error.contiguous(), options) + gamma * (x_t - z_k)
            z_t = x_t - grad_xt * M_inv
            mask = z_t.abs() < M_inv
            x_k = z_t.clone()
            x_k[mask] = 0
        out = x_k
        if self.is_res_save:
            self.res_save(out, res_name)

    def configure_optimizers(self):
        optimizer = [torch.optim.AdamW(self.model[i].parameters(), lr=self.lr) for i in range(100)]
        scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[i], T_max=200) for i in range(100)]
        return optimizer, scheduler     

    def show_win_norm(self, y):
        x = y.clone()
        x[x<self.show_win[0]] = self.show_win[0]
        x[x>self.show_win[1]] = self.show_win[1]
        x = (x - self.show_win[0]) / (self.show_win[1] - self.show_win[0]) * 255
        return x

    def vis_show(self, loss, x, y, out, mode='Train'):
        self.vis.plot(mode + ' Loss', loss.item())
        self.vis.img(mode + ' Ground Truth', self.show_win_norm(y).cpu())
        self.vis.img(mode + ' Input', self.show_win_norm(x).cpu())
        self.vis.img(mode + ' Result', self.show_win_norm(out).cpu())

    def res_save(self, out, res_name):
        res = out.cpu().numpy()
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)
        for i in range(res.shape[0]):
            scio.savemat(self.res_dir + '/' + res_name[i], {'data':res[i].squeeze()})

class data_loader(Dataset):
    def __init__(self, root, dose, mode):
        self.x_dir_name = 'input_' + dose
        self.x_path = os.path.join(root, mode, self.x_dir_name)   
        self.mode = mode
        self.files_x = np.array(sorted(glob.glob(os.path.join(self.x_path, 'data') + '*.mat')))
        
    def __getitem__(self, index):
        file_x = self.files_x[index]
        file_p = file_x.replace('input', 'projection')
        file_y = file_x.replace(self.x_dir_name, 'label')
        input_data = scio.loadmat(file_x)['data']
        prj_data = scio.loadmat(file_p)['data']
        label_data = scio.loadmat(file_y)['data']
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        if self.mode == 'train' or self.mode == 'vali':
            return input_data, prj_data, label_data
        elif self.mode == 'test':
            res_name = file_x[-13:]
            return input_data, prj_data, label_data, res_name

    def __len__(self):
        return len(self.files_x)
    
if __name__ == "__main__":
    args = get_parameters()
    network = net(args)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=3, mode="min")
    trainer = pl.Trainer(gpus=args.gpu_ids if args.is_specified_gpus else args.gpus, log_every_n_steps=1, max_epochs=args.epochs, callbacks=[checkpoint_callback])
    train_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=args.cpus)
    vali_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'vali'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
    test_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'test'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
    trainer.fit(network, train_loader, vali_loader)
    # trainer.fit(network, train_loader, vali_loader, ckpt_path='lightning_logs/version_3/checkpoints/last.ckpt')
    trainer.test(network, test_loader, ckpt_path='best')
    