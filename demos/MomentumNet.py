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
cuda = True if torch.cuda.is_available() else False
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

class net(nn.Module):
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
        if self.is_vis_show:
            self.vis = Visualizer(env='MomentumNet')
        self.train_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=args.cpus)
        self.vali_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'vali'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
        self.test_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'test'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
        if os.path.exists('checkpoints.pth'):
            state_dict = torch.load('checkpoints.pth')
            self.model.load_state_dict(state_dict['model'])
            self.current_module_idx = state_dict['current_module_idx']
        else:
            self.current_module_idx = 0
        if cuda:
            self.model = self.model.cuda()
        self.optimizer = [torch.optim.AdamW(self.model[i].parameters(), lr=self.lr) for i in range(100)]
        self.scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer[i], T_max=200) for i in range(100)]


    def forward(self, x, idx):
        out = self.model[idx](x)
        return out

    def train(self):
        options = self.options.cuda()
        for module_idx in range(self.current_module_idx, 100):
            if module_idx > 0:
                state_dict = torch.load('checkpoints.pth')
                self.model.load_state_dict(state_dict['model'])
                self.model[module_idx].load_state_dict(self.model[module_idx-1].state_dict())
            optimizer = self.optimizer[module_idx]
            scheduler = self.scheduler[module_idx]
            vali_loss_min = None
            loss_increase = 0
            for epoch in range(self.epochs):                
                self.model.train()
                for batch_index, data in enumerate(self.train_loader):
                    x, p, y = data                
                    if cuda:
                        x = x.cuda()
                        p = p.cuda()
                        y = y.cuda()
                    x_k = x.clone()
                    if module_idx > 0:
                        W = torch.exp(-p) * 2e5
                        # W = W / W.amax((2,3), keepdim=True)
                        M_tmp = ctlib.projection(torch.ones_like(x).contiguous(), options) * W
                        M = ctlib.projection_t(M_tmp.contiguous(), options)
                        svdvals = torch.linalg.svdvals(M)
                        gamma = (svdvals[...,0] - svdvals[...,-1]) / 167.64
                        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                        M_tilde = M + gamma
                        M_inv = 1 / M_tilde                        
                        x_pre = x_k
                        t_old = 1.0
                        rho = 0.5
                        for i in range(module_idx):
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
                    
                    optimizer.zero_grad()
                    out = self(x_k, module_idx)
                    loss = F.mse_loss(out, y)
                    loss.backward()
                    optimizer.step()
                    print(
                        "[Module %d/100] [Epoch %d/%d] [Batch %d/%d]: [loss: %f]"
                        % (module_idx+1, epoch+1, self.epochs, batch_index+1, len(self.train_loader), loss.item())
                    )
                scheduler.step()
                self.model.eval()
                vali_loss = self.validation(module_idx)
                if vali_loss_min is None or vali_loss < vali_loss_min:
                    torch.save(
                        {'model':self.model.state_dict(), 'current_module_idx':module_idx}, 'checkpoints.pth'
                    )
                    vali_loss_min = vali_loss
                    loss_increase = 0
                else:
                    loss_increase = loss_increase + 1
                if loss_increase >= 5:
                    break
    
    def validation(self, module_idx):
        options = self.options.cuda()
        loss = 0
        for batch_index, data in enumerate(self.vali_loader):
            x, p, y = data
            if cuda:
                x = x.cuda()
                p = p.cuda()
                y = y.cuda()
            with torch.no_grad():
                W = torch.exp(-p) * 2e5
                # W = W / W.amax((2,3), keepdim=True)
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
                for i in range(module_idx):
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
                x_k = self(x_k, module_idx)
            loss0 = F.mse_loss(x_k, y)
            loss += loss0.item()
        loss = loss / len(self.vali_loader)
        return loss

    def test(self):
        options = self.options.cuda()
        self.model.eval()
        for batch_index, data in enumerate(self.test_loader):
            x, p, y, res_name = data
            if cuda:
                x = x.cuda()
                p = p.cuda()
                y = y.cuda()            
            with torch.no_grad():
                W = torch.exp(-p) * 2e5
                # W = W / W.amax((2,3), keepdim=True)
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
                for i in range(99):
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
                out = self(x_k, 99)
                out = x_k
            if self.is_res_save:
                self.res_save(out, res_name)   

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
    # network.train()
    network.test()
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=3, mode="min")
    # trainer = pl.Trainer(gpus=args.gpu_ids if args.is_specified_gpus else args.gpus, log_every_n_steps=1, max_epochs=args.epochs, callbacks=[checkpoint_callback])
    # train_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=args.cpus)
    # vali_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'vali'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
    # test_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'test'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
    # trainer.fit(network, train_loader, vali_loader)
    # # trainer.fit(network, train_loader, vali_loader, ckpt_path='lightning_logs/version_1/checkpoints/last.ckpt')
    # trainer.test(network, test_loader, ckpt_path='best')
    
