import os
import glob
import json
import argparse
import scipy.io as scio
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from recon.models import KSAE
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
        self.model = KSAE()
        self.epochs = args.epochs
        self.alpha = 0.05
        self.gamma = 0.5
        self.beta = 0.005
        self.Nsd = 5
        self.iteration = 50
        self.lr = args.lr
        self.is_vis_show = args.is_vis_show
        self.show_win = args.show_win
        self.is_res_save = args.is_res_save
        self.res_dir = args.res_dir     
        self.options = options
        if self.is_vis_show:
            self.vis = Visualizer(env='KSAE')

    def forward(self, x):
        out = self.model(x)
        return out

    def patch_sample(self, x, patch_size=16, stride=12, h_ind=None, w_ind=None):
        B, C, H, W = x.shape
        Ph = H-patch_size+1
        Pw = W-patch_size+1
        if h_ind is None:
            h_ind = list(range(0, Ph, stride))
            h_ind.append(Ph-1)
            h_ind = np.asarray(h_ind)
            h_ind[1:-1] += np.random.randint((stride - patch_size + 1) / 2, (patch_size - stride) / 2, [len(h_ind)-2])
            h_ind[h_ind > Ph-1] = Ph - 1
        if w_ind is None:
            w_ind = list(range(0, Pw, stride))
            w_ind.append(Pw-1)
            w_ind = np.asarray(w_ind)
            w_ind[1:-1] += np.random.randint((stride - patch_size + 1) / 2, (patch_size - stride) / 2, [len(w_ind)-2])
            w_ind[w_ind > Pw-1] = Pw - 1

        y = torch.empty(B, C, len(h_ind), len(w_ind), patch_size, patch_size, device=x.device)
        for i in range(len(h_ind)):
            for j in range(len(w_ind)):
                y[:,:,i,j,:,:] = x[:,:,h_ind[i]:h_ind[i]+patch_size,w_ind[j]:w_ind[j]+patch_size]

        return y, h_ind, w_ind

    def patch_put(self, y, H, W, h_ind, w_ind, patch_size=16):

        x = torch.zeros(y.size(0), y.size(1), H, W, device=y.device)
        for i in range(len(h_ind)):
            for j in range(len(w_ind)):
                x[:,:,h_ind[i]:h_ind[i]+patch_size,w_ind[j]:w_ind[j]+patch_size] += y[:,:,i,j,:,:]

        return x
    
    def training_step(self, batch, batch_idx):
        x, p, y = batch
        x, h_ind, w_ind = self.patch_sample(x)
        y, _, _, = self.patch_sample(y, h_ind=h_ind, w_ind=w_ind)
        out = self(x)
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        if self.is_vis_show:
            self.vis_show(loss.detach(), x.detach(), y.detach(), out.detach())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, p, y = batch
        options = self.options.to(x.device)
        W = torch.exp(-p)
        M_tmp = ctlib.projection(torch.ones_like(x).contiguous(), options) * W
        M = ctlib.projection_t(M_tmp.contiguous(), options)
        x_t = x.clone()
        z_t = x.clone()
        x_old = x.clone()
        for i in range(self.iteration):
            x_patch, h_ind, w_ind = self.patch_sample(x_t)
            overlap_mask = self.patch_put(torch.ones_like(x_patch), x.size(2), x.size(3), h_ind, w_ind)
            with torch.no_grad():
                y_patch = self(x_patch)
            y_patch.requires_grad = True
            for k in range(self.Nsd):
                y_patch_res = self(y_patch)
                patch_loss = F.mse_loss(y_patch_res, x_patch, reduction='sum')
                grad = torch.autograd.grad(patch_loss, y_patch)[0]
                grad_norm = (grad ** 2).sum((-2, -1), keepdim=True).sqrt()
                y_patch = y_patch - self.alpha * grad / grad_norm
            y_error = (ctlib.projection(x_t.contiguous(), options) - p) * W
            grad_xt_1 = ctlib.projection_t(y_error.contiguous(), options)
            with torch.no_grad():
                y_patch_res = self(y_patch)
            grad_xt_2 = self.beta * self.patch_put(x_patch - y_patch_res, x.size(2), x.size(3), h_ind, w_ind)
            grad_xt = (grad_xt_1 + grad_xt_2) / (M + self.beta * overlap_mask)
            x_t = z_t - grad_xt
            z_t = x_t + self.gamma * (x_t - x_old)
            x_old = x_t
        loss = F.mse_loss(x_t, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, p, y, res_name = batch
        options = self.options.to(x.device)
        W = torch.exp(-p)
        M_tmp = ctlib.projection(torch.ones_like(x).contiguous(), options) * W
        M = ctlib.projection_t(M_tmp.contiguous(), options)
        x_t = x.clone()
        z_t = x.clone()
        x_old = x.clone()
        for i in range(self.iteration):
            x_patch, h_ind, w_ind = self.patch_sample(x_t)
            overlap_mask = self.patch_put(torch.ones_like(x_patch), x.size(2), x.size(3), h_ind, w_ind)
            with torch.no_grad():
                y_patch = self(x_patch)
            y_patch.requires_grad = True
            for k in range(self.Nsd):
                y_patch_res = self(y_patch)
                patch_loss = F.mse_loss(y_patch_res, x_patch, reduction='sum')
                grad = torch.autograd.grad(patch_loss, y_patch)[0]
                grad_norm = (grad ** 2).sum((-2, -1), keepdim=True).sqrt()
                y_patch = y_patch - self.alpha * grad / grad_norm
            y_error = (ctlib.projection(x_t.contiguous(), options) - p) * W
            grad_xt_1 = ctlib.projection_t(y_error.contiguous(), options)
            with torch.no_grad():
                y_patch_res = self(y_patch)
            grad_xt_2 = self.beta * self.patch_put(x_patch - y_patch_res, x.size(2), x.size(3), h_ind, w_ind)
            grad_xt = (grad_xt_1 + grad_xt_2) / (M + self.beta * overlap_mask)
            x_t = z_t - grad_xt
            z_t = x_t + self.gamma * (x_t - x_old)
            x_old = x_t
        out = x_t
        if self.is_res_save:
            self.res_save(out, res_name)

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]     

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
    vali_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'vali'), batch_size=args.batch_size*8, shuffle=False, num_workers=args.cpus)
    test_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'test'), batch_size=args.batch_size*8, shuffle=False, num_workers=args.cpus)
    trainer.fit(network, train_loader, vali_loader)
    # trainer.fit(network, train_loader, vali_loader, ckpt_path='lightning_logs/version_1/checkpoints/last.ckpt')
    trainer.test(network, test_loader, ckpt_path='best')
    