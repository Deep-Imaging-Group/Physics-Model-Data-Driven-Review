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
from recon.models import iCTNet
from vis_tools import Visualizer

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
        self.model = iCTNet(args.views, args.dets, args.width, args.height, args.dAng)
        self.epochs = args.epochs
        self.lr = args.lr
        self.is_vis_show = args.is_vis_show
        self.show_win = args.show_win
        self.is_res_save = args.is_res_save
        self.res_dir = args.res_dir     
        self.automatic_optimization = False
        if self.is_vis_show:
            self.vis = Visualizer(env='iCTNet')
        dets = args.dets
        dDet = args.dDet
        s2r = args.s2r
        d2r = args.d2r
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
        self.filter = nn.Parameter(filter, requires_grad=False)
        self.dets = dets

    def forward(self, p):
        out = self.model(p)
        return out

    def training_step(self, batch, batch_idx):
        current_epoch = self.trainer.current_epoch    
        x, p, y, z = batch
        if current_epoch < 200:
            out = self.model.segment1(p)
            loss = F.mse_loss(out, z)
        elif current_epoch < 400:
            out = self.model.segment2(z)
            loss = F.mse_loss(out, z)
        elif current_epoch < 600:
            pf = torch.nn.functional.conv2d(z, self.filter, padding=(0,self.dets-1))
            out = self.model.segment3(z)
            loss = F.mse_loss(out, pf)
        elif current_epoch < 800:
            pf = torch.nn.functional.conv2d(z, self.filter, padding=(0,self.dets-1))
            out = self.model.segment4(pf)
            loss = F.mse_loss(out, y)
        else:
            out = self(p)
            loss = F.mse_loss(out, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        current_epoch = self.trainer.current_epoch
        x, p, y, z = batch
        if current_epoch >= 800:
            out = self(p)
            loss = F.mse_loss(out, y)
        else:
            loss = 1e5
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, p, y, z, res_name = batch
        out = self(p)
        if self.is_res_save:
            self.res_save(out, res_name)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[920, 960], gamma=0.1)
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
        file_py = file_x.replace(self.x_dir_name, 'projection')
        input_data = scio.loadmat(file_x)['data']
        prj_data = scio.loadmat(file_p)['data']
        label_data = scio.loadmat(file_y)['data']
        prj_label = scio.loadmat(file_py)['data']
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        prj_label = torch.FloatTensor(prj_label).unsqueeze_(0)
        if self.mode == 'train' or self.mode == 'vali':
            return input_data, prj_data, label_data, prj_label
        elif self.mode == 'test':
            res_name = file_x[-13:]
            return input_data, prj_data, label_data, prj_label, res_name

    def __len__(self):
        return len(self.files_x)
    
if __name__ == "__main__":
    args = get_parameters()
    network = net(args)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=3, mode="min")
    trainer = pl.Trainer(gpus=args.gpu_ids if args.is_specified_gpus else args.gpus, log_every_n_steps=1, max_epochs=args.epochs, callbacks=[checkpoint_callback], strategy="ddp")
    train_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=args.cpus)
    vali_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'vali'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
    test_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'test'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
    trainer.fit(network, train_loader, vali_loader)
    # trainer.fit(network, train_loader, vali_loader, ckpt_path='lightning_logs/version_1/checkpoints/last.ckpt')
    trainer.test(network, test_loader, ckpt_path='best')
    