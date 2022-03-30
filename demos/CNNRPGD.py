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
from recon.models import FBPConvNet
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
        self.model = FBPConvNet()
        self.epochs = args.epochs
        self.T1 = int(self.epochs / 3)
        self.T2 = self.T1 * 2
        self.lr = args.lr
        self.gamma = args.gamma
        self.is_vis_show = args.is_vis_show
        self.show_win = args.show_win
        self.is_res_save = args.is_res_save
        self.res_dir = args.res_dir     
        self.options = options
        if self.is_vis_show:
            self.vis = Visualizer(env='CNNRPGD')

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        current_epoch = self.trainer.current_epoch
        x, p, y = batch
        if current_epoch >= self.T1:
            with torch.no_grad():
                x_t = self(x).detach()
        if current_epoch >= self.T1 and current_epoch < self.T2:
            x = torch.cat((x, x_t), dim=0)
            y = torch.cat((y, y), dim=0)
        elif current_epoch >= self.T2:
            x = torch.cat((x, x_t, y), dim=0)
            y = torch.cat((y, y, y), dim=0)
        out = self(x)
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        if self.is_vis_show:
            self.vis_show(loss.detach(), x.detach(), y.detach(), out.detach())
        return loss
    
    def validation_step(self, batch, batch_idx):
        current_epoch = self.trainer.current_epoch
        x, p, y = batch
        if current_epoch >= self.T1:
            with torch.no_grad():
                x_t = self(x).detach()
        if current_epoch >= self.T1 and current_epoch < self.T2:
            x = torch.cat((x, x_t), dim=0)
            y = torch.cat((y, y), dim=0)
        elif current_epoch >= self.T2:
            x = torch.cat((x, x_t, y), dim=0)
            y = torch.cat((y, y, y), dim=0)
        out = self(x)
        loss = F.mse_loss(out, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, p, y, res_name = batch
        options = self.options.to(x.device)
        alpha = torch.ones(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        # c_set = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        # gamma_set = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        # c_set = [0.5]
        # gamma_set = [0.5, 0.4, 0.3, 0.2, 0.09, 0.08, 0.07, 0.06]
        c_set = [0.5]
        gamma_set=[0.1]
        for c in c_set:
            for gamma in gamma_set:
                x_k = x.clone()
                for i in range(100):
                    p_tmp = ctlib.projection(x_k.contiguous(), options)
                    p_error = p - p_tmp
                    x_error = ctlib.projection_t(p_error.contiguous(), options)
                    z_k = self(x_k + gamma * x_error)
                    norm_new = torch.linalg.vector_norm(z_k - x_k, dim=(2,3), keepdim=True)
                    if i > 0:
                        mask = norm_new > c * norm_old
                        alpha = (c * norm_old/ (norm_new + 1e-8) * mask + ~mask) * alpha
                    norm_old = norm_new
                    x_k = (1 - alpha) * x_k + alpha * z_k            
                out = x_k
                if self.is_res_save:
                    self.res_dir = 'result_c_' + str(c) + '_gamma_' + str(gamma)
                    self.res_save(out, res_name)   

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.99)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.T1], gamma=0.1)
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
    trainer = pl.Trainer(gpus=args.gpu_ids if args.is_specified_gpus else args.gpus, log_every_n_steps=1, max_epochs=args.epochs, callbacks=[checkpoint_callback],strategy="ddp")
    train_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=args.cpus)
    vali_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'vali'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
    test_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'test'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
    trainer.fit(network, train_loader, vali_loader)
    # trainer.fit(network, train_loader, vali_loader, ckpt_path='lightning_logs/version_3/checkpoints/last.ckpt')
    trainer.test(network, test_loader, ckpt_path='best')
    