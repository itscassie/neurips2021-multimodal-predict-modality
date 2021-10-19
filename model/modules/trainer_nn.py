import os
import logging
import numpy as np
import anndata as ad
from scipy.sparse import csc_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils.metric import rmse
from utils.dataloader import SeqDataset
from utils.loss import L1regularization
from opts import model_opts, DATASET
from modules.model_ae import AutoEncoder, UnbAutoEncoder, Decoder
from tensorboardX import SummaryWriter

class TrainProcess():
    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter(log_dir=f'../../runs/{args.exp_name}')
        self.device = torch.device(f'cuda:{args.gpu_ids[0]}') if args.gpu_ids else torch.device('cpu')

        if args.selection:
            self.trainset = SeqDataset(
                DATASET[args.mode]['train_mod1'], DATASET[args.mode]['train_mod2'],
                mod1_idx_path=args.mod1_idx_path, mod2_idx_path=args.mod2_idx_path
                )
            self.testset = SeqDataset(
                DATASET[args.mode]['test_mod1'], DATASET[args.mode]['test_mod2'], 
                mod1_idx_path=args.mod1_idx_path, mod2_idx_path=args.mod2_idx_path
                )

            args.mod1_dim = args.select_dim if args.mod1_idx_path != None else args.mod1_dim
            args.mod2_dim = args.select_dim if args.mod2_idx_path != None else args.mod2_dim
            
        else:
            self.trainset = SeqDataset(DATASET[args.mode]['train_mod1'], DATASET[args.mode]['train_mod2'])
            self.testset = SeqDataset(DATASET[args.mode]['test_mod1'], DATASET[args.mode]['test_mod2'])

        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)

        if self.args.arch == 'nn':
            self.model = AutoEncoder(
                input_dim=args.mod1_dim, out_dim=args.mod2_dim, 
                feat_dim=args.emb_dim, hidden_dim=args.hid_dim).to(self.device).float()
        elif self.args.arch == 'unb_ae':
            self.model = UnbAutoEncoder(
                input_dim=args.mod1_dim, out_dim=args.mod2_dim, 
                feat_dim=args.emb_dim, hid_dim_en=100, hid_dim_de=args.hid_dim).to(self.device).float()
        elif self.args.arch == 'decoder':
            self.model = Decoder(
                input_dim=args.mod1_dim, out_dim=args.mod2_dim, 
                hidden_dim=args.hid_dim).to(self.device).float()
        logging.info(self.model)

        self.mse_loss = nn.MSELoss()
        self.l1reg_loss = L1regularization(weight_decay=0.1)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.args.lr * (0.5 ** ((epoch - 0) // self.args.lr_decay_epoch))
        if (epoch - 0) % self.args.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.isfile(self.args.checkpoint):
                logging.info(f"loading checkpoint: {self.args.checkpoint}")
                checkpoint = torch.load(self.args.checkpoint)
                self.model.load_state_dict(checkpoint)
            else:
                logging.info(f"no resume checkpoint found at {self.args.checkpoint}")

    def train_epoch(self, epoch):
        self.model.train()
        self.adjust_learning_rate(self.optimizer, epoch)

        total_loss = 0.0
        total_rec_loss = 0.0
        print(f'Epoch {epoch+1:2d} / {self.args.epoch}')

        for batch_idx, (mod1_seq, mod2_seq) in enumerate(self.train_loader):

            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()
            mod2_rec = self.model(mod1_seq)
            rec_loss = self.mse_loss(mod2_rec, mod2_seq)
            l1reg_loss = self.l1reg_loss(self.model) * self.args.reg_loss_weight

            loss = rec_loss + l1reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_rec_loss += rec_loss.item()

            print(f'Epoch {epoch+1:2d} [{batch_idx+1:2d} /{len(self.train_loader):2d}] | ' + \
                f'Total: {total_loss / (batch_idx + 1):.4f} | ' + \
                f'Rec: {rec_loss.item():.4f} | ' + \
                f'L1 Reg: {l1reg_loss.item():.4f}')

        
        
        train_rmse = np.sqrt(total_rec_loss / len(self.train_loader))
        self.writer.add_scalar("train_rmse", train_rmse, epoch)
        self.writer.add_scalar("rec_loss", rec_loss.item(), epoch)
        print(f'Epoch {epoch+1:3d} / {self.args.epoch} | Train RMSE: {train_rmse:.4f}', end=" ")
        self.eval_epoch(epoch)

        # save checkpoint
        if not self.args.dryrun:
            filename = f"../../weights/model_{self.args.arch}_{self.args.mode}_{self.args.name}.pt"
            print(f"saving weight to {filename} ...")
            torch.save(self.model.state_dict(), filename)

    def eval_epoch(self, epoch):
        self.model.eval()

        total_rec_loss = 0.0
        for batch_idx, (mod1_seq, mod2_seq) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()
            mod2_rec = self.model(mod1_seq)
            rec_loss = self.mse_loss(mod2_rec, mod2_seq)
            total_rec_loss += rec_loss.item()
        test_rmse = np.sqrt(total_rec_loss / len(self.test_loader))
        print(f'| Eval RMSE: {test_rmse:.4f}')
        self.writer.add_scalar("test_rmse", test_rmse, epoch)


    def run(self):
        self.load_checkpoint()
        print("start training ...")
        for e in range(self.args.epoch):
            self.train_epoch(e)

    def eval(self):
        print("start eval...")
        self.model.eval()

        logging.info(f"Mode: {self.args.mode}")
        
        # train set rmse
        use_numpy = True if self.args.mode == 'atac2gex' else False
        mod2_pred = np.zeros((1, self.args.mod2_dim)) if use_numpy else []

        for batch_idx, (mod1_seq, _) in enumerate(self.train_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_rec = self.model(mod1_seq)

            if use_numpy:
                mod2_rec = mod2_rec.data.cpu().numpy()
                mod2_pred = np.vstack((mod2_pred, mod2_rec))
            else:
                mod2_pred.append(mod2_rec)

        if use_numpy:
            mod2_pred = mod2_pred[1:,]
        else:
            mod2_pred = torch.cat(mod2_pred).detach().cpu().numpy()

        mod2_pred = csc_matrix(mod2_pred)

        mod2_sol = ad.read_h5ad(DATASET[self.args.mode]['train_mod2']).X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"Train RMSE: {rmse_pred:5f}")

        # test set rmse
        mod2_pred = []
        for batch_idx, (mod1_seq, _) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            
            mod2_rec = self.model(mod1_seq)
            mod2_pred.append(mod2_rec)

        mod2_pred = torch.cat(mod2_pred).detach().cpu().numpy()
        mod2_pred = csc_matrix(mod2_pred)

        mod2_sol = ad.read_h5ad(DATASET[self.args.mode]['test_mod2']).X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"Eval RMSE: {rmse_pred:5f}")