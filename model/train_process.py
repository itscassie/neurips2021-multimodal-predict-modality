import os
import logging
import numpy as np
import anndata as ad
from scipy.sparse import csc_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from opts import model_opts, DATASET
from dataloader import SeqDataset
from model_ae import AutoEncoder
from metric import rmse

class TrainProcess():
    def __init__(self, args):
        self.args = args

        self.trainset = SeqDataset(DATASET[args.mode]['train_mod1'], DATASET[args.mode]['train_mod2'])
        self.testset = SeqDataset(DATASET[args.mode]['test_mod1'])

        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)

        self.model = AutoEncoder(
            input_dim=args.mod1_dim, out_dim=args.mod2_dim, 
            feat_dim=args.feat_dim, hidden_dim=args.hid_dim).cuda().float()
        logging.info(self.model)

        self.mse_loss = nn.MSELoss()

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

        total_rec_loss = 0.0
        print(f'Epoch {epoch+1:2d} / {self.args.epoch}')

        for batch_idx, (mod1_seq, mod2_seq) in enumerate(self.train_loader):

            mod1_seq = mod1_seq.cuda().float()
            mod2_seq = mod2_seq.cuda().float()
            mod2_rec = self.model(mod1_seq)
            rec_loss = self.mse_loss(mod2_rec, mod2_seq)

            self.optimizer.zero_grad()
            rec_loss.backward()
            self.optimizer.step()

            total_rec_loss += rec_loss.item()

            print(f'Epoch {epoch+1:2d} [{batch_idx+1:2d} /{len(self.train_loader):2d}] | ' + \
                f'Total: {total_rec_loss / (batch_idx + 1):.4f}')
        
        # save checkpoint
        PATH = f"weights/model_{args.arch}_{self.args.mode}.pt"
        print(f"saving weight to {PATH} ...")
        torch.save(self.model.state_dict(), PATH)

    def run(self):
        print("start training ...")
        for e in range(self.args.epoch):
            self.train_epoch(e)

    def eval(self):
        print("start eval...")
        self.model.eval()

        mod2_matrix = np.zeros((1, self.args.mod2_dim))

        for batch_idx, (mod1_seq, mod2_seq) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.cuda().float()
            
            mod2_rec = self.model(mod1_seq)
            
            mod2_rec = mod2_rec.data.cpu().numpy()
            mod2_matrix = np.vstack((mod2_matrix, mod2_rec))

        mod2_matrix = csc_matrix(mod2_matrix[1:,])
        mod2_sol = ad.read_h5ad(DATASET[self.args.mode]['test_mod2']).X

        rmse_pred = rmse(mod2_sol, mod2_matrix)
        logging.info(f"RMSE: {rmse_pred:5f}")

