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

from opts import DATASET
from dataloader import SeqDataset
from model_ae import BN_common_encoder, BN_concat_decoder
from metric import rmse

class TrainProcess():
    def __init__(self, args):
        self.args = args

        self.trainset = SeqDataset(DATASET[args.mode]['train_mod1'], DATASET[args.mode]['train_mod2'])
        self.testset = SeqDataset(DATASET[args.mode]['test_mod1'])

        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)

        self.encoder = BN_common_encoder(args.mod1_dim, args.mod2_dim, args.emb_dim, args.hid_dim).cuda()
        self.decoder = BN_concat_decoder(args.emb_dim, args.mod1_dim, args.mod2_dim, args.hid_dim).cuda()
        logging.info(self.encoder)
        logging.info(self.decoder)

        self.mse_loss = nn.MSELoss()

        self.optimizer = optim.SGD(
            [{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}],
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
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            else:
                logging.info(f"no resume checkpoint found at {self.args.checkpoint}")

    def train_epoch(self, epoch):
        self.encoder.train()
        self.decoder.train()
        self.adjust_learning_rate(self.optimizer, epoch)

        total_rec_loss = 0.0
        print(f'Epoch {epoch+1:2d} / {self.args.epoch}')

        for batch_idx, (mod1_seq, mod2_seq) in enumerate(self.train_loader):

            mod1_seq = mod1_seq.cuda().float()
            mod2_seq = mod2_seq.cuda().float()
            
            # model forward
            mod1_common_emb, mod2_common_emb = self.encoder(mod1_seq, mod2_seq)
            zero_resid_emb = torch.zeros_like(mod1_common_emb)
            mod1_recon, mod2_recon = self.decoder(mod1_common_emb, mod2_common_emb,
                                                    zero_resid_emb, zero_resid_emb)


            mod1_loss = self.args.rec_loss_weight * self.mse_loss(mod1_seq, mod1_recon) 
            mod2_loss = self.args.rec_loss_weight * self.mse_loss(mod2_seq, mod2_recon) 
            rec_loss = mod1_loss + mod2_loss

            common_emb_loss = self.args.cmn_loss_weight \
                              * self.mse_loss(mod1_common_emb, mod2_common_emb)

            total_loss = rec_loss + common_emb_loss

            # back prop
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            total_rec_loss += rec_loss.item()

            print(f'Epoch {epoch+1:2d} [{batch_idx+1:2d}/{len(self.train_loader):2d}] | ' + \
                f'Total : {total_rec_loss / (batch_idx + 1):.3f} | ' + \
                f'mod 1 : {mod1_loss.item() :.3f} | ' + \
                f'mod 2 : {mod2_loss.item() :.3f} | ' + \
                f'common: {common_emb_loss.item() :.3f}'
            )
        
        # save checkpoint
        filename = f"weights/model_{self.args.arch}_{self.args.mode}.pt"
        print(f"saving weight to {filename} ...")
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict()
        }, filename)

    def run(self):
        self.load_checkpoint()
        print("start training ...")
        for e in range(self.args.epoch):
            self.train_epoch(e)

    def eval(self):
        print("start eval...")
        self.encoder.eval()
        self.decoder.eval()

        mod2_pred = []
        for batch_idx, (mod1_seq, _) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.cuda().float()
            mod2_zeros = torch.zeros([mod1_seq.size()[0], self.args.mod2_dim]).cuda().float()
            
            # mod1 seq => encode (1) => mod1_cm_emb => decode (2) => mod2_rec
            mod1_cm_emb, mod2_cm_emb = self.encoder(mod1_seq, mod2_zeros)
            zero_resid_emb = torch.zeros_like(mod1_cm_emb)
            _, mod2_rec = self.decoder(mod2_cm_emb, mod1_cm_emb,
                                        zero_resid_emb, zero_resid_emb)

            mod2_pred.append(mod2_rec)

        mod2_pred = torch.cat(mod2_pred).detach().cpu().numpy()
        mod2_pred = csc_matrix(mod2_pred)

        mod2_sol = ad.read_h5ad(DATASET[self.args.mode]['test_mod2']).X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"RMSE: {rmse_pred:5f}")

