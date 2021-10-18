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
from utils.dataloader import SeqDataset
from modules.model_ae import *
from utils.loss import *
from utils.metric import rmse

class TrainProcess():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')

        self.trainset = SeqDataset(DATASET[args.mode]['train_mod1'], DATASET[args.mode]['train_mod2'])
        self.testset = SeqDataset(DATASET[args.mode]['test_mod1'])

        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)

        self.cmn_encoder = BN_common_encoder(args.mod1_dim, args.mod2_dim, args.emb_dim, args.hid_dim).to(self.device)
        self.resid_encoder = BN_resid_encoder(args.mod1_dim, args.mod2_dim, args.emb_dim, args.hid_dim).to(self.device)
        self.decoder = BN_concat_decoder(args.emb_dim, args.mod1_dim, args.mod2_dim, args.hid_dim).to(self.device)

        logging.info(self.cmn_encoder)
        logging.info(self.resid_encoder)
        logging.info(self.decoder)

        self.mse_loss = nn.MSELoss()
        self.cos_loss = CosineLoss()

        self.optimizer = optim.SGD([
            {'params': self.cmn_encoder.parameters()}, 
            {'params': self.resid_encoder.parameters()},
            {'params': self.decoder.parameters()}],
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
                self.cmn_encoder.load_state_dict(checkpoint['cmn_encoder_state_dict'])
                self.resid_encoder.load_state_dict(checkpoint['resid_encoder_state_dict'])
                self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            else:
                logging.info(f"no resume checkpoint found at {self.args.checkpoint}")

    def load_pretrain(self):
        if self.args.pretrain_weight is not None:
            if os.path.isfile(self.args.pretrain_weight):
                logging.info(f"loading pretrain weight: {self.args.pretrain_weight}")
                checkpoint = torch.load(self.args.pretrain_weight)
                self.cmn_encoder.load_state_dict(checkpoint['cmn_encoder_state_dict'])
                self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            else:
                logging.info(f"no resume checkpoint found at {self.args.pretrain_weight}")

    def pretrain_epoch(self, epoch):
        self.cmn_encoder.train()
        self.decoder.train()
        self.adjust_learning_rate(self.optimizer, epoch)

        # for param in self.cmn_encoder.parameters():
        #     print(param.requires_grad)

        total_rec_loss = 0.0
        print(f'Epoch {epoch+1:2d} / {self.args.pretrain_epoch}')

        for batch_idx, (mod1_seq, mod2_seq) in enumerate(self.train_loader):

            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()
            
            # model forward
            mod1_cmn_emb, mod2_cmn_emb = self.cmn_encoder(mod1_seq, mod2_seq)
            zero_resid_emb = torch.zeros_like(mod1_cmn_emb)
            mod1_rec, mod2_rec = self.decoder(mod1_cmn_emb, mod2_cmn_emb,
                                            zero_resid_emb, zero_resid_emb)


            mod1_loss = self.args.rec_loss_weight * self.mse_loss(mod1_seq, mod1_rec) 
            mod2_loss = self.args.rec_loss_weight * self.mse_loss(mod2_seq, mod2_rec) 
            rec_loss = mod1_loss + mod2_loss

            cmn_emb_loss = self.args.cmn_loss_weight \
                              * self.mse_loss(mod1_cmn_emb, mod2_cmn_emb)

            total_loss = rec_loss + cmn_emb_loss

            # back prop
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            total_rec_loss += rec_loss.item()

            print(f'Epoch {epoch+1:2d} [{batch_idx+1:2d}/{len(self.train_loader):2d}] | ' + \
                f'Total: {total_rec_loss / (batch_idx + 1):.3f} | ' + \
                f'mod 1: {mod1_loss.item() :.3f} | ' + \
                f'mod 2: {mod2_loss.item() :.3f} | ' + \
                f'Common: {cmn_emb_loss.item() :.3f}'
            )
        
        # save checkpoint
        filename = f"../../weights/model_pretrain_{self.args.arch}_{self.args.mode}.pt"
        print(f"saving weight to {filename} ...")
        torch.save({
            'epoch': epoch,
            'cmn_encoder_state_dict': self.cmn_encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict()
        }, filename)

    def train_epoch(self, epoch):
        self.cmn_encoder.requires_grad_(False)
        self.resid_encoder.train()
        self.decoder.train()
        self.adjust_learning_rate(self.optimizer, epoch)

        # for param in self.cmn_encoder.parameters():
        #     print(param.requires_grad)

        total_rec_loss = 0.0
        print(f'Epoch {epoch+1:2d} / {self.args.epoch}')

        for batch_idx, (mod1_seq, mod2_seq) in enumerate(self.train_loader):

            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()
            
            # model forward
            mod1_cmn_emb, mod2_cmn_emb = self.cmn_encoder(mod1_seq, mod2_seq)
            mod1_resid_emb, mod2_resid_emb = self.resid_encoder(mod1_seq, mod2_seq)
            
            mod1_rec, mod2_rec = self.decoder(mod1_cmn_emb, mod2_cmn_emb,
                                            mod1_resid_emb, mod2_resid_emb)


            mod1_loss = self.args.rec_loss_weight * self.mse_loss(mod1_seq, mod1_rec) 
            mod2_loss = self.args.rec_loss_weight * self.mse_loss(mod2_seq, mod2_rec) 
            rec_loss = mod1_loss + mod2_loss

            cmn_emb_loss = self.args.cmn_loss_weight \
                        * self.mse_loss(mod1_cmn_emb, mod2_cmn_emb)

            ortho_loss = self.args.cos_loss_weight \
                        * self.cos_loss(mod1_cmn_emb, mod2_cmn_emb, mod1_resid_emb, mod2_resid_emb)

            total_loss = rec_loss + cmn_emb_loss + ortho_loss

            # back prop
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            total_rec_loss += rec_loss.item()

            print(f'Epoch {epoch+1:2d} [{batch_idx+1:2d}/{len(self.train_loader):2d}] | ' + \
                f'Total: {total_rec_loss / (batch_idx + 1):.3f} | ' + \
                f'mod 1: {mod1_loss.item() :.3f} | ' + \
                f'mod 2: {mod2_loss.item() :.3f} | ' + \
                f'common: {cmn_emb_loss.item() :.3f} | ' + \
                f'ortho: {ortho_loss.item() :.3f} | '
            )
        
        # save checkpoint
        filename = f"weights/model_{self.args.arch}_{self.args.mode}.pt"
        print(f"saving weight to {filename} ...")
        torch.save({
            'epoch': epoch,
            'cmn_encoder_state_dict': self.cmn_encoder.state_dict(),
            'resid_encoder_state_dict': self.resid_encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict()
        }, filename)

    def run(self):
        if self.args.pretrain_weight is not None:
            print("loading pretrain weight ...")
            self.load_pretrain()

        print("start pretraining ...")
        for e in range(self.args.pretrain_epoch):
            self.pretrain_epoch(e)

        if self.args.checkpoint is not None:
            print("loading checkpoint ...")
            self.load_checkpoint()

        print("start training ...")
        for e in range(self.args.epoch):
            self.train_epoch(e)

    def eval(self):
        print("start eval...")

        self.cmn_encoder.eval()
        self.resid_encoder.eval()
        self.decoder.eval()
        logging.info(f"Mode: {self.args.mode}")
        
        # train set rmse
        use_numpy = True if self.args.mode == 'atac2gex' else False
        mod2_pred = np.zeros((1, self.args.mod2_dim)) if use_numpy else []

        for batch_idx, (mod1_seq, _) in enumerate(self.train_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_zeros = torch.zeros([mod1_seq.size()[0], self.args.mod2_dim]).to(self.device).float()
            #  mod1 seq => (mod1) cmn encoder => mod1_cmn_emb => (mod2) decoder => mod2_rec
            mod1_cmn_emb, _ = self.cmn_encoder(mod1_seq, mod2_zeros)
            zero_emb = torch.zeros_like(mod1_cmn_emb)
            _, mod2_rec = self.decoder(zero_emb, mod1_cmn_emb,
                                        zero_emb, zero_emb)

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
            mod2_zeros = torch.zeros([mod1_seq.size()[0], self.args.mod2_dim]).to(self.device).float()
            """
            mod1 seq => (mod1) cmn encoder => mod1_cmn_emb => (mod2) decoder => mod2_rec
            """
            mod1_cmn_emb, _ = self.cmn_encoder(mod1_seq, mod2_zeros)
            zero_emb = torch.zeros_like(mod1_cmn_emb)
            _, mod2_rec = self.decoder(zero_emb, mod1_cmn_emb,
                                        zero_emb, zero_emb)

            mod2_pred.append(mod2_rec)

        mod2_pred = torch.cat(mod2_pred).detach().cpu().numpy()
        mod2_pred = csc_matrix(mod2_pred)

        mod2_sol = ad.read_h5ad(DATASET[self.args.mode]['test_mod2']).X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"Eval RMSE: {rmse_pred:5f}")

