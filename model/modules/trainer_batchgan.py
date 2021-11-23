import os
import pickle
import logging
import numpy as np
import anndata as ad
from scipy.sparse import csc_matrix
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from utils.metric import rmse
from utils.dataloader import BatchSeqDataset
from utils.loss import L1regularization
from opts import model_opts, DATASET
from modules.model_ae import BatchRemovalGAN

class TrainProcess():
    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter(log_dir=f'../../runs/{args.exp_name}')
        self.device = torch.device(f'cuda:{args.gpu_ids[0]}') if args.gpu_ids else torch.device('cpu')

        mod1_idf = np.load(args.idf_path) if args.tfidf != 0 else None
        self.trainset = BatchSeqDataset(
            DATASET[args.mode]['train_mod1'], DATASET[args.mode]['train_mod2'],
            mod1_idx_path=args.mod1_idx_path, tfidf=args.tfidf, mod1_idf=mod1_idf, 
            batch_list=args.train_batch
        )
        logging.info(f"TRAIN_NUM: {len(self.trainset):5d}")
        if args.mode not in ['gex2atac_p2', 'gex2adt_p2', 'adt2gex_p2', 'atac2gex_p2']:
            self.testset = BatchSeqDataset(
                DATASET[args.mode]['test_mod1'], DATASET[args.mode]['test_mod2'], 
                mod1_idx_path=args.mod1_idx_path, tfidf=args.tfidf, mod1_idf=mod1_idf, 
                batch_list=args.test_batch
            )
            logging.info(f"TEST_NUM : {len(self.testset):5d}")

        else:
            train_size = int(0.9 * len(self.trainset))
            test_size = len(self.trainset) - train_size
            logging.info(f"(SPLITED) TRAIN_SIZE: {train_size:5d}, TEST_SIZE: {test_size:5d}")
            self.trainset, self.testset = random_split(
                self.trainset, [train_size, test_size], 
                generator=torch.Generator().manual_seed(args.seed)
            )

        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)

        self.model = BatchRemovalGAN(
            input_dim=args.mod1_dim, out_dim=args.mod2_dim, 
            feat_dim=args.emb_dim, hidden_dim=args.hid_dim, dropout=args.dropout
        ).to(self.device).float()
        
        logging.info(self.model)

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.l1reg_loss = L1regularization(weight_decay=0.1)
        self.adv_loss = nn.CrossEntropyLoss()
        self.eval_best = float("inf")

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

        self.optimizer_G = optim.Adam(self.model.parameters(), lr=args.lr*2e-3, betas=[0.5, 0.999])
        self.optimizer_D = optim.Adam(self.model.parameters(), lr=args.lr*2e-3, betas=[0.5, 0.999])

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

        for batch_idx, (mod1_seq, mod2_seq, b_label) in enumerate(self.train_loader):

            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()
            b_label = b_label.to(self.device).long()

            
            # Reconstruction phase
            mod2_rec, _ = self.model(mod1_seq)
            rec_loss = self.mse_loss(mod2_rec, mod2_seq)
            l1reg_loss = self.l1reg_loss(self.model) * self.args.reg_loss_weight
            loss = rec_loss + l1reg_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # G phase
            _, cls_prob = self.model(mod1_seq)
            g_loss = -self.adv_loss(cls_prob, b_label) * 0.1 # maximize class loss

            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()

            
            # D phase
            _, cls_prob = self.model(mod1_seq)
            d_loss = self.adv_loss(cls_prob, b_label) * 0.1 # minimize class loss
    
            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()


            total_loss += loss.item()
            total_rec_loss += rec_loss.item()

            print(f'Epoch {epoch+1:2d} [{batch_idx+1:2d} /{len(self.train_loader):2d}] | ' + \
                f'Total: {total_loss / (batch_idx + 1):.4f} | ' + \
                f'Rec L2: {rec_loss.item():.4f} | ' + \
                f'L1 Reg: {l1reg_loss.item():.4f} | ' + \
                f'G: {g_loss.item() :.5f} | ' + \
                f'D: {d_loss.item() :.5f} | ')
        
        
        train_rmse = np.sqrt(total_rec_loss / len(self.train_loader))
        test_rmse = self.eval_epoch(epoch)
        (self.eval_best, save_best) = (test_rmse, True) if test_rmse < self.eval_best else (self.eval_best, False)

        logging.info(
            f'Epoch {epoch+1:3d} / {self.args.epoch} | Train RMSE: {train_rmse:.4f}' + \
            f'| Eval RMSE: {test_rmse:.4f} | Eval best: {self.eval_best:.4f}'
        )
        self.writer.add_scalar("train_rmse", train_rmse, epoch)
        self.writer.add_scalar("rec_loss", rec_loss.item(), epoch)
        self.writer.add_scalar("test_rmse", test_rmse, epoch)
        
        # save checkpoint
        if not self.args.dryrun:
            filename = f"../../weights/model_{self.args.exp_name}.pt"
            print(f"saving weight to {filename} ...")
            torch.save(self.model.state_dict(), filename)

            # best weight
            if save_best and epoch > self.args.save_best_from:
                filename = f"../../weights/model_best_{self.args.exp_name}.pt"
                print(f"saving best weight to {filename} ...")
                torch.save(self.model.state_dict(), filename)

    def eval_epoch(self, epoch):
        self.model.eval()

        total_rec_loss = 0.0
        for batch_idx, (mod1_seq, mod2_seq, b_label) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()            
            mod2_rec, _ = self.model(mod1_seq)

            rec_loss = self.mse_loss(mod2_rec, mod2_seq)
            total_rec_loss += rec_loss.item()
        test_rmse = np.sqrt(total_rec_loss / len(self.test_loader))

        return test_rmse


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
        if self.args.mode in ['gex2atac', 'gex2adt', 'adt2gex']:
            use_numpy = False
            mod2_pred = []
        else:
            use_numpy = True
            mod2_pred = np.zeros((1, self.args.mod2_dim))

        for batch_idx, (mod1_seq, _, _) in enumerate(self.train_loader):
            mod1_seq = mod1_seq.to(self.device).float()            
            mod2_rec, _ = self.model(mod1_seq)
            
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

        mod2_data = ad.read_h5ad(DATASET[self.args.mode]['train_mod2'])
        if len(self.args.train_batch) != 0:
            batch_data = mod2_data[mod2_data.obs["batch"]=='s0'] # empty anndata
            for b in self.args.train_batch:
                batch_data = ad.concat(
                    (batch_data, mod2_data[mod2_data.obs["batch"]==b]), 
                    axis=0, join="outer", index_unique="-")
            mod2_data = batch_data
        mod2_sol = mod2_data.X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"Train RMSE: {rmse_pred:5f}")

        # test set rmse
        mod2_pred = []
        for batch_idx, (mod1_seq, _, _) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            
            mod2_rec, _ = self.model(mod1_seq)
            mod2_pred.append(mod2_rec)

        mod2_pred = torch.cat(mod2_pred).detach().cpu().numpy()
        mod2_pred = csc_matrix(mod2_pred)

        mod2_sol = ad.read_h5ad(DATASET[self.args.mode]['test_mod2']).X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"Eval RMSE: {rmse_pred:5f}")