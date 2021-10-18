import os
import math
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
from modules.model_ae import Decoder
from modules.model_concrete import ConcreteSelect

class TrainProcess():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')

        self.trainset = SeqDataset(DATASET[args.mode]['train_mod1'])
        self.testset = SeqDataset(DATASET[args.mode]['test_mod1'])

        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)

        self.selector = ConcreteSelect(
            input_dim=args.mod1_dim, select_dim=args.select_dim).to(self.device).float()
        self.decoder = Decoder(
            input_dim=args.select_dim, out_dim=args.mod1_dim, hidden_dim=args.hid_dim
            ).to(self.device).float()
        
        logging.info(self.selector)
        logging.info(self.decoder)

        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

        self.selector_optimizer = optim.Adam(self.selector.parameters(), lr=self.args.lr, betas=[0.5, 0.999])
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.args.lr, betas=[0.5, 0.999])

    def adjust_learning_rate(self, optimizers, epoch, lr_origin, name=''):
        lr = self.args.lr * (0.5 ** ((epoch - 0) // self.args.lr_decay_epoch))
        if (epoch - 0) % self.args.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
    
    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.isfile(self.args.checkpoint):
                logging.info(f"loading checkpoint: {self.args.checkpoint}")
                checkpoint = torch.load(self.args.checkpoint)
                self.selector.load_state_dict(checkpoint['selector_state_dict'])
                self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            else:
                logging.info(f"no resume checkpoint found at {self.args.checkpoint}")

    def train_epoch(self, epoch, temp):
        self.selector.train()
        self.decoder.train()

        optimizers = [self.decoder_optimizer]
        self.adjust_learning_rate(optimizers, epoch, 0.5 * self.args.lr, 'selector')

        # initialize iterator
        total_rec_loss = 0.0
        min_temp = 0.01
        start_temp = 100.
        steps_per_epoch = (len(self.train_loader.dataset) + self.args.batch_size - 1) // self.args.batch_size

        print(f'Epoch{epoch+1:3d} / {self.args.epoch}')

        for batch_idx, (mod1_seq, _) in enumerate(self.train_loader):

            mod1_seq = mod1_seq.to(self.device).float()

            # concrete
            alpha = math.exp(math.log(min_temp / start_temp) / (self.args.epoch * steps_per_epoch))            
            temp = max(min_temp, temp * alpha)
            selected_features = self.selector(mod1_seq, temp)
            raw_rec = self.decoder(selected_features)

            # loss function
            mod1_selected = mod1_seq[:, torch.argmax(self.selector.logits, dim=1)]
            concrete_loss = self.mse_loss(raw_rec, mod1_seq)

            self.selector_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            concrete_loss.backward()
            self.selector_optimizer.step()
            self.decoder_optimizer.step()


            # show progress
            total_rec_loss += concrete_loss.item()
            mean_max_prob = torch.mean(torch.max(self.softmax(self.selector.logits)))


            logging.info(f'Epoch {epoch+1:2d} [{batch_idx+1:3d} / {len(self.train_loader):3d}] | ' + \
                f'Total: {total_rec_loss / (batch_idx + 1):.4f} | ')
            logging.info(f'                     | Temp: {temp:1.3f}, MMProb: {mean_max_prob.item():.5f}')
        
        index = np.array(torch.argmax(self.selector.logits, dim=1).cpu())

        # save checkpoint
        filename = f"../../weights/model_concrete_{self.args.mode}_{self.args.name}.pt"
        print(f"saving weight to {filename} ...")
        torch.save({
            'epoch': epoch,
            'selector_state_dict': self.selector.state_dict(),
            'decoder_state_dict': self.decoder.state_dict()
            }, filename)

        return temp, mean_max_prob.item(), index

    def run(self):
        self.load_checkpoint()
        logging.info("Selection Process")

        tryout_limit = 5
        start_temp = 100.
        mean_max_target = 0.998
        curr_epoch = 0

        for i in range(tryout_limit):
            temp = start_temp
            for epoch in range(curr_epoch, self.args.epoch):
                logging.info(f'Epoch: {epoch + 1}')
                temp, prob, index = self.train_epoch(epoch, temp)
                curr_epoch = epoch + 1
           
            if prob >= mean_max_target:
                break
            self.args.epoch *= 3

        # Saving index
        logging.info(f'Saving Index (dim = {index.shape[0]})')
        if not os.path.exists(f"../../indexs/{self.args.mode}"):
            os.makedirs(f"../../indexs/{self.args.mode}")

        index_file = open(f'../../indexs/{self.args.mode}/index_{self.args.name}.txt', 'w')
        index_file.write(f'index num: {index.shape[0]}\n')
        for ind in index:
            index_file.write(str(ind)+'\n')


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