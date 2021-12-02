""" trainer of pix2pix architecture """
import os
import logging
import numpy as np
import anndata as ad
from scipy.sparse import csc_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from opts import DATASET
from utils.metric import rmse
from utils.dataloader import SeqDataset
from modules.model_ae import Pix2Pix


class TrainProcess:
    """ the training process for pix2pix arch """
    def __init__(self, args):
        self.args = args
        self.device = (
            torch.device("cuda:{}".format(args.gpu_ids[0]))
            if args.gpu_ids
            else torch.device("cpu")
        )

        mod1_idf = np.load(args.idf_path) if args.tfidf != 0 else None
        self.trainset = SeqDataset(
            DATASET[args.mode]["train_mod1"],
            DATASET[args.mode]["train_mod2"],
            mod1_idx_path=args.mod1_idx_path,
            tfidf=args.tfidf,
            mod1_idf=mod1_idf,
        )
        self.testset = SeqDataset(
            DATASET[args.mode]["test_mod1"],
            DATASET[args.mode]["test_mod2"],
            mod1_idx_path=args.mod1_idx_path,
            tfidf=args.tfidf,
            mod1_idf=mod1_idf,
        )

        self.train_loader = DataLoader(
            self.trainset, batch_size=args.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.testset, batch_size=args.batch_size, shuffle=False
        )

        self.model = (
            Pix2Pix(
                input_dim=args.mod1_dim,
                out_dim=args.mod2_dim,
                feat_dim=args.emb_dim,
                hidden_dim=args.hid_dim,
            )
            .to(self.device)
            .float()
        )
        logging.info(self.model)

        self.mse_loss = nn.MSELoss()
        self.adv_loss = nn.BCELoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=5e-4,
        )
        self.optimizer_g = optim.Adam(
            self.model.parameters(), lr=args.lr * 2e-3, betas=[0.5, 0.999]
        )
        self.optimizer_d = optim.Adam(
            self.model.parameters(), lr=args.lr * 2e-3, betas=[0.5, 0.999]
        )

    def adjust_learning_rate(self, optimizer, epoch):
        """ learning rate adjustment method """
        lr = self.args.lr * (0.5 ** ((epoch - 0) // self.args.lr_decay_epoch))
        if (epoch - 0) % self.args.lr_decay_epoch == 0:
            print("LR is set to {}".format(lr))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def load_checkpoint(self):
        """ load pre-trained model checkpoint """
        if self.args.checkpoint is not None:
            if os.path.isfile(self.args.checkpoint):
                logging.info(f"loading checkpoint: {self.args.checkpoint}")
                checkpoint = torch.load(self.args.checkpoint)
                self.model.load_state_dict(checkpoint)
            else:
                logging.info(f"no resume checkpoint found at {self.args.checkpoint}")

    def train_epoch(self, epoch):
        """ training process of each epoch """
        self.model.train()
        self.adjust_learning_rate(self.optimizer, epoch)

        total_rec_loss = 0.0
        print(f"Epoch {epoch+1:2d} / {self.args.epoch}")

        for batch_idx, (mod1_seq, mod2_seq) in enumerate(self.train_loader):
            bsz = mod1_seq.shape[0]
            real_label = torch.ones(bsz).to(self.device)
            fake_label = torch.zeros(bsz).to(self.device)

            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()

            # Reconstruction phase
            mod2_rec, _, _ = self.model(mod1_seq, mod2_seq)
            rec_loss = self.mse_loss(mod2_rec, mod2_seq)
            self.optimizer.zero_grad()
            rec_loss.backward()
            self.optimizer.step()

            # G phase
            _, z_fake_score, _ = self.model(mod1_seq, mod2_seq)
            z_loss = self.adv_loss(z_fake_score, real_label)
            g_loss = z_loss

            self.optimizer_g.zero_grad()
            g_loss.backward()
            self.optimizer_g.step()

            # D phase
            _, z_fake_score, z_real_score = self.model(mod1_seq, mod2_seq)
            z_loss = self.adv_loss(z_fake_score, fake_label) + self.adv_loss(
                z_real_score, real_label
            )
            d_loss = z_loss * 0.5

            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()

            total_rec_loss += rec_loss.item()

            print(
                f"Epoch {epoch+1:2d} [{batch_idx+1:2d} /{len(self.train_loader):2d}] | "
                + f"Total: {total_rec_loss / (batch_idx + 1):.4f} | "
                + f"G: {g_loss.item() :.5f} | "
                + f"D: {d_loss.item() :.5f} | "
            )

        # save checkpoint
        filename = (
            f"../../weights/model_{self.args.arch}_{self.args.mode}_{self.args.name}.pt"
        )
        print(f"saving weight to {filename} ...")
        torch.save(self.model.state_dict(), filename)

    def run(self):
        """ run the whole training process """
        self.load_checkpoint()
        print("start training ...")
        for e in range(self.args.epoch):
            self.train_epoch(e)

    def eval(self):
        """ eval the trained model on train / test set """
        print("start eval...")
        self.model.eval()

        logging.info(f"Mode: {self.args.mode}")

        # train set rmse
        use_numpy = True if self.args.mode == "atac2gex" else False
        mod2_pred = np.zeros((1, self.args.mod2_dim)) if use_numpy else []

        for _, (mod1_seq, _) in enumerate(self.train_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = (
                torch.zeros([mod1_seq.size()[0], self.args.mod2_dim])
                .to(self.device)
                .float()
            )
            mod2_rec, _, _ = self.model(mod1_seq, mod2_seq)

            if use_numpy:
                mod2_rec = mod2_rec.data.cpu().numpy()
                mod2_pred = np.vstack((mod2_pred, mod2_rec))
            else:
                mod2_pred.append(mod2_rec)

        if use_numpy:
            mod2_pred = mod2_pred[
                1:,
            ]
        else:
            mod2_pred = torch.cat(mod2_pred).detach().cpu().numpy()

        mod2_pred = csc_matrix(mod2_pred)

        mod2_sol = ad.read_h5ad(DATASET[self.args.mode]["train_mod2"]).X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"Train RMSE: {rmse_pred:5f}")

        # test set rmse
        mod2_pred = []
        for _, (mod1_seq, _) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = (
                torch.zeros([mod1_seq.size()[0], self.args.mod2_dim])
                .to(self.device)
                .float()
            )
            mod2_rec, _, _ = self.model(mod1_seq, mod2_seq)
            mod2_pred.append(mod2_rec)

        mod2_pred = torch.cat(mod2_pred).detach().cpu().numpy()
        mod2_pred = csc_matrix(mod2_pred)

        mod2_sol = ad.read_h5ad(DATASET[self.args.mode]["test_mod2"]).X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"Eval RMSE: {rmse_pred:5f}")
