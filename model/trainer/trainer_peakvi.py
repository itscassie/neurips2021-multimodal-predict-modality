""" trainer of peakvi architecture """
import os
import logging
import numpy as np
import anndata as ad
from scipy.sparse import csc_matrix
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from opts import DATASET
from utils.metric import rmse
from utils.loss import L1regularization
from utils.dataloader import SeqDataset
from modules.model_peakvi import ModTransferPEAKVI, PEAKVIModTransfer


class TrainProcess:
    """ the training process for peakvi architecture """
    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter(log_dir=f"../../runs/{args.exp_name}")
        self.device = (
            torch.device(f"cuda:{args.gpu_ids[0]}")
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

        if args.mode in ["atac2gex"]:
            self.model = (
                PEAKVIModTransfer(
                    mod1_dim=args.mod1_dim,
                    mod2_dim=args.mod2_dim,
                    feat_dim=args.emb_dim,
                    hidden_dim=args.hid_dim,
                )
                .to(self.device)
                .float()
            )

        elif args.mode in ["gex2atac"]:
            self.model = (
                ModTransferPEAKVI(
                    mod1_dim=args.mod1_dim,
                    mod2_dim=args.mod2_dim,
                    feat_dim=args.emb_dim,
                    hidden_dim=args.hid_dim,
                )
                .to(self.device)
                .float()
            )
        logging.info(self.model)

        self.mse_loss = nn.MSELoss()
        self.l1reg_loss = L1regularization(weight_decay=0.1)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, eps=0.01, weight_decay=1e-6
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
            print("loading checkpoint ...")
            if os.path.isfile(self.args.checkpoint):
                logging.info(f"loading checkpoint: {self.args.checkpoint}")
                checkpoint = torch.load(self.args.checkpoint)
                self.model.load_state_dict(checkpoint)
            else:
                logging.info(f"no resume checkpoint found at {self.args.checkpoint}")

    def load_pretrain_ae(self):
        """ load pre-trained autoencoder part of the model """
        if self.args.pretrain_weight is not None:
            print("loading pretrain ae weight ...")
            if os.path.isfile(self.args.pretrain_weight):
                logging.info(f"loading checkpoint: {self.args.pretrain_weight}")
                checkpoint = torch.load(self.args.pretrain_weight)
                self.model.autoencoder.load_state_dict(checkpoint)
            else:
                logging.info(
                    f"no resume checkpoint found at {self.args.pretrain_weight}"
                )

    def train_epoch(self, epoch):
        """ training process of each epoch """
        if epoch > 100:
            if args.mode in ["atac2gex"]:
                self.model.peakvae.requires_grad_(False)
                self.args.lr = 0.1
            elif args.mode in ["gex2atac"]:
                self.model.autoencoder.requires_grad_(False)
                self.args.lr = 1e-3
        self.model.train()
        # self.adjust_learning_rate(self.optimizer, epoch)

        total_loss = 0.0
        total_rec_loss = 0.0
        print(f"Epoch {epoch+1:2d} / {self.args.epoch}")

        for batch_idx, (mod1_seq, mod2_seq) in enumerate(self.train_loader):

            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()

            mod2_rec, inference_outputs, generative_outputs = self.model(mod1_seq)

            # peakvi loss
            if self.args.mode in ["atac2gex"]:
                output_loss = self.model.loss(
                    mod1_seq, inference_outputs, generative_outputs
                )
            elif self.args.mode in ["gex2atac"]:
                output_loss = self.model.loss(
                    mod2_seq, inference_outputs, generative_outputs
                )

            peakvi_loss = (
                output_loss["loss"] * int(epoch > 0)
                if self.args.mode in ["gex2atac"]
                else output_loss["loss"]
            )
            reconst_loss = output_loss["rl"]
            kl_local = output_loss["kld"]

            # rec loss
            if self.args.mode in ["atac2gex"]:
                rec_loss = self.mse_loss(mod2_rec, mod2_seq) * int(epoch > 20)

            elif self.args.mode in ["gex2atac"]:
                if epoch < 0:
                    rec_loss = self.mse_loss(mod2_rec, mod2_seq)
                elif epoch >= 0:
                    rec_loss = self.mse_loss(generative_outputs["p"], mod2_seq)

            # l1 loss
            l1reg_loss = (
                self.l1reg_loss(self.model)
                * self.args.reg_loss_weight
                * int(epoch > 20)
            )

            loss = peakvi_loss + rec_loss + l1reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_rec_loss += rec_loss.item()

            print(
                f"Epoch {epoch+1:2d} [{batch_idx+1:2d} /{len(self.train_loader):2d}] | "
                + f"SCVI Total: {total_loss / (batch_idx + 1):.4f} | "
                + f"SCVI Recon: {torch.mean(reconst_loss).item():.4f} | "
                + f"SCVI KL: {torch.mean(kl_local).item():3.4f} | "
                + f"MOD2 REC: {rec_loss.item() * self.args.rec_loss_weight:2.4f} | "
                + f"L1 Reg: {l1reg_loss.item():.4f}"
            )

        train_rmse = np.sqrt(total_rec_loss / len(self.train_loader))
        self.writer.add_scalar("train_rmse", train_rmse, epoch)
        self.writer.add_scalar("rec_loss", rec_loss.item(), epoch)
        logging.info(
            f"Epoch {epoch+1:3d} / {self.args.epoch} | Train RMSE: {train_rmse:.4f}"
        )
        # print(f'Epoch {epoch+1:3d} / {self.args.epoch} | Train RMSE: {train_rmse:.4f}', end=" ")
        self.eval_epoch(epoch)

        # save checkpoint
        if not self.args.dryrun:
            filename = f"../../weights/model_{self.args.arch}_{self.args.mode}_{self.args.name}.pt"
            print(f"saving weight to {filename} ...")
            torch.save(self.model.state_dict(), filename)

    def eval_epoch(self, epoch):
        """ eval process of each epoch """
        self.model.eval()
        total_rec_loss = 0.0
        for _, (mod1_seq, mod2_seq) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()

            mod2_rec, _, generative_outputs = self.model(mod1_seq)
            if self.args.mode in ["atac2gex"]:
                rec_loss = self.mse_loss(mod2_rec, mod2_seq)
            elif self.args.mode in ["gex2atac"]:
                rec_loss = self.mse_loss(generative_outputs["p"], mod2_seq)
            total_rec_loss += rec_loss.item()

        test_rmse = np.sqrt(total_rec_loss / len(self.test_loader))
        # print(f'| Eval RMSE: {test_rmse:.4f}')
        logging.info(
            f"Epoch {epoch+1:3d} / {self.args.epoch} | Eval RMSE: {test_rmse:.4f}"
        )
        self.writer.add_scalar("test_rmse", test_rmse, epoch)

    def run(self):
        """ run the whole training process """
        self.load_pretrain_ae()
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
        use_numpy = True if self.args.mode in ["atac2gex"] else False
        mod2_pred = np.zeros((1, self.args.mod2_dim)) if use_numpy else []

        for _, (mod1_seq, _) in enumerate(self.train_loader):
            mod1_seq = mod1_seq.to(self.device).float()

            mod2_rec, _, generative_outputs = self.model(mod1_seq)
            if self.args.mode in ["atac2gex"]:
                mod2_scvi_rec = mod2_rec
            elif self.args.mode in ["gex2atac"]:
                mod2_scvi_rec = generative_outputs["p"]

            if use_numpy:
                mod2_scvi_rec = mod2_scvi_rec.data.cpu().numpy()
                mod2_pred = np.vstack((mod2_pred, mod2_scvi_rec))
            else:
                mod2_pred.append(mod2_scvi_rec)

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

            mod2_rec, _, generative_outputs = self.model(mod1_seq)
            if self.args.mode in ["atac2gex"]:
                mod2_scvi_rec = mod2_rec
            elif self.args.mode in ["gex2atac"]:
                mod2_scvi_rec = generative_outputs["p"]
            mod2_pred.append(mod2_scvi_rec)

        mod2_pred = torch.cat(mod2_pred).detach().cpu().numpy()
        mod2_pred = csc_matrix(mod2_pred)

        mod2_sol = ad.read_h5ad(DATASET[self.args.mode]["test_mod2"]).X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"Eval RMSE: {rmse_pred:5f}")
