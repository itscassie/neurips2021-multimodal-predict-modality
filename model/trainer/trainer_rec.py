""" trainer of reconstruction architecture (baseline for scvi and peakvi) """
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
from modules.model_scvi import VAE
from modules.model_ae import AutoEncoder
from modules.model_peakvi import PEAKVAE


class TrainProcess:
    """ the training process for rec arch """
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

        if args.arch == "rec":
            self.model = (
                AutoEncoder(
                    input_dim=args.mod1_dim,
                    out_dim=args.mod1_dim,
                    feat_dim=args.emb_dim,
                    hidden_dim=args.hid_dim,
                )
                .to(self.device)
                .float()
            )
        elif args.arch == "peakrec":
            self.model = PEAKVAE(args.mod1_dim).to(self.device).float()

        elif args.arch == "scvirec":
            self.model = VAE(args.mod1_dim).to(self.device).float()

        logging.info(self.model)

        self.mse_loss = nn.MSELoss()
        self.l1reg_loss = L1regularization(weight_decay=0.1)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=5e-4,
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

        total_loss = 0.0
        total_rec_loss = 0.0
        print(f"Epoch {epoch+1:2d} / {self.args.epoch}")

        for batch_idx, (mod1_seq, _) in enumerate(self.train_loader):

            mod1_seq = mod1_seq.to(self.device).float()
            if self.args.arch == "rec":
                mod1_rec = self.model(mod1_seq)
                rec_loss = self.mse_loss(mod1_rec, mod1_seq)

            elif self.args.arch == "peakrec":
                inference_outputs = self.model.inference(mod1_seq)
                generative_outputs = self.model.generative(inference_outputs)
                output_loss = self.model.loss(
                    mod1_seq, inference_outputs, generative_outputs
                )
                peakvi_loss = output_loss["loss"]
                reconst_loss = output_loss["rl"]
                kl_local = output_loss["kld"]
                rec_loss = self.mse_loss(generative_outputs["p"], mod1_seq)

            elif self.args.arch == "scvirec":
                inference_outputs = self.model.inference(mod1_seq)
                generative_outputs = self.model.generative(inference_outputs)
                output_loss = self.model.loss(
                    mod1_seq, inference_outputs, generative_outputs
                )

                scvi_loss = output_loss["loss"]
                reconst_loss = output_loss["reconst_loss"]
                kl_local = output_loss["kl_local"]["kl_divergence_z"]
                kl_global = output_loss["kl_global"]
                rec_loss = self.mse_loss(generative_outputs["px_rate"], mod1_seq)

            l1reg_loss = self.l1reg_loss(self.model) * self.args.reg_loss_weight

            if self.args.arch == "rec":
                loss = rec_loss + l1reg_loss
            elif self.args.arch == "peakrec":
                loss = peakvi_loss + l1reg_loss
            elif self.args.arch == "scvirec":
                loss = scvi_loss + l1reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_rec_loss += rec_loss.item()

            print(
                f"Epoch {epoch+1:2d} [{batch_idx+1:2d} /{len(self.train_loader):2d}] | "
                + f"Total: {total_loss / (batch_idx + 1):.4f} | "
                + f"Rec: {rec_loss.item():.4f} | "
                + f"KL: {torch.mean(kl_local).item():3.4f} | "
                + f"L1 Reg: {l1reg_loss.item():.4f}"
            )

        train_rmse = np.sqrt(total_rec_loss / len(self.train_loader))
        self.writer.add_scalar("train_rmse", train_rmse, epoch)
        self.writer.add_scalar("rec_loss", rec_loss.item(), epoch)
        print(
            f"Epoch {epoch+1:3d} / {self.args.epoch} | Train RMSE: {train_rmse:.4f}",
            end=" ",
        )
        self.eval_epoch(epoch)

        # save checkpoint
        if not self.args.dryrun:
            filename = f"../../weights/model_{self.args.exp_name}.pt"
            print(f"saving weight to {filename} ...")
            torch.save(self.model.state_dict(), filename)

    def eval_epoch(self, epoch):
        """ eval process of each epoch """
        self.model.eval()

        total_rec_loss = 0.0
        for _, (mod1_seq, _) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            if self.args.arch == "rec":
                mod1_rec = self.model(mod1_seq)
                rec_loss = self.mse_loss(mod1_rec, mod1_seq)
            elif self.args.arch == "peakrec":
                inference_outputs = self.model.inference(mod1_seq)
                generative_outputs = self.model.generative(inference_outputs)
                rec_loss = self.mse_loss(generative_outputs["p"], mod1_seq)
            elif self.args.arch == "scvirec":
                inference_outputs = self.model.inference(mod1_seq)
                generative_outputs = self.model.generative(inference_outputs)
                rec_loss = self.mse_loss(generative_outputs["px_rate"], mod1_seq)

            total_rec_loss += rec_loss.item()
        test_rmse = np.sqrt(total_rec_loss / len(self.test_loader))
        print(f"| Eval RMSE: {test_rmse:.4f}")
        self.writer.add_scalar("test_rmse", test_rmse, epoch)

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
        use_numpy = True if self.args.mode in ["atac2gex", "gex2adt"] else False
        mod1_pred = np.zeros((1, self.args.mod1_dim)) if use_numpy else []

        for _, (mod1_seq, _) in enumerate(self.train_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            if self.args.arch == "rec":
                mod1_rec = self.model(mod1_seq)
            elif self.args.arch == "peakrec":
                inference_outputs = self.model.inference(mod1_seq)
                generative_outputs = self.model.generative(inference_outputs)
                mod1_rec = generative_outputs["p"]
            elif self.args.arch == "scvirec":
                inference_outputs = self.model.inference(mod1_seq)
                generative_outputs = self.model.generative(inference_outputs)
                mod1_rec = generative_outputs["px_rate"]

            if use_numpy:
                mod1_rec = mod1_rec.data.cpu().numpy()
                mod1_pred = np.vstack((mod1_pred, mod1_rec))
            else:
                mod1_pred.append(mod1_rec)

        if use_numpy:
            mod1_pred = mod1_pred[
                1:,
            ]
        else:
            mod1_pred = torch.cat(mod1_pred).detach().cpu().numpy()

        mod1_pred = csc_matrix(mod1_pred)

        mod1_sol = ad.read_h5ad(DATASET[self.args.mode]["train_mod1"]).X
        rmse_pred = rmse(mod1_sol, mod1_pred)
        logging.info(f"Train RMSE: {rmse_pred:5f}")

        # test set rmse
        mod1_pred = []
        for _, (mod1_seq, _) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()

            if self.args.arch == "rec":
                mod1_rec = self.model(mod1_seq)
            elif self.args.arch == "peakrec":
                inference_outputs = self.model.inference(mod1_seq)
                generative_outputs = self.model.generative(inference_outputs)
                mod1_rec = generative_outputs["p"]
            elif self.args.arch == "scvirec":
                inference_outputs = self.model.inference(mod1_seq)
                generative_outputs = self.model.generative(inference_outputs)
                mod1_rec = generative_outputs["px_rate"]

            mod1_pred.append(mod1_rec)

        mod1_pred = torch.cat(mod1_pred).detach().cpu().numpy()
        mod1_pred = csc_matrix(mod1_pred)

        mod1_sol = ad.read_h5ad(DATASET[self.args.mode]["test_mod1"]).X
        rmse_pred = rmse(mod1_sol, mod1_pred)
        logging.info(f"Eval RMSE: {rmse_pred:5f}")
