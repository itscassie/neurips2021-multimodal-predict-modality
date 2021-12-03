"""
trainer of cycle architecture
works well on adt2gex, atac2gex, gex2atac subtasks
used the cycle consistancy loss the enhance the reconstruction effect
"""
import os
import logging
import numpy as np
import anndata as ad
from scipy.sparse import csc_matrix
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from opts import DATASET
from utils.metric import rmse
from utils.dataloader import SeqDataset
from utils.loss import L1regularization
from modules.model_ae import AutoEncoder


class TrainProcess:
    """the training process for cycle arch"""

    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter(log_dir=f"../runs/{args.exp_name}")
        self.device = (
            torch.device(f"cuda:{args.gpu_ids[0]}") if args.gpu_ids else torch.device("cpu")
        )

        mod1_idf = np.load(args.idf_path) if args.tfidf != 0 else None
        self.trainset = SeqDataset(
            DATASET[args.mode]["train_mod1"],
            DATASET[args.mode]["train_mod2"],
            mod1_idx_path=args.mod1_idx_path,
            tfidf=args.tfidf,
            mod1_idf=mod1_idf,
            batch_list=args.train_batch,
            norm=args.norm,
            gene_activity=args.gene_activity,
        )
        logging.info(f"TRAIN_NUM: {len(self.trainset):5d}")

        # do random split of dataset in phase 2
        if args.mode not in ["gex2atac_p2", "gex2adt_p2", "adt2gex_p2", "atac2gex_p2"]:
            self.testset = SeqDataset(
                DATASET[args.mode]["test_mod1"],
                DATASET[args.mode]["test_mod2"],
                mod1_idx_path=args.mod1_idx_path,
                tfidf=args.tfidf,
                mod1_idf=mod1_idf,
                batch_list=args.test_batch,
                norm=args.norm,
                gene_activity=args.gene_activity,
            )
            logging.info(f"TEST_NUM : {len(self.testset):5d}")
        else:
            train_size = int(0.9 * len(self.trainset))
            test_size = len(self.trainset) - train_size
            logging.info(f"(SPLITED) TRAIN_SIZE: {train_size:5d}, TEST_SIZE: {test_size:5d}")
            self.trainset, self.testset = random_split(
                self.trainset,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(args.seed),
            )

        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)

        self.model_AtoB = (
            AutoEncoder(
                input_dim=args.mod1_dim,
                out_dim=args.mod2_dim,
                feat_dim=args.emb_dim,
                hidden_dim=args.hid_dim,
                dropout=args.dropout,
            )
            .to(self.device)
            .float()
        )

        self.model_BtoA = (
            AutoEncoder(
                input_dim=args.mod2_dim,
                out_dim=args.mod1_dim,
                feat_dim=args.emb_dim,
                hidden_dim=args.hid_dim,
                dropout=args.dropout,
            )
            .to(self.device)
            .float()
        )

        logging.info(self.model_AtoB)
        logging.info(self.model_BtoA)

        self.mse_loss = nn.MSELoss()
        self.adv_loss = nn.BCELoss()
        self.l1reg_loss = L1regularization(weight_decay=0.1)
        self.eval_best = float("inf")

        self.optimizer = optim.SGD(
            [{"params": self.model_AtoB.parameters()}, {"params": self.model_BtoA.parameters()}],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=5e-4,
        )

        step_size = self.args.lr_decay_epoch
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[step_size, step_size * 2, step_size * 3],
            gamma=0.5,
            verbose=True,
        )

    def load_checkpoint(self):
        """load pre-trained model checkpoint"""
        if self.args.checkpoint is not None:
            if os.path.isfile(self.args.checkpoint):
                logging.info(f"loading checkpoint: {self.args.checkpoint}")
                checkpoint = torch.load(self.args.checkpoint)
                self.model_AtoB.load_state_dict(checkpoint["AtoB_state_dict"])
                self.model_BtoA.load_state_dict(checkpoint["BtoA_state_dict"])
            else:
                logging.info(f"no resume checkpoint found at {self.args.checkpoint}")

    def train_epoch(self, epoch):
        """training process of each epoch"""
        self.model_AtoB.train()
        self.model_BtoA.train()

        total_rec_loss = 0.0
        total_cycle_loss = 0.0
        total_rec_loss_B = 0.0
        print(f"Epoch {epoch+1:2d} / {self.args.epoch}")

        for batch_idx, (mod1_seq, mod2_seq) in enumerate(self.train_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()

            # (1) A to B to A:

            # (1-1) Reconstruction phase
            mod2_rec = self.model_AtoB(mod1_seq)
            mod1_rec = self.model_BtoA(mod2_rec)

            # (1-2) Rec loss
            rec_loss_B = self.mse_loss(mod2_rec, mod2_seq)
            # (1-3) Cycle loss
            cycle_loss_A = self.mse_loss(mod1_rec, mod1_seq)
            # l1 regularization
            l1reg_loss = (
                self.l1reg_loss(self.model_AtoB) + self.l1reg_loss(self.model_BtoA)
            ) * self.args.reg_loss_weight

            # ABA total loss
            ABA_loss = (rec_loss_B + l1reg_loss) * 10 + cycle_loss_A

            ABA_loss.backward()
            self.optimizer.step()

            # (2) B to A to B:
            # (2-1) Reconstruction phase
            mod1_rec2 = self.model_BtoA(mod2_seq)
            mod2_rec2 = self.model_AtoB(mod1_rec2)

            # (1-2) Rec loss
            rec_loss_A = self.mse_loss(mod1_rec2, mod1_seq)
            # (1-3) Cycle loss
            cycle_loss_B = self.mse_loss(mod2_rec2, mod2_seq)
            # l1 regularization
            l1reg_loss = (
                self.l1reg_loss(self.model_AtoB) + self.l1reg_loss(self.model_BtoA)
            ) * self.args.reg_loss_weight

            # BAB total loss
            BAB_loss = rec_loss_A + l1reg_loss + cycle_loss_B

            self.optimizer.zero_grad()
            BAB_loss.backward()
            self.optimizer.step()

            rec_loss = rec_loss_A + rec_loss_B
            cycle_loss = cycle_loss_A + cycle_loss_B
            total_rec_loss += rec_loss.item()
            total_cycle_loss += cycle_loss.item()
            total_rec_loss_B += rec_loss_B.item()

            print(
                f"Epoch {epoch+1:2d} [{batch_idx+1:2d} /{len(self.train_loader):2d}] | "
                + f"Mod2: {rec_loss_B.item():.4f} | "
                + f"Rec: {total_rec_loss / (batch_idx + 1):.4f} | "
                + f"Cyc: {total_cycle_loss / (batch_idx + 1):.4f} | "
                + f"ABA: {ABA_loss.item() :.4f} | "
                + f"BAB: {BAB_loss.item() :.4f} | "
                + f"L1: {l1reg_loss.item():.4f}"
            )
        self.scheduler.step()

        train_rmse = np.sqrt(total_rec_loss_B / len(self.train_loader))
        test_rmse = self.eval_epoch()
        (self.eval_best, save_best) = (
            (test_rmse, True) if test_rmse < self.eval_best else (self.eval_best, False)
        )

        logging.info(
            f"Epoch {epoch+1:3d} / {self.args.epoch} | Train RMSE: {train_rmse:.4f} "
            + f"| Eval RMSE: {test_rmse:.4f} | Eval best: {self.eval_best:.4f}"
        )

        self.writer.add_scalar("train_rmse", train_rmse, epoch)
        self.writer.add_scalar("rec_loss", rec_loss_B.item(), epoch)
        self.writer.add_scalar("test_rmse", test_rmse, epoch)

        # save checkpoint
        if not self.args.dryrun:
            filename = f"../weights/model_{self.args.exp_name}.pt"
            print(f"saving weight to {filename} ...")
            torch.save(
                {
                    "epoch": epoch,
                    "AtoB_state_dict": self.model_AtoB.state_dict(),
                    "BtoA_state_dict": self.model_BtoA.state_dict(),
                },
                filename,
            )

            # A to B
            filenameAtoB = f"../weights/model_AtoB_{self.args.exp_name}.pt"
            print(f"saving AtoB weight to {filenameAtoB} ...")
            torch.save(self.model_AtoB.state_dict(), filenameAtoB)

            if save_best and epoch > self.args.save_best_from:
                filename = f"../weights/model_best_{self.args.exp_name}.pt"
                print(f"saving best weight to {filename} ...")
                torch.save(
                    {
                        "epoch": epoch,
                        "AtoB_state_dict": self.model_AtoB.state_dict(),
                        "BtoA_state_dict": self.model_BtoA.state_dict(),
                    },
                    filename,
                )

                # A to B
                filenameAtoB = f"../weights/model_best_AtoB_{self.args.exp_name}.pt"
                print(f"saving best AtoB weight to {filenameAtoB} ...")
                torch.save(self.model_AtoB.state_dict(), filenameAtoB)

    def eval_epoch(self):
        """eval process of each epoch"""
        self.model_AtoB.eval()

        total_rec_loss = 0.0
        for _, (mod1_seq, mod2_seq) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_seq = mod2_seq.to(self.device).float()
            mod2_rec = self.model_AtoB(mod1_seq)
            rec_loss = self.mse_loss(mod2_rec, mod2_seq)
            total_rec_loss += rec_loss.item()
        test_rmse = np.sqrt(total_rec_loss / len(self.test_loader))

        return test_rmse

    def run(self):
        """run the whole training process"""
        self.load_checkpoint()
        print("start training ...")
        for epoch in range(self.args.epoch):
            self.train_epoch(epoch)

    def eval(self):
        """eval the trained model on train / test set"""
        print("start eval...")
        self.model_AtoB.eval()
        self.model_BtoA.eval()

        logging.info(f"Mode: {self.args.mode}")

        # train set rmse
        if self.args.mode in ["gex2atac", "gex2adt", "adt2gex"]:
            use_numpy = False
            mod2_pred = []
        else:
            use_numpy = True
            mod2_pred = np.zeros((1, self.args.mod2_dim))

        for _, (mod1_seq, _) in enumerate(self.train_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_rec = self.model_AtoB(mod1_seq)

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

        mod2_data = ad.read_h5ad(DATASET[self.args.mode]["train_mod2"])
        if len(self.args.train_batch) != 0:
            batch_data = mod2_data[mod2_data.obs["batch"] == "s0"]  # empty anndata
            for batch in self.args.train_batch:
                batch_data = ad.concat(
                    (batch_data, mod2_data[mod2_data.obs["batch"] == batch]),
                    axis=0,
                    join="outer",
                    index_unique="-",
                )
            mod2_data = batch_data
        mod2_sol = mod2_data.X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"Train RMSE: {rmse_pred:5f}")

        # test set rmse
        mod2_pred = []
        for _, (mod1_seq, _) in enumerate(self.test_loader):
            mod1_seq = mod1_seq.to(self.device).float()
            mod2_rec = self.model_AtoB(mod1_seq)
            mod2_pred.append(mod2_rec)

        mod2_pred = torch.cat(mod2_pred).detach().cpu().numpy()
        mod2_pred = csc_matrix(mod2_pred)

        mod2_sol = ad.read_h5ad(DATASET[self.args.mode]["test_mod2"]).X
        rmse_pred = rmse(mod2_sol, mod2_pred)
        logging.info(f"Eval RMSE: {rmse_pred:5f}")

    def save_AtoB(self):
        """ save only one way model (A to B) from the whole cycle model """
        checkpoint_name = (
            self.args.checkpoint.replace("../", "").replace("weights/", "").replace("model_", "")
        )
        filename = f"../weights/model_AtoB_{checkpoint_name}"
        print(f"saving AtoB weight to {filename} ...")
        torch.save(self.model_AtoB.state_dict(), filename)
