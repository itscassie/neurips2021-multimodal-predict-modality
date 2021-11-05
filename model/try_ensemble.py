import time
import numbers
import argparse
import numpy as np
import anndata as ad
from datetime import datetime
from scipy.sparse import csc_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from modules.model_ae import AutoEncoder
from utils.dataloader import SeqDataset
from opts import model_opts as opt


from torchensemble.fusion import FusionRegressor
from torchensemble.voting import VotingRegressor
from torchensemble.bagging import BaggingRegressor
from torchensemble.gradient_boosting import GradientBoostingRegressor
from torchensemble.snapshot_ensemble import SnapshotEnsembleRegressor

from torchensemble.utils.logging import set_logger

"""
parser = argparse.ArgumentParser()
opt(parser)
args = parser.parse_args()
"""

ADT2GEX = [
'../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
'../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
'../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
'../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod2.h5ad'
]

GEX2ADT = [
'../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad', 
'../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad', 
'../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
'../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad'
]

ATAC2GEX = [
'../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
'../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
'../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
'../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
]

GEX2ATAC = [
'../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad',
'../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad',
'../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
'../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad' 
]

# DATAPTH = [ATAC2GEX, ADT2GEX, GEX2ADT, GEX2ATAC]
mode = ADT2GEX
    
train_mod1_pth = mode[0]
train_mod2_pth = mode[1]
test_mod1_pth = mode[2]
test_mod2_pth = mode[3]

test_mod1 = ad.read_h5ad(test_mod1_pth)
train_mod1 = ad.read_h5ad(train_mod1_pth)
train_mod2 = ad.read_h5ad(train_mod2_pth)
test_mod2 = ad.read_h5ad(test_mod2_pth)

# change here 
print(f"MOD1: {train_mod1.var['feature_types'][0]}")
print(f"MOD2: {train_mod2.var['feature_types'][0]}")
print(f"MOD1_DIM: {train_mod1.X.shape[1]}, MOD2_DIM: {train_mod2.X.shape[1]}")
print(f"TRAIN_NUM: {train_mod1.X.shape[0]}, TEST_NUM: {test_mod1.X.shape[0]}")

assert int(train_mod1.X.shape[1]) == int(test_mod1.X.shape[1]), "mod1 feature # train != test"
assert int(train_mod2.X.shape[1]) == int(test_mod2.X.shape[1]), "mod1 feature # train != test"
assert int(train_mod1.X.shape[0]) == int(train_mod2.X.shape[0]), "train # mod1 != mod2"
assert int(test_mod1.X.shape[0]) == int(test_mod2.X.shape[0]), "train # mod1 != mod2"

MOD1_DIM = int(train_mod1.X.shape[1])
MOD2_DIM = int(train_mod2.X.shape[1])
TRAIN_NUM = int(train_mod1.X.shape[0])
TEST_NUM = int(test_mod1.X.shape[0])

FEAT_DIM = 128
HIDDEN_DIM = 1000
EPOCH = 50
BATCH_SIZE = 2048

trainset = SeqDataset(train_mod1_pth, train_mod2_pth)
testset = SeqDataset(test_mod1_pth, test_mod2_pth)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

model_ae = AutoEncoder(input_dim=MOD1_DIM, out_dim=MOD2_DIM, feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM).cuda().double()
print(model_ae)

recon_loss = nn.MSELoss()
optimizer = optim.SGD(model_ae.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def display_records(records, logger):
    msg = (
        "{:<28} | Testing MSE: {:.4f} | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )

    print("\n")
    for method, training_time, evaluating_time, mse in records:
        logger.info(msg.format(method, mse, training_time, evaluating_time))

n_estimators = 5
lr = 1e-3
weight_decay = 5e-4
epochs = 200

# Utils
batch_size = 2048
records = []
torch.manual_seed(0)

# Load data
logger = set_logger("regression")

# # FusionRegressor
# model = FusionRegressor(
#     estimator=model_ae, n_estimators=n_estimators, cuda=True
# )

# # Set the optimizer
# model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

# tic = time.time()
# model.fit(train_loader, epochs=epochs)
# toc = time.time()
# training_time = toc - tic

# tic = time.time()
# testing_mse = np.sqrt(model.evaluate(test_loader))
# toc = time.time()
# evaluating_time = toc - tic

# records.append(
#     ("FusionRegressor", training_time, evaluating_time, testing_mse)
# )

# # """
# # VotingRegressor
# model = VotingRegressor(
#     estimator=model_ae, n_estimators=n_estimators, cuda=True
# )

# # Set the optimizer
# model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

# tic = time.time()
# model.fit(train_loader, epochs=epochs)
# toc = time.time()
# training_time = toc - tic

# tic = time.time()
# testing_mse = np.sqrt(model.evaluate(test_loader))
# toc = time.time()
# evaluating_time = toc - tic

# records.append(
#     ("VotingRegressor", training_time, evaluating_time, testing_mse)
# )

# # BaggingRegressor
# model = BaggingRegressor(
#     estimator=model_ae, n_estimators=n_estimators, cuda=True
# )

# # Set the optimizer
# model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

# tic = time.time()
# model.fit(train_loader, epochs=epochs)
# toc = time.time()
# training_time = toc - tic

# tic = time.time()
# testing_mse = np.sqrt(model.evaluate(test_loader))
# toc = time.time()
# evaluating_time = toc - tic

# records.append(
#     ("BaggingRegressor", training_time, evaluating_time, testing_mse)
# )
"""
# GradientBoostingRegressor
model = GradientBoostingRegressor(
    estimator=model_ae, n_estimators=n_estimators, cuda=True
)

# Set the optimizer
model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

tic = time.time()
model.fit(train_loader, epochs=epochs)
toc = time.time()
training_time = toc - tic

tic = time.time()
testing_mse = np.sqrt(model.evaluate(test_loader))
toc = time.time()
evaluating_time = toc - tic

records.append(
    (
        "GradientBoostingRegressor",
        training_time,
        evaluating_time,
        testing_mse,
    )
)
"""
# SnapshotEnsembleRegressor
model = SnapshotEnsembleRegressor(
    estimator=model_ae, n_estimators=n_estimators, cuda=True
)

# Set the optimizer
model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

tic = time.time()
model.fit(train_loader, epochs=epochs)
toc = time.time()
training_time = toc - tic

tic = time.time()
testing_mse = np.sqrt(model.evaluate(test_loader))
toc = time.time()
evaluating_time = toc - tic

records.append(
    (
        "SnapshotEnsembleRegressor",
        training_time,
        evaluating_time,
        testing_mse,
    )
)
# """
display_records(records, logger)

"""
print("start training...")
print('start time: ', datetime.now().strftime('%H:%M:%S'))
for epoch in range(EPOCH):
    model_ae.train()
    running_loss = 0.0
    print(f'(Epoch {epoch+1:2d} / {EPOCH})')
    for batch_idx, (mod1_seq, mod2_seq) in enumerate(train_loader):

        mod1_seq = mod1_seq.cuda().float()
        mod2_seq = mod2_seq.cuda().float()
        mod2_rec = model_ae(mod1_seq)
        rec_loss = recon_loss(mod2_rec, mod2_seq)

        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()

        running_loss += rec_loss.item()

        print(f'Epoch {epoch+1:2d} [{batch_idx+1:2d} /{len(train_loader):2d}] | ' + \
            f'Total: {running_loss / (batch_idx + 1):.4f}')

# PATH = f"model_{train_mod1.var['feature_types'][0]}2{train_mod2.var['feature_types'][0]}.pt"
# print("saving weight to {PATH} ...")
# torch.save(model_ae.state_dict(), PATH)

print("start eval...")
model_ae.load_state_dict(torch.load(PATH))
model_ae.eval()

mod2_matrix = np.zeros((1, MOD2_DIM))

for batch_idx, (mod1_seq, mod2_seq) in enumerate(test_loader):
    mod1_seq = mod1_seq.cuda().float()
    mod2_rec = model_ae(mod1_seq)
    
    mod2_rec = mod2_rec.data.cpu().numpy()
    mod2_matrix = np.vstack((mod2_matrix, mod2_rec))

mod2_matrix = csc_matrix(mod2_matrix[1:,])
mod2_pred = ad.AnnData(X=mod2_matrix)

def rmse(ad_sol, ad_pred):
    tmp = ad_sol.X - ad_pred.X
    rmse = np.sqrt(tmp.power(2).mean())
    return rmse

print(f"{train_mod1.var['feature_types'][0]} 2 {train_mod2.var['feature_types'][0]} RMSE: {rmse(mod2_pred, test_mod2):5f}")

# """