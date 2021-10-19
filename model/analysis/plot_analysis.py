import os
import sys
import logging
import anndata as ad
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

sys.path.append('../../model')
from modules.model_ae import AutoEncoder
from utils.dataloader import SeqDataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

ADT2GEX = [
'../../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
'../../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
'../../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
'../../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod2.h5ad'
]

GEX2ADT = [
'../../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad', 
'../../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad', 
'../../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
'../../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad'
]

ATAC2GEX = [
'../../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
'../../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
'../../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
'../../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
]

GEX2ATAC = [
'../../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad',
'../../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad',
'../../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
'../../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad' 
]

logging.basicConfig(level=logging.INFO)

# DATAPTH = [ADT2GEX, GEX2ADT, ATAC2GEX, GEX2ATAC]
DATAPTH = [GEX2ATAC]
for (i, mode) in enumerate(DATAPTH):
    print(f"DIRECTION [{i + 1} / {len(DATAPTH)}]")

    train_mod1_pth = mode[0]
    train_mod2_pth = mode[1]
    test_mod1_pth = mode[2]
    test_mod2_pth = mode[3]

    test_mod1 = ad.read_h5ad(test_mod1_pth)
    test_mod2 = ad.read_h5ad(test_mod2_pth)
    train_mod1 = ad.read_h5ad(train_mod1_pth)
    train_mod2 = ad.read_h5ad(train_mod2_pth)

    MOD1_DIM = int(test_mod1.X.shape[1])
    MOD2_DIM = int(test_mod2.X.shape[1])

    if train_mod2.var['feature_types'][0] == 'ATAC':
        logging.info("GEX to ATAC")
        LOAD_MODEL = MOD1_DIM == 13431 and MOD2_DIM == 10000
        
        FEAT_DIM = 128
        HIDDEN_DIM = 1000
        MODEL_PTH = '../weights/model_cycle_AtoB_gex2atac_e100.pt'


    elif train_mod2.var['feature_types'][0] == 'ADT':
        logging.info("GEX to ADT")
        LOAD_MODEL = MOD1_DIM == 13953 and MOD2_DIM == 134
        
        FEAT_DIM = 128
        HIDDEN_DIM = 1000
        MODEL_PTH = '../weights/model_nn_gex2adt_l1reg_e100.pt'

    elif train_mod1.var['feature_types'][0] == 'ADT':
        logging.info("ADT to GEX")
        LOAD_MODEL = MOD1_DIM == 134 and MOD2_DIM == 13953
        
        FEAT_DIM = 128
        HIDDEN_DIM = 1000
        MODEL_PTH = '../weights/model_cycle_AtoB_adt2gex_e250.pt'

    elif train_mod1.var['feature_types'][0] == 'ATAC':
        logging.info("ATAC to GEX")
        LOAD_MODEL = MOD1_DIM == 116490 and MOD2_DIM == 13431
        
        FEAT_DIM = 128
        HIDDEN_DIM = 1000
        MODEL_PTH = '../weights/model_cycle_AtoB_atac2gex_e250.pt'


    ### nn model inference
    testset = SeqDataset(test_mod1_pth)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    model_ae = AutoEncoder(input_dim=MOD1_DIM, out_dim=MOD2_DIM, feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM).float()

    model_ae.load_state_dict(torch.load(MODEL_PTH))
    model_ae.eval()

    mod2_matrix = np.zeros((1, MOD2_DIM))

    for batch_idx, (mod1_seq, mod2_seq) in enumerate(test_loader):
        mod1_seq = mod1_seq.float()
        mod2_rec = model_ae(mod1_seq)

        mod2_rec = mod2_rec.data.cpu().numpy()
        mod2_matrix = np.vstack((mod2_matrix, mod2_rec))

    y_nn_pred = mod2_matrix[1:,]


    ### apply PCA
    train = ad.concat(
            {"train": train_mod1, "test": test_mod1},
            axis=0,
            join="outer",
            label="group",
            fill_value=0,
            index_unique="-"
        )
    
    embedder_mod1 = TruncatedSVD(n_components=50)
    mod1_pca = embedder_mod1.fit_transform(train.X)

    embedder_mod2 = TruncatedSVD(n_components=50)
    mod2_pca = embedder_mod2.fit_transform(train_mod2.X)

    X_train = mod1_pca[train.obs['group'] == 'train']
    X_test = mod1_pca[train.obs['group'] == 'test']
    y_train = mod2_pca

    assert len(X_train) + len(X_test) == len(mod1_pca)

    reg = LinearRegression()

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    y_pca_pred = y_pred @ embedder_mod2.components_

    ### plt analysis
    mode = f"{train_mod1.var['feature_types'][0]}2{train_mod2.var['feature_types'][0]}"
    fdir = os.makedirs(f'figures/{mode}', exist_ok=True)

    # GT data
    logging.info("plt GT dist")
    
    plt.figure(figsize=(8,8))
    plt.hist(test_mod2.X.toarray().flatten(), bins=100)
    plt.xlim(-0.1, 1.1)
    plt.title(f'{mode} GT Data Distribution')
    plt.savefig(f'figures/{mode}/{mode}_GT_Dist.png')
    plt.close()

    # nn pred
    logging.info("plt NN results")
    
    plt.figure(figsize=(8,8))
    plt.hist(y_nn_pred.flatten(), bins=100)
    plt.xlim(-0.1, 1.1)
    plt.title(f'{mode} NN Pred Distribution')
    plt.savefig(f'figures/{mode}/{mode}_NN_Dist.png')
    plt.close()

    nn_error = y_nn_pred - test_mod2.X.toarray()
    plt.figure(figsize=(8,8))
    plt.scatter(test_mod2.X.toarray().flatten(), nn_error.flatten(), s=1)
    plt.title(f'{mode} NN Pred GT-Error Plot')
    plt.xlabel('Mod2 Ground Truth')
    plt.ylabel('Error')
    plt.savefig(f'figures/{mode}/{mode}_NN_GT_Error.png')
    plt.close()


    ## pca pred
    logging.info("plt PCA results")

    plt.figure(figsize=(8,8))
    plt.hist(y_pca_pred.flatten(), bins=100)
    plt.xlim(-0.1, 1.1)
    plt.title(f'{mode} PCA Pred Distribution')
    plt.savefig(f'figures/{mode}/{mode}_PCA_Dist.png')
    plt.close()

    pca_error = y_pca_pred - test_mod2.X.toarray()
    plt.figure(figsize=(8,8))
    plt.scatter(test_mod2.X.toarray().flatten(), pca_error.flatten(), s=1)
    plt.title(f'{mode} PCA Pred GT-Error Plot')
    plt.xlabel('Mod2 Ground Truth')
    plt.ylabel('Error')
    plt.savefig(f'figures/{mode}/{mode}_PCA_GT_Error.png')
    plt.close()








