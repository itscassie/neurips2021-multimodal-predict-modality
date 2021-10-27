import os
import numpy as np
import anndata as ad
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

DATAPTH = [ADT2GEX, GEX2ADT, ATAC2GEX, GEX2ATAC,]

def idf_matrix(X_raw):
    X_idf = np.zeros_like(X_raw)
    X_idf[X_raw > 0] = 1
    idf = np.log(X_raw.shape[0] / (np.sum(X_idf, axis=0, keepdims=True) + 1))
    return idf

for (i, mode) in enumerate(DATAPTH):
    print(f"DIRECTION [{i + 1} / {len(DATAPTH)}]")
    
    train_mod1_pth = mode[0]
    train_mod2_pth = mode[1]
    test_mod1_pth = mode[2]

    train_mod1 = ad.read_h5ad(train_mod1_pth)
    train_mod2 = ad.read_h5ad(train_mod2_pth)
    test_mod1 = ad.read_h5ad(test_mod1_pth)
    
    X_raw = np.concatenate((train_mod1.layers["counts"].toarray(), test_mod1.layers["counts"].toarray()), axis=0)
    print(X_raw.shape)
    
    X_idf = idf_matrix(X_raw)
    print(X_idf.shape)
    
    file_path = f"../../../idf_matrix/{str(train_mod1.var['feature_types'][0]).lower()}2{str(train_mod2.var['feature_types'][0]).lower()}"
    print(file_path)
    os.makedirs(file_path, exist_ok=True)
    np.save(f'{file_path}/mod1_idf.npy', X_idf)
    print(f'finish saving {file_path}/mod1_idf.npy')
    