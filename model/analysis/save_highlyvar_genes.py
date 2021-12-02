""" save highly variable using scanpy package """
import os
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd

# phase 1 v1 dataset
V1_DIR = "../../output/datasets/predict_modality"
CITE_PTH = "openproblems_bmmc_cite_phase1"
MULTIOME_PTH = "openproblems_bmmc_multiome_phase1"

ADT2GEX_PTH = "openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_"
GEX2ADT_PTH = "openproblems_bmmc_cite_phase1_rna.censor_dataset.output_"
ATAC2GEX_PTH = "openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_"
GEX2ATAC_PTH = "openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_"

ADT2GEX = [
    f"{V1_DIR}/{CITE_PTH}_mod2/{ADT2GEX_PTH}train_mod1.h5ad",
    f"{V1_DIR}/{CITE_PTH}_mod2/{ADT2GEX_PTH}train_mod2.h5ad",
    f"{V1_DIR}/{CITE_PTH}_mod2/{ADT2GEX_PTH}test_mod1.h5ad",
    f"{V1_DIR}/{CITE_PTH}_mod2/{ADT2GEX_PTH}test_mod2.h5ad",
]

GEX2ADT = [
    f"{V1_DIR}/{CITE_PTH}_rna/{GEX2ADT_PTH}train_mod1.h5ad",
    f"{V1_DIR}/{CITE_PTH}_rna/{GEX2ADT_PTH}train_mod2.h5ad",
    f"{V1_DIR}/{CITE_PTH}_rna/{GEX2ADT_PTH}test_mod1.h5ad",
    f"{V1_DIR}/{CITE_PTH}_rna/{GEX2ADT_PTH}test_mod2.h5ad",
]

ATAC2GEX = [
    f"{V1_DIR}/{MULTIOME_PTH}_mod2/{ATAC2GEX_PTH}train_mod1.h5ad",
    f"{V1_DIR}/{MULTIOME_PTH}_mod2/{ATAC2GEX_PTH}train_mod2.h5ad",
    f"{V1_DIR}/{MULTIOME_PTH}_mod2/{ATAC2GEX_PTH}test_mod1.h5ad",
    f"{V1_DIR}/{MULTIOME_PTH}_mod2/{ATAC2GEX_PTH}test_mod2.h5ad",
]

GEX2ATAC = [
    f"{V1_DIR}/{MULTIOME_PTH}_rna/{GEX2ATAC_PTH}train_mod1.h5ad",
    f"{V1_DIR}/{MULTIOME_PTH}_rna/{GEX2ATAC_PTH}train_mod2.h5ad",
    f"{V1_DIR}/{MULTIOME_PTH}_rna/{GEX2ATAC_PTH}test_mod1.h5ad",
    f"{V1_DIR}/{MULTIOME_PTH}_rna/{GEX2ATAC_PTH}test_mod2.h5ad",
]

# config
DATAPTH = [GEX2ADT]  # choices: ATAC2GEX, ADT2GEX, GEX2ADT, GEX2ATAC
NTOP = 10000  # n_top_genes

for (i, mode) in enumerate(DATAPTH):
    print(f"DIRECTION [{i + 1} / {len(DATAPTH)}]")
    train_mod1_pth = mode[0]
    train_mod2_pth = mode[1]
    test_mod1_pth = mode[2]

    train_mod1 = sc.read_h5ad(train_mod1_pth)
    train_mod2 = sc.read_h5ad(train_mod2_pth)
    test_mod1 = sc.read_h5ad(test_mod1_pth)

    X_raw = sc.concat(
        {"train": train_mod1, "test": test_mod1},
        axis=0,
        join="outer",
        label="group",
        fill_value=0,
        index_unique="-",
    )
    print(X_raw.shape)

    sc.pp.highly_variable_genes(X_raw, n_top_genes=NTOP)
    X_raw = X_raw[:, X_raw.var.highly_variable]

    train_highly = X_raw[: train_mod1.X.shape[0], :]
    train_highly = ad.AnnData(
        X=train_highly.X,
        obs=train_highly.obs,
        var=pd.DataFrame({"feature_types": train_mod1.var["feature_types"][X_raw.var_names]}),
        uns=train_highly.uns,
        layers=train_highly.layers,
    )

    test_highly = X_raw[train_mod1.X.shape[0] :, :]
    test_highly = ad.AnnData(
        X=test_highly.X,
        obs=test_highly.obs,
        var=pd.DataFrame({"feature_types": test_mod1.var["feature_types"][X_raw.var_names]}),
        uns=test_highly.uns,
        layers=test_highly.layers,
    )
    print(train_highly)
    print(test_highly)

    mod1_vars = np.array(train_mod1.var_names)
    mod1_highly_idx = [
        int(np.where(mod1_vars == np.array(X_raw.var_names[i]))[0])
        for i in range(np.array(X_raw.var_names).shape[0])
    ]

    mod1 = str(train_mod1.var['feature_types'][0]).lower()
    mod2 = str(train_mod2.var['feature_types'][0]).lower()
    file_path = f"../../../indexs/{mod1}2{mod2}"
    os.makedirs(file_path, exist_ok=True)
    index_file = open(f"{file_path}/index_highly{NTOP}.txt", "w")
    index_file.write(f"index num: {len(mod1_highly_idx)}\n")
    for ind in mod1_highly_idx:
        index_file.write(str(ind) + "\n")

    print(f"finish saving {file_path}/index_highly{NTOP}.txt")
