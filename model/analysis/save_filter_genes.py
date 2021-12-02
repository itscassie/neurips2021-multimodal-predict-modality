""" saved filtered features using the scanpy package """
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

# phase 1 v2 dataset
V2_DIR = "../../output/datasets_phase1v2/predict_modality"
CITE_V2_PTH = "openproblems_bmmc_cite_phase1v2"
MULTIOME_V2_PTH = "openproblems_bmmc_multiome_phase1v2"

ADT2GEX_V2_PTH = "openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_"
GEX2ADT_V2_PTH = "openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_"
ATAC2GEX_V2_PTH = "openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_"
GEX2ATAC_V2_PTH = "openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_"

ADT2GEX_V2 = {
    f"{V2_DIR}/{CITE_V2_PTH}_mod2/{ADT2GEX_V2_PTH}train_mod1.h5ad",
    f"{V2_DIR}/{CITE_V2_PTH}_mod2/{ADT2GEX_V2_PTH}train_mod2.h5ad",
    f"{V2_DIR}/{CITE_V2_PTH}_mod2/{ADT2GEX_V2_PTH}test_mod1.h5ad",
    f"{V2_DIR}/{CITE_V2_PTH}_mod2/{ADT2GEX_V2_PTH}test_mod2.h5ad",
}

GEX2ADT_V2 = {
    f"{V2_DIR}/{CITE_V2_PTH}_rna/{GEX2ADT_V2_PTH}train_mod1.h5ad",
    f"{V2_DIR}/{CITE_V2_PTH}_rna/{GEX2ADT_V2_PTH}train_mod2.h5ad",
    f"{V2_DIR}/{CITE_V2_PTH}_rna/{GEX2ADT_V2_PTH}test_mod1.h5ad",
    f"{V2_DIR}/{CITE_V2_PTH}_rna/{GEX2ADT_V2_PTH}test_mod2.h5ad",
}

ATAC2GEX_V2 = {
    f"{V2_DIR}/{MULTIOME_V2_PTH}_mod2/{ATAC2GEX_V2_PTH}train_mod1.h5ad",
    f"{V2_DIR}/{MULTIOME_V2_PTH}_mod2/{ATAC2GEX_V2_PTH}train_mod2.h5ad",
    f"{V2_DIR}/{MULTIOME_V2_PTH}_mod2/{ATAC2GEX_V2_PTH}test_mod1.h5ad",
    f"{V2_DIR}/{MULTIOME_V2_PTH}_mod2/{ATAC2GEX_V2_PTH}test_mod2.h5ad",
}

GEX2ATAC_V2 = {
    f"{V2_DIR}/{MULTIOME_V2_PTH}_rna/{GEX2ATAC_V2_PTH}train_mod1.h5ad",
    f"{V2_DIR}/{MULTIOME_V2_PTH}_rna/{GEX2ATAC_V2_PTH}train_mod2.h5ad",
    f"{V2_DIR}/{MULTIOME_V2_PTH}_rna/{GEX2ATAC_V2_PTH}test_mod1.h5ad",
    f"{V2_DIR}/{MULTIOME_V2_PTH}_rna/{GEX2ATAC_V2_PTH}test_mod2.h5ad",
}

# config
DATAPTH = [ATAC2GEX_V2]  # choices: ATAC2GEX, ADT2GEX, GEX2ADT, GEX2ATAC
PCT = 2  # percentage of filter gene
"""
ATAC2GEX
mod 1 (17394, 116490)
{pct: 5, min_cells: 869, feature_num: 13680}
{pct: 2, min_cells: 347, feature_num: 29917}

GEX2ADT
mod 1 (30077, 13953)
{pct: 5, min_cells: 1503, feature_num: 7470}
{pct: 2, min_cells: 601, feature_num: 10370}

ATAC2GEX_v2
mod 1 (31329, 116490)
{pct: 2, min_cells: 626, feature_num: 33354}

"""
for (i, mode) in enumerate(DATAPTH):
    print(f"DIRECTION [{i + 1} / {len(DATAPTH)}]")
    train_mod1_pth = mode[0]
    train_mod2_pth = mode[1]
    test_mod1_pth = mode[2]

    train_mod1 = ad.read_h5ad(train_mod1_pth)
    train_mod2 = ad.read_h5ad(train_mod2_pth)
    test_mod1 = ad.read_h5ad(test_mod1_pth)

    X_raw = ad.concat(
        {"train": train_mod1, "test": test_mod1},
        axis=0,
        join="outer",
        label="group",
        fill_value=0,
        index_unique="-",
    )
    min_cells = int(X_raw.X.shape[0] * 0.01 * PCT)
    print(f"min cells: {min_cells}")
    print(X_raw.shape)

    sc.pp.filter_genes(X_raw, min_cells=min_cells)

    train_5pct = X_raw[: train_mod1.X.shape[0], :]
    train_5pct = ad.AnnData(
        X=train_5pct.X,
        obs=train_5pct.obs,
        var=pd.DataFrame({"feature_types": train_mod1.var["feature_types"][X_raw.var_names]}),
        uns=train_5pct.uns,
        layers=train_5pct.layers,
    )

    test_5pct = X_raw[train_mod1.X.shape[0] :, :]
    test_5pct = ad.AnnData(
        X=test_5pct.X,
        obs=test_5pct.obs,
        var=pd.DataFrame({"feature_types": test_mod1.var["feature_types"][X_raw.var_names]}),
        uns=test_5pct.uns,
        layers=test_5pct.layers,
    )
    print(train_5pct)
    print(test_5pct)

    mod1_vars = np.array(train_mod1.var_names)
    mod1_5pct_idx = [
        int(np.where(mod1_vars == np.array(X_raw.var_names[i]))[0])
        for i in range(np.array(X_raw.var_names).shape[0])
    ]
    mod1 = str(train_mod1.var['feature_types'][0]).lower()
    mod2 = str(train_mod2.var['feature_types'][0]).lower()
    file_path = f"../../../indexs/{mod1}2{mod2}"
    os.makedirs(file_path, exist_ok=True)
    index_file = open(f"{file_path}/index_{PCT}pct_v2.txt", "w")
    index_file.write(f"index num: {len(mod1_5pct_idx)}\n")
    for ind in mod1_5pct_idx:
        index_file.write(str(ind) + "\n")

    print(f"finish saving {file_path}/index_{PCT}pct_v2.txt")
