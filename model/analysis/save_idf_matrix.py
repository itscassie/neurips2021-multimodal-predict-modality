""" this function save idf matrixs from the dataset """
import os
import numpy as np
import anndata as ad

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

# phase 2 dataset
P2_DIR = "../../output/datasets_phase2/predict_modality"
CITE_P2_PTH = "openproblems_bmmc_cite_phase2"
MULTIOME_P2_PTH = "openproblems_bmmc_multiome_phase2"

ADT2GEX_P2_PTH = "openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_"
GEX2ADT_P2_PTH = "openproblems_bmmc_cite_phase2_rna.censor_dataset.output_"
ATAC2GEX_P2_PTH = "openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_"
GEX2ATAC_P2_PTH = "openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_"

ADT2GEX_P2 = [
    f"{P2_DIR}/{CITE_P2_PTH}_mod2/{ADT2GEX_P2_PTH}train_mod1.h5ad",
    f"{P2_DIR}/{CITE_P2_PTH}_mod2/{ADT2GEX_P2_PTH}train_mod2.h5ad",
]

GEX2ADT_P2 = [
    f"{P2_DIR}/{CITE_P2_PTH}_rna/{GEX2ADT_P2_PTH}train_mod1.h5ad",
    f"{P2_DIR}/{CITE_P2_PTH}_rna/{GEX2ADT_P2_PTH}train_mod2.h5ad",
]

ATAC2GEX_P2 = [
    f"{P2_DIR}/{MULTIOME_P2_PTH}_mod2/{ATAC2GEX_P2_PTH}train_mod1.h5ad",
    f"{P2_DIR}/{MULTIOME_P2_PTH}_mod2/{ATAC2GEX_P2_PTH}train_mod2.h5ad",
]

GEX2ATAC_P2 = [
    f"{P2_DIR}/{MULTIOME_P2_PTH}_rna/{GEX2ATAC_P2_PTH}train_mod1.h5ad",
    f"{P2_DIR}/{MULTIOME_P2_PTH}_rna/{GEX2ATAC_P2_PTH}train_mod2.h5ad",
]

# place the desired data path
DATAPTH = [ADT2GEX]


def idf_matrix(x_raw):
    """ returns idf matrix """
    x_idf = np.zeros_like(x_raw)
    x_idf[x_raw > 0] = 1
    idf = np.log(x_raw.shape[0] / (np.sum(x_idf, axis=0, keepdims=True) + 1))
    return idf


for (i, mode) in enumerate(DATAPTH):
    print(f"DIRECTION [{i + 1} / {len(DATAPTH)}]")

    train_mod1_pth = mode[0]
    train_mod2_pth = mode[1]
    # test_mod1_pth = mode[2]

    train_mod1 = ad.read_h5ad(train_mod1_pth)
    train_mod2 = ad.read_h5ad(train_mod2_pth)
    # test_mod1 = ad.read_h5ad(test_mod1_pth)

    # x_raw_matrix = np.concatenate(
    #     (train_mod1.layers["counts"].toarray(), test_mod1.layers["counts"].toarray()), axis=0
    # )
    x_raw_matrix = train_mod1.layers["counts"].toarray()
    print(x_raw_matrix.shape)

    x_idf_matrix = idf_matrix(x_raw_matrix)
    print(x_idf_matrix.shape)

    mod1 = str(train_mod1.var['feature_types'][0]).lower()
    mod2 = str(train_mod2.var['feature_types'][0]).lower()
    file_path = f"../../../idf_matrix/{mod1}2{mod2}"
    print(file_path)
    os.makedirs(file_path, exist_ok=True)
    np.save(f"{file_path}/mod1_idf_p2.npy", x_idf_matrix)
    print(f"finish saving {file_path}/mod1_idf_p2.npy")
