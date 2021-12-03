""" prints basic data statistics """
from collections import Counter
import anndata as ad

# phase 1 v1 dataset
V1_DIR = "output/datasets/predict_modality"
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
V2_DIR = "output/datasets_phase1v2/predict_modality"
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
P2_DIR = "output/datasets_phase2/predict_modality"
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


def print_data_stats(datapath, verbose=False):
    """print basic data statistics"""
    for (i, mode) in enumerate(datapath):
        train_mod1_pth, train_mod2_pth, test_mod1_pth, test_mod2_pth = "", "", "", ""
        test_mod1, test_mod2 = None, None
        print(f"\nDIRECTION [{i + 1} / {len(datapath)}]")

        train_mod1_pth = mode[0]
        train_mod2_pth = mode[1]
        train_mod1 = ad.read_h5ad(train_mod1_pth)
        train_mod2 = ad.read_h5ad(train_mod2_pth)
        if verbose:
            print("train mod1:", train_mod1)
            print("train mod2:", train_mod2)

        try:
            test_mod1_pth = mode[2]
            test_mod2_pth = mode[3]
            test_mod1 = ad.read_h5ad(test_mod1_pth)
            test_mod2 = ad.read_h5ad(test_mod2_pth)
            if verbose:
                print("test mod1:", test_mod1)
                print("test mod2:", test_mod2)
        except:
            pass

        # change here
        print(f"MOD1: {train_mod1.var['feature_types'][0]}")
        print(f"MOD2: {train_mod2.var['feature_types'][0]}")
        print(f"MOD1_DIM: {train_mod1.X.shape[1]}, MOD2_DIM: {train_mod2.X.shape[1]}")
        print(f"TRAIN_NUM: {train_mod1.X.shape[0]}")
        print("TRAIN_BATCH: ", sorted(set(train_mod1.obs["batch"])))
        print("TRAIN STATS: ", sorted(Counter(train_mod1.obs["batch"]).items()))

        try:
            print(f"TEST_NUM: {test_mod1.X.shape[0]}")
            print("TEST_BATCH: ", sorted(set(test_mod1.obs["batch"])))
            print("TEST STATS: ", sorted(Counter(test_mod1.obs["batch"]).items()))
        except:
            pass


if __name__ == "__main__":
    # DATAPTH = [ADT2GEX, GEX2ADT, ATAC2GEX, GEX2ATAC]
    # DATAPTH = [ADT2GEX_V2, GEX2ADT_V2, ATAC2GEX_V2, GEX2ATAC_V2]
    DATAPTH = [ADT2GEX_P2, GEX2ADT_P2, ATAC2GEX_P2, GEX2ATAC_P2]
    print_data_stats(DATAPTH)
