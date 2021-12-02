# Dependencies:
# pip: scikit-learn, anndata, scanpy
#
# Python starter kit for the NeurIPS 2021 Single-Cell Competition.
# Parts with `TODO` are supposed to be changed by you.
#
# More documentation:
#
# https://viash.io/docs/creating_components/python/

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

logging.basicConfig(level=logging.INFO)

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
meta = {"resources_dir": "."}
par = {
    "input_train_mod1": "output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad",
    "input_train_mod2": "output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad",
    "input_test_mod1": "output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad",
    "distance_method": "minkowski",
    "output": "output.h5ad",
    "n_pcs": 50,
}
## VIASH END
sys.path.append(meta["resources_dir"])
from model.modules.model_ae import AutoEncoder
from model.utils.dataloader import SeqDataset

# Methods
method_id = "mse"

logging.info("Reading `h5ad` files...")
input_train_mod1 = ad.read_h5ad(par["input_train_mod1"])
input_train_mod2 = ad.read_h5ad(par["input_train_mod2"])
input_test_mod1 = ad.read_h5ad(par["input_test_mod1"])

# Check data shape
LOAD_MODEL = True
MOD1_DIM = int(input_train_mod1.X.shape[1])
MOD2_DIM = int(input_train_mod2.X.shape[1])

if input_train_mod2.var["feature_types"][0] == "ATAC":
    logging.info("GEX to ATAC")
    LOAD_MODEL = MOD1_DIM == 13431 and MOD2_DIM == 10000

    FEAT_DIM = 128
    HIDDEN_DIM = 1000
    MODEL_PTH = meta["resources_dir"] + "/model/weights/gex2atac_weight.pt"


elif input_train_mod2.var["feature_types"][0] == "ADT":
    logging.info("GEX to ADT")
    LOAD_MODEL = MOD1_DIM == 13953 and MOD2_DIM == 134

    FEAT_DIM = 50
    HIDDEN_DIM = 1000
    MODEL_PTH = meta["resources_dir"] + "/model/weights/gex2adt_weight.pt"

elif input_train_mod1.var["feature_types"][0] == "ADT":
    logging.info("ADT to GEX")
    LOAD_MODEL = MOD1_DIM == 134 and MOD2_DIM == 13953

    FEAT_DIM = 128
    HIDDEN_DIM = 1000
    MODEL_PTH = meta["resources_dir"] + "/model/weights/model_ADT2GEX.pt"

elif input_train_mod1.var["feature_types"][0] == "ATAC":
    logging.info("ATAC to GEX")
    LOAD_MODEL = MOD1_DIM == 116490 and MOD2_DIM == 13431

    FEAT_DIM = 128
    HIDDEN_DIM = 1000
    MODEL_PTH = meta["resources_dir"] + "/model/weights/model_ATAC2GEX.pt"


# Method: use pretrain model / use PCA
if LOAD_MODEL:
    logging.info("Use pretrain model...")
    logging.info(f"Model Path: {MODEL_PTH}")

    # Model inference
    testset = SeqDataset(par["input_test_mod1"])
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    model_ae = AutoEncoder(
        input_dim=MOD1_DIM, out_dim=MOD2_DIM, feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM
    ).float()
    logging.info(model_ae)

    model_ae.load_state_dict(torch.load(MODEL_PTH, map_location=torch.device("cpu")))
    model_ae.eval()

    mod2_matrix = np.zeros((1, MOD2_DIM))

    for batch_idx, (mod1_seq, mod2_seq) in enumerate(test_loader):
        mod1_seq = mod1_seq.float()
        mod2_rec = model_ae(mod1_seq)

        mod2_rec = mod2_rec.data.cpu().numpy()
        mod2_matrix = np.vstack((mod2_matrix, mod2_rec))

    y_pred = csc_matrix(
        mod2_matrix[
            1:,
        ]
    )
    logging.info("Finish Prediction")

else:
    logging.info("PCA...")

    # PCA methods
    input_train = ad.concat(
        {"train": input_train_mod1, "test": input_test_mod1},
        axis=0,
        join="outer",
        label="group",
        fill_value=0,
        index_unique="-",
    )
    # Do PCA on the input data
    logging.info("Performing dimensionality reduction on modality 1 values...")
    embedder_mod1 = TruncatedSVD(n_components=50)
    mod1_pca = embedder_mod1.fit_transform(input_train.X)

    logging.info("Performing dimensionality reduction on modality 2 values...")
    embedder_mod2 = TruncatedSVD(n_components=50)
    mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

    # split dimred back up
    X_train = mod1_pca[input_train.obs["group"] == "train"]
    X_test = mod1_pca[input_train.obs["group"] == "test"]
    y_train = mod2_pca

    assert len(X_train) + len(X_test) == len(mod1_pca)

    logging.info("Running Linear regression...")

    # KNN regressor later on.
    reg = LinearRegression()

    # Train the model on the PCA reduced modality 1 and 2 data
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Project the predictions back to the modality 2 feature space
    y_pred = y_pred @ embedder_mod2.components_

    # Store as sparse matrix to be efficient.
    y_pred = csc_matrix(y_pred)

# Saving data to anndata format
logging.info("Storing annotated data...")

adata = ad.AnnData(
    X=y_pred,
    obs=input_test_mod1.obs,
    var=input_train_mod2.var,
    uns={
        "dataset_id": input_train_mod1.uns["dataset_id"],
        "method_id": method_id,
    },
)
adata.write_h5ad(par["output"], compression="gzip")
