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
task = "atac2gex"
mode = {
    "gex2atac": "output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_",
    "gex2adt": "output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_",
    "adt2gex": "output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_",
    "atac2gex": "output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_",
    "sample": "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.",
}

par = {
    "input_train_mod1": mode[task] + "train_mod1.h5ad",
    "input_train_mod2": mode[task] + "train_mod2.h5ad",
    "input_test_mod1": mode[task] + "test_mod1.h5ad",
    # added for test, remove later
    # 'input_test_mod2': mode[task] + 'test_mod2.h5ad',
    "distance_method": "minkowski",
    "output": "output.h5ad",
    "n_pcs": 50,
}
## VIASH END
sys.path.append(meta["resources_dir"])
from model.modules.model_ae import AutoEncoder, BatchRemovalGAN
from model.utils.dataloader import SeqDataset

# load model
def pretrin_nn(
    test_mod1,
    model_pth,
    mod1_dim,
    mod2_dim,
    feat_dim,
    hidden_dim,
    mod1_idx_path=None,
    tfidf=0,
    idf_matrix=None,
    gene_activity=False,
    log=False,
):

    logging.info("Use pretrain model...")
    logging.info(f"Model Path: {model_pth}")

    # Dataset
    testset = SeqDataset(
        test_mod1,
        mod1_idx_path=mod1_idx_path,
        tfidf=tfidf,
        mod1_idf=idf_matrix,
        gene_activity=gene_activity,
    )
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)
    model_ae = AutoEncoder(
        input_dim=mod1_dim, out_dim=mod2_dim, feat_dim=feat_dim, hidden_dim=hidden_dim
    ).float()
    if log:
        logging.info(model_ae)

    # Load weight
    # model_ae.load_state_dict(torch.load(model_pth)) # gpu
    model_ae.load_state_dict(
        torch.load(model_pth, map_location=torch.device("cpu"))
    )  # cpu

    # Model inference
    model_ae.eval()
    mod2_matrix = np.zeros((1, mod2_dim))
    for batch_idx, (mod1_seq, mod2_seq) in enumerate(test_loader):
        mod1_seq = mod1_seq.float()
        mod2_rec = model_ae(mod1_seq)

        mod2_rec = mod2_rec.data.cpu().numpy()
        mod2_matrix = np.vstack((mod2_matrix, mod2_rec))

    y_pred = np.array(
        mod2_matrix[
            1:,
        ]
    )
    logging.info("Finish Prediction")

    return y_pred


# pretrain batch gan
def pretrin_batchgan(
    test_mod1,
    model_pth,
    mod1_dim,
    mod2_dim,
    feat_dim,
    hidden_dim,
    mod1_idx_path=None,
    tfidf=0,
    idf_matrix=None,
    cls_num=10,
):

    logging.info("Use pretrain model...")
    logging.info(f"Model Path: {model_pth}")

    # Dataset
    testset = SeqDataset(test_mod1)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)
    model_ae = BatchRemovalGAN(
        input_dim=mod1_dim,
        out_dim=mod2_dim,
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        cls_num=cls_num,
    ).float()
    # logging.info(model_ae)

    # Load weight
    # model_ae.load_state_dict(torch.load(model_pth)) # gpu
    model_ae.load_state_dict(
        torch.load(model_pth, map_location=torch.device("cpu"))
    )  # cpu

    # Model inference
    model_ae.eval()
    mod2_matrix = np.zeros((1, mod2_dim))

    for batch_idx, (mod1_seq, _) in enumerate(test_loader):
        mod1_seq = mod1_seq.float()
        mod2_rec, _ = model_ae(mod1_seq)

        mod2_rec = mod2_rec.data.cpu().numpy()
        mod2_matrix = np.vstack((mod2_matrix, mod2_rec))

    y_pred = np.array(
        mod2_matrix[
            1:,
        ]
    )
    logging.info("Finish Prediction")

    return y_pred


# do pca
def pca(input_train_mod1, input_test_mod1, n=50):

    logging.info("Use PCA...")

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
    embedder_mod1 = TruncatedSVD(n_components=n)
    mod1_pca = embedder_mod1.fit_transform(input_train.X)

    logging.info("Performing dimensionality reduction on modality 2 values...")
    embedder_mod2 = TruncatedSVD(n_components=n)
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
    logging.info("Finish Prediction")

    return np.array(y_pred)


# Methods
method_id = "mse_v4"

logging.info("Reading `h5ad` files...")
input_train_mod1 = ad.read_h5ad(par["input_train_mod1"])
input_train_mod2 = ad.read_h5ad(par["input_train_mod2"])
input_test_mod1 = ad.read_h5ad(par["input_test_mod1"])

# Check data shape
LOAD_MODEL = True
MOD1_DIM = int(input_train_mod1.X.shape[1])
MOD2_DIM = int(input_train_mod2.X.shape[1])
FEAT_DIM = 128
HIDDEN_DIM = 1000

if input_train_mod2.var["feature_types"][0] == "ATAC":
    logging.info("GEX to ATAC")
    LOAD_MODEL = MOD1_DIM == 13431 and MOD2_DIM == 10000

    if LOAD_MODEL:
        # model (1) pca
        y_pred_pca = pca(input_train_mod1, input_test_mod1, n=120)

        # model (2) nn
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/gex2atac/model_best_AtoB_cycle_gex2atac_v2_Nov17-23-18.pt"
        )
        y_pred_nn = pretrin_nn(
            par["input_test_mod1"], model_pth, MOD1_DIM, MOD2_DIM, FEAT_DIM, HIDDEN_DIM
        )

        # model (3) concat
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/gex2atac/model_best_AtoB_cycle_gex2atac_v2_tfidfconcat_Nov18-07-03.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/gex2atac/mod1_idf_v2.npy"
        )
        y_pred_concat = pretrin_nn(
            par["input_test_mod1"],
            model_pth,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # ensemble
        y_pred = (
            np.array(y_pred_pca) + np.array(y_pred_nn) + np.array(y_pred_concat)
        ) / 3

elif input_train_mod2.var["feature_types"][0] == "ADT":
    logging.info("GEX to ADT")
    LOAD_MODEL = MOD1_DIM == 13953 and MOD2_DIM == 134

    if LOAD_MODEL:

        # model (6) nn
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/gex2adt/model_best_nn_gex2adt_v2_Nov16-16-02.pt"
        )
        y_pred_nn = pretrin_nn(
            par["input_test_mod1"], model_pth, MOD1_DIM, MOD2_DIM, FEAT_DIM, HIDDEN_DIM
        )

        # model (7) nn 2
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/gex2adt/model_best_nn_gex2adt_v2_reg1_Nov18-21-24.pt"
        )
        y_pred_nn2 = pretrin_nn(
            par["input_test_mod1"], model_pth, MOD1_DIM, MOD2_DIM, FEAT_DIM, HIDDEN_DIM
        )

        # model (8) bgan
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/gex2adt/model_best_batchgan_gex2adt_v2_Nov16-15-47.pt"
        )
        y_pred_bgan = pretrin_batchgan(
            par["input_test_mod1"],
            model_pth,
            MOD1_DIM,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            cls_num=10,
        )

        # model (10) concat
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/gex2adt/model_best_nn_gex2adt_v2_tfidfconcat_Nov18-14-53.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/gex2adt/mod1_idf_v2.npy"
        )
        y_pred_concat = pretrin_nn(
            par["input_test_mod1"],
            model_pth,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # model (11) pca
        y_pred_pca = pca(input_train_mod1, input_test_mod1, n=50)

        # model (19) concat 2
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/gex2adt/model_nn_gex2adt_v2_tfidfconcat_Nov17-13-24.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/gex2adt/mod1_idf.npy"
        )
        y_pred_concat2 = pretrin_nn(
            par["input_test_mod1"],
            model_pth,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # ensemble
        # (test v1) use best
        y_pred = (
            np.array(y_pred_nn)
            + np.array(y_pred_nn2)
            + np.array(y_pred_bgan)
            + np.array(y_pred_concat)
            + np.array(y_pred_pca)
            + np.array(y_pred_concat2)
        ) / 6


elif input_train_mod1.var["feature_types"][0] == "ADT":
    logging.info("ADT to GEX")
    LOAD_MODEL = MOD1_DIM == 134 and MOD2_DIM == 13953

    if LOAD_MODEL:
        # model (1) pca
        y_pred_pca = pca(input_train_mod1, input_test_mod1, n=50)

        # model (2) concat
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/adt2gex/model_AtoB_cycle_adt2gex_concat_.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/adt2gex/mod1_idf.npy"
        )
        y_pred_concat = pretrin_nn(
            par["input_test_mod1"],
            model_pth,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # model (3) concat 2
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/adt2gex/model_best_AtoB_cycle_adt2gex_tfidfconcat_Nov03-14-47.pt"
        )
        y_pred_concat2 = pretrin_nn(
            par["input_test_mod1"],
            model_pth,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # model (5) concat 3
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/adt2gex/model_best_AtoB_cycle_adt2gex_v2_tfidfconcat_Nov17-21-42.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/adt2gex/mod1_idf_v2.npy"
        )
        y_pred_concat3 = pretrin_nn(
            par["input_test_mod1"],
            model_pth,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # model (6) nn
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/adt2gex/model_best_AtoB_cycle_adt2gex_v2_Nov17-18-23.pt"
        )
        y_pred_nn = pretrin_nn(
            par["input_test_mod1"], model_pth, MOD1_DIM, MOD2_DIM, FEAT_DIM, HIDDEN_DIM
        )

        # ensemble
        # (test_v1) v1 + v2 model
        y_pred = (
            np.array(y_pred_pca)
            + np.array(y_pred_concat)
            + np.array(y_pred_concat2)
            + np.array(y_pred_concat3)
            + np.array(y_pred_nn)
        ) / 5
        # (test_v2) v2 model
        # y_pred = (
        #     np.array(y_pred_pca) + np.array(y_pred_concat3) + np.array(y_pred_nn)
        #     ) / 3

elif input_train_mod1.var["feature_types"][0] == "ATAC":
    logging.info("ATAC to GEX")
    LOAD_MODEL = MOD1_DIM == 116490 and MOD2_DIM == 13431

    if LOAD_MODEL:
        # model (4) ga
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/atac2gex/model_AtoB_cycle_atac2gex_v2_ga_Nov19-00-04.pt"
        )
        y_pred_ga = pretrin_nn(
            par["input_test_mod1"],
            model_pth,
            19039,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            gene_activity=True,
        )

        # model (11) ga 2
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/atac2gex/model_AtoB_cycle_atac2gex_v2_ga_Nov19-10-45.pt"
        )
        y_pred_ga2 = pretrin_nn(
            par["input_test_mod1"],
            model_pth,
            19039,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            gene_activity=True,
        )

        # model (14) tfidf
        model_pth = (
            meta["resources_dir"]
            + "/model/weights_v4/atac2gex/model_best_AtoB_cycle_atac2gex_v2_tfidf_Nov16-23-08.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/atac2gex/mod1_idf.npy"
        )
        y_pred_tfidf = pretrin_nn(
            par["input_test_mod1"],
            model_pth,
            MOD1_DIM,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=1,
            idf_matrix=mod1_idf,
        )

        # ensemble
        y_pred = (
            np.array(y_pred_ga) + np.array(y_pred_ga2) + np.array(y_pred_tfidf)
        ) / 3


if not LOAD_MODEL:
    # PCA method
    y_pred = pca(input_train_mod1, input_test_mod1, n=50)

y_pred = csc_matrix(y_pred)

# use only in python test
"""
from model.utils.metric import rmse
mod2_sol = ad.read_h5ad(par['input_test_mod2']).X
logging.info(rmse(y_pred, mod2_sol))
# """

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
