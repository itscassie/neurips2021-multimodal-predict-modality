""" 
This is the script of modality prediction of all 4 subtasks
Dependencies: 
pip: scikit-learn, anndata, scanpy, numpy
"""

import sys
import logging
import anndata as ad
import numpy as np

import torch
from torch.utils.data import DataLoader

from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
meta = {"resources_dir": "."}
task = "sample"
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
    # added for test, comment if unused
    "input_test_mod2": mode[task] + "test_mod2.h5ad",
    "distance_method": "minkowski",
    "output": "output.h5ad",
    "n_pcs": 50,
}
## VIASH END
sys.path.append(meta["resources_dir"])
from model.utils.metric import rmse
from model.modules.model_ae import AutoEncoder, BatchRemovalGAN
from model.utils.dataloader import SeqDataset

# Methods
method_id = "mse_v10"

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
    """
    load the pre-trained nn / cycle AtoB model
    input the mod 1 test data, output the mod 2 prediction

    Parameters
    ----------
    test_mod1
        Path to .h5ad file of mod1 test set
    model_pth
        Path to pre-trained model
    mod1_dim
        The dimension of mod1 dataset
    mod2_dim
        The dimension of mod2 dataset
    feat_dim
        The dimension of pre-trained model embedding feature
    hidden_dim
        The dimension of pre-trained model hidden layer
    mod1_idx_path
        The path to mod1 index path (.np file), use when selection=True
        e.g., 2pct, 5pct mode
        Default: None
    tfidf
        The tfidf mode. 
        0: do not use the tfidf feature (mod1_dim = mod1_dim)
        1: use the tfidf feature (mod1_dim = mod1_dim)
        2: concat raw & tfidf feature (mod1_dim = mod1_dim * 2)
        3: concat gene activity & tfidf feature (mod1_dim = mod1_dim + ga_dim)
        Default: 0
    idf_matrix
        The path to pre-calculated idf matrix, required if tfidf != 0
        Default: None
    gene_activity
        Use gene activity feautre in atac2gex mode
        Dafault: False
    log
        Show the pre-trained model archeture
        Default: False
    """

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
    for _, (mod1_seq, _) in enumerate(test_loader):
        mod1_seq = mod1_seq.float()
        mod2_rec = model_ae(mod1_seq)

        mod2_rec = mod2_rec.data.cpu().numpy()
        mod2_matrix = np.vstack((mod2_matrix, mod2_rec))

    mod2_pred = np.array(
        mod2_matrix[
            1:,
        ]
    )
    logging.info("Finish Prediction")

    return mod2_pred


# pretrain batch gan
def pretrin_batchgan(
    test_mod1,
    model_pth,
    mod1_dim,
    mod2_dim,
    feat_dim,
    hidden_dim,
    cls_num=10,
):
    """
    load the pre-trained batchgan model
    input the mod 1 test data, output the mod 2 prediction

    Parameters
    ----------
    test_mod1
        Path to .h5ad file of mod1 test set
    model_pth
        Path to pre-trained model
    mod1_dim
        The dimension of mod1 dataset
    mod2_dim
        The dimension of mod2 dataset
    cls_num
        The batch number that batchgan was trained
    """

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

    for _, (mod1_seq, _) in enumerate(test_loader):
        mod1_seq = mod1_seq.float()
        mod2_rec, _ = model_ae(mod1_seq)

        mod2_rec = mod2_rec.data.cpu().numpy()
        mod2_matrix = np.vstack((mod2_matrix, mod2_rec))

    mod2_pred = np.array(
        mod2_matrix[
            1:,
        ]
    )
    logging.info("Finish Prediction")

    return mod2_pred


# do pca
def pca(
    input_train_mod1,
    input_train_mod2,
    input_test_mod1,
    n=50,
    alg="randomized",
    n_iter=5,
    seed=6666,
):
    """
    Apply PCA on the data
    input the mod 1 & mod2 train data, mod1 test data
    output the mod 2 prediction

    Parameters
    ----------
    input_train_mod1
        The anndata format of mod 1 input training data
    input_train_mod2
        The anndata format of mod 2 input training data
    input_test_mod1
        The anndata format of mod 1 input testing data
    n: int, default=50
        Desired dimensionality of output data. Must be strictly less than the number of features.
    alg: {‘arpack’, ‘randomized’}, default=’randomized’
        SVD solver to use. 
        Either “arpack” for the ARPACK wrapper in SciPy (scipy.sparse.linalg.svds),
        or “randomized” for the randomized algorithm due to Halko (2009).
    n_iter: int, default=5
        Number of iterations for randomized SVD solver. Not used by ARPACK.
    seed: int, default=6666 
        Used during randomized svd. 
        Pass an int for reproducible results across multiple function calls.
        or use None for nonreproducible random states
    """
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
    embedder_mod1 = TruncatedSVD(
        n_components=n, algorithm=alg, n_iter=n_iter, random_state=seed
    )
    mod1_pca = embedder_mod1.fit_transform(input_train.X)

    logging.info("Performing dimensionality reduction on modality 2 values...")
    embedder_mod2 = TruncatedSVD(
        n_components=n, algorithm=alg, n_iter=n_iter, random_state=seed
    )
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
    mod2_pred = reg.predict(X_test)

    # Project the predictions back to the modality 2 feature space
    mod2_pred = mod2_pred @ embedder_mod2.components_
    logging.info("Finish Prediction")

    return np.array(mod2_pred)


logging.info("Reading `h5ad` files...")
INPUT_TRAIN_MOD1 = ad.read_h5ad(par["input_train_mod1"])
INPUT_TRAIN_MOD2 = ad.read_h5ad(par["input_train_mod2"])
INPUT_TEST_MOD1 = ad.read_h5ad(par["input_test_mod1"])

# Check data shape
LOAD_MODEL = True
MOD1_DIM = int(INPUT_TRAIN_MOD1.X.shape[1])
MOD2_DIM = int(INPUT_TRAIN_MOD2.X.shape[1])
FEAT_DIM = 128
HIDDEN_DIM = 1000

# check input format and apply different methods for each subtask
if INPUT_TRAIN_MOD2.var["feature_types"][0] == "ATAC":
    logging.info("GEX to ATAC")
    LOAD_MODEL = MOD1_DIM == 13431 and MOD2_DIM == 10000

    if LOAD_MODEL:
        # model (1) pca
        y1_pred_pca = pca(
            INPUT_TRAIN_MOD1, INPUT_TRAIN_MOD2, INPUT_TEST_MOD1, n=50, alg="arpack"
        )

        # model (8b) concat
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/gex2atac/model_best_AtoB_cycle_gex2atac_v2_tfidfconcat_Nov18-07-03.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/gex2atac/mod1_idf_v2.npy"
        )
        y8b_pred_concat = pretrin_nn(
            par["input_test_mod1"],
            MODEL_PTH,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # ensemble
        y_pred = (np.array(y1_pred_pca) + np.array(y8b_pred_concat)) / 2

elif INPUT_TRAIN_MOD2.var["feature_types"][0] == "ADT":
    logging.info("GEX to ADT")
    LOAD_MODEL = MOD1_DIM == 13953 and MOD2_DIM == 134

    if LOAD_MODEL:

        # model (8b) nn
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/gex2adt/model_best_nn_gex2adt_v2_reg1_Nov18-21-24.pt"
        )
        y8b_pred_nn = pretrin_nn(
            par["input_test_mod1"], MODEL_PTH, MOD1_DIM, MOD2_DIM, FEAT_DIM, HIDDEN_DIM
        )

        # model (9b) nn
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/gex2adt/model_best_nn_gex2adt_v2_new_Nov20-09-35.pt"
        )
        y9b_pred_nn = pretrin_nn(
            par["input_test_mod1"], MODEL_PTH, MOD1_DIM, MOD2_DIM, FEAT_DIM, HIDDEN_DIM
        )

        # model (11b) nn
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/gex2adt/model_best_nn_gex2adt_v2_dropout0.3_Nov23-19-20.pt"
        )
        y11b_pred_nn = pretrin_nn(
            par["input_test_mod1"], MODEL_PTH, MOD1_DIM, MOD2_DIM, FEAT_DIM, HIDDEN_DIM
        )

        # model (23) pca
        y23_pred_pca = pca(
            INPUT_TRAIN_MOD1, INPUT_TRAIN_MOD2, INPUT_TEST_MOD1, n=50, alg="arpack"
        )

        # model (25b) batchgan
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/gex2adt/model_best_batchgan_gex2adt_v2_dropout0.3_Nov24-22-36.pt"
        )
        y25b_pred_bgan = pretrin_batchgan(
            par["input_test_mod1"],
            MODEL_PTH,
            MOD1_DIM,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            cls_num=10,
        )

        # model (16b) concat
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/gex2adt/model_best_nn_gex2adt_v2_tfidfconcat_Nov18-14-53.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/gex2adt/mod1_idf_v2.npy"
        )
        y16b_pred_concat = pretrin_nn(
            par["input_test_mod1"],
            MODEL_PTH,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # model (17b) concat
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/gex2adt/model_best_nn_gex2adt_v2_tfidfconcat_new_Nov20-00-53.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/gex2adt/mod1_idf.npy"
        )
        y17b_pred_concat = pretrin_nn(
            par["input_test_mod1"],
            MODEL_PTH,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # model (18b) concat
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/gex2adt/model_best_nn_gex2adt_v2_tfidfconcat_dropout0.3_Nov24-10-24.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/gex2adt/mod1_idf_v2.npy"
        )
        y18b_pred_concat = pretrin_nn(
            par["input_test_mod1"],
            MODEL_PTH,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # model (22) pca
        y22_pred_pca = pca(
            INPUT_TRAIN_MOD1, INPUT_TRAIN_MOD2, INPUT_TEST_MOD1, n=50, n_iter=10
        )

        # model (27) pca
        y27_pred_pca = pca(INPUT_TRAIN_MOD1, INPUT_TRAIN_MOD2, INPUT_TEST_MOD1, n=100)

        # ensemble (10)
        y_pred = (
            np.array(y8b_pred_nn)
            + np.array(y9b_pred_nn)
            + np.array(y11b_pred_nn)
            + np.array(y23_pred_pca)
            + np.array(y25b_pred_bgan)
            + np.array(y16b_pred_concat)
            + np.array(y17b_pred_concat)
            + np.array(y18b_pred_concat)
            + np.array(y22_pred_pca)
            + np.array(y27_pred_pca)
        ) / 10


elif INPUT_TRAIN_MOD1.var["feature_types"][0] == "ADT":
    logging.info("ADT to GEX")
    LOAD_MODEL = MOD1_DIM == 134 and MOD2_DIM == 13953

    if LOAD_MODEL:
        # model (1) pca
        y1_pred_pca = pca(INPUT_TRAIN_MOD1, INPUT_TRAIN_MOD2, INPUT_TEST_MOD1, n=50)

        # model (2) concat
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/adt2gex/model_AtoB_cycle_adt2gex_concat_.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/adt2gex/mod1_idf.npy"
        )
        y2_pred_concat = pretrin_nn(
            par["input_test_mod1"],
            MODEL_PTH,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # model (3) concat
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/adt2gex/model_best_AtoB_cycle_adt2gex_tfidfconcat_Nov03-14-47.pt"
        )
        y3_pred_concat = pretrin_nn(
            par["input_test_mod1"],
            MODEL_PTH,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # model (5) concat
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/adt2gex/model_best_AtoB_cycle_adt2gex_v2_tfidfconcat_Nov17-21-42.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/adt2gex/mod1_idf_v2.npy"
        )
        y5_pred_concat = pretrin_nn(
            par["input_test_mod1"],
            MODEL_PTH,
            MOD1_DIM * 2,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=2,
            idf_matrix=mod1_idf,
        )

        # model (6) nn
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/adt2gex/model_best_AtoB_cycle_adt2gex_v2_Nov17-18-23.pt"
        )
        y6_pred_nn = pretrin_nn(
            par["input_test_mod1"], MODEL_PTH, MOD1_DIM, MOD2_DIM, FEAT_DIM, HIDDEN_DIM
        )

        # ensemble
        y_pred = (
            np.array(y1_pred_pca)
            + np.array(y2_pred_concat)
            + np.array(y3_pred_concat)
            + np.array(y5_pred_concat)
            + np.array(y6_pred_nn)
        ) / 5


elif INPUT_TRAIN_MOD1.var["feature_types"][0] == "ATAC":
    logging.info("ATAC to GEX")
    LOAD_MODEL = MOD1_DIM == 116490 and MOD2_DIM == 13431

    if LOAD_MODEL:
        # model (5b) ga
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/atac2gex/model_best_AtoB_cycle_atac2gex_v2_ga_Nov19-00-04.pt"
        )
        y5b_pred_ga = pretrin_nn(
            par["input_test_mod1"],
            MODEL_PTH,
            19039,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            gene_activity=True,
        )

        # model (6b) ga
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/atac2gex/model_best_AtoB_cycle_atac2gex_v2_ga_Nov19-10-45.pt"
        )
        y6b_pred_ga = pretrin_nn(
            par["input_test_mod1"],
            MODEL_PTH,
            19039,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            gene_activity=True,
        )

        # model (8b) tfidf
        MODEL_PTH = (
            meta["resources_dir"]
            + "/model/weights_v10/atac2gex/model_best_AtoB_cycle_atac2gex_v2_tfidf_Nov16-23-08.pt"
        )
        mod1_idf = np.load(
            meta["resources_dir"] + "/model/idf_matrix/atac2gex/mod1_idf.npy"
        )
        y8b_pred_tfidf = pretrin_nn(
            par["input_test_mod1"],
            MODEL_PTH,
            MOD1_DIM,
            MOD2_DIM,
            FEAT_DIM,
            HIDDEN_DIM,
            tfidf=1,
            idf_matrix=mod1_idf,
        )

        # ensemble
        y_pred = (
            np.array(y5b_pred_ga) + np.array(y6b_pred_ga) + np.array(y8b_pred_tfidf)
        ) / 3

if not LOAD_MODEL:
    # PCA method
    y_pred = pca(
        INPUT_TRAIN_MOD1, INPUT_TRAIN_MOD2, INPUT_TEST_MOD1, n=50, alg="arpack"
    )

y_pred = csc_matrix(y_pred)

# use only in python test
# uncomment this block if you want to test rmse
# """
mod2_sol = ad.read_h5ad(par["input_test_mod2"]).X
logging.info(rmse(y_pred, mod2_sol))
# """

# Saving data to anndata format
logging.info("Storing annotated data...")

adata = ad.AnnData(
    X=y_pred,
    obs=INPUT_TEST_MOD1.obs,
    var=INPUT_TRAIN_MOD2.var,
    uns={
        "dataset_id": INPUT_TRAIN_MOD1.uns["dataset_id"],
        "method_id": method_id,
    },
)
adata.write_h5ad(par["output"], compression="gzip")
