""" opt file for the model """


def model_opts(parser):
    """model opts"""
    # model / data setting
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "gex2atac",
            "gex2adt",
            "adt2gex",
            "atac2gex",
            "gex2atac_v2",
            "gex2adt_v2",
            "adt2gex_v2",
            "atac2gex_v2",
            "gex2atac_p2",
            "gex2adt_p2",
            "adt2gex_p2",
            "atac2gex_p2",
        ],
        help="Desired training mode, \
        v2: phase 1 v2 data, p2: phase 2 data",
    )
    
    parser.add_argument(
        "--train",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Training or evaluating the model"
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="nn",
        choices=["nn", "cycle", "batchgan"],
        help="Desired training architecture",
    )

    parser.add_argument(
        "--train_batch",
        type=str,
        nargs="*",
        default=[],
        help="Desired training batch \
            v1: train = ['s1d1', 's2d1', 's2d4', 's3d6', 's3d7'], v1 test  = ['s1d2'] \
            v2 train = ['s1d1', 's1d2', 's1d3', 's2d1', 's2d4', 's2d5', 's3d6', 's3d7'] \
            v2 test  = ['s1d2', 's3d10'] \
            p3 train = [s1: d1, d2, d3; s2: d1, d4, d5; s3: d1, d3, d6, d7, d10]",
    )
    parser.add_argument(
        "--test_batch",
        type=str,
        nargs="*",
        default=[],
        help="Desired testing batch"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6666,
        help="Seed used for reproducibility in spliting phase 2 dataset into train/tes set. \
            Useless if training on phase 1 v1 & phase 1 v2 data."
    )

    # optimization
    parser.add_argument("--epoch", "-e", type=int, default=200)
    parser.add_argument("--batch_size", "-bs", type=int, default=2048)
    parser.add_argument("--lr", "-lr", type=float, default=0.1)
    parser.add_argument("--lr_decay_epoch", type=int, default=40)
    parser.add_argument("--momentum", type=float, default=0.9)

    # model architecture
    parser.add_argument("--dropout", "-dp", type=float, default=0.2)
    parser.add_argument("--hid_dim", type=int, default=1000)
    parser.add_argument("--emb_dim", type=int, default=128)

    # loss functions
    parser.add_argument("--reg_loss_weight", type=float, default=0)
    parser.add_argument("--rec_loss_weight", type=float, default=10)

    # data preprocessing
    parser.add_argument(
        "--norm",
        action="store_true",
        help="True for normalize mod1 input data batch-wise",
    )
    parser.add_argument(
        "--gene_activity",
        action="store_true",
        help="True for use gene activity feature in mod1 input, \
            Can be apply only on atac2gex* mode",
    )
    parser.add_argument(
        "--selection",
        action="store_true",
        help="True for using the selected feature index",
    )
    parser.add_argument(
        "--mod1_idx_path",
        type=str,
        default=None,
        help="The path to mod1 index path (.np), required when selection=True")
    parser.add_argument(
        "--tfidf",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="The tfidf mode. \
        0: raw data input \
        1: tfidf input \
        2: concat [raw, tfidf] feature \
        3: concat [gene activity, tfidf] feature",
    )
    parser.add_argument(
        "--idf_path",
        type=str,
        default=None,
        help="The path to pre-calculated idf matrix, required if tfidf != 0",
    )

    # save/load model
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="True: saves weights, logs, runs (tensorboard) during training, \
            False: saves runs (tensorboard) only during training",
    )
    parser.add_argument("--save_best_from", "-best", type=int, default=50)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained model checkpoint",
    )

    # others
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument(
        "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0, use -1 for CPU"
    )


# phase 1 v1 dataset
V1_DIR = "../output/datasets/predict_modality"
CITE_PTH = "openproblems_bmmc_cite_phase1"
MULTIOME_PTH = "openproblems_bmmc_multiome_phase1"

ADT2GEX_PTH = "openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_"
GEX2ADT_PTH = "openproblems_bmmc_cite_phase1_rna.censor_dataset.output_"
ATAC2GEX_PTH = "openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_"
GEX2ATAC_PTH = "openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_"

ADT2GEX = {
    "train_mod1": f"{V1_DIR}/{CITE_PTH}_mod2/{ADT2GEX_PTH}train_mod1.h5ad",
    "train_mod2": f"{V1_DIR}/{CITE_PTH}_mod2/{ADT2GEX_PTH}train_mod2.h5ad",
    "test_mod1": f"{V1_DIR}/{CITE_PTH}_mod2/{ADT2GEX_PTH}test_mod1.h5ad",
    "test_mod2": f"{V1_DIR}/{CITE_PTH}_mod2/{ADT2GEX_PTH}test_mod2.h5ad",
}

GEX2ADT = {
    "train_mod1": f"{V1_DIR}/{CITE_PTH}_rna/{GEX2ADT_PTH}train_mod1.h5ad",
    "train_mod2": f"{V1_DIR}/{CITE_PTH}_rna/{GEX2ADT_PTH}train_mod2.h5ad",
    "test_mod1": f"{V1_DIR}/{CITE_PTH}_rna/{GEX2ADT_PTH}test_mod1.h5ad",
    "test_mod2": f"{V1_DIR}/{CITE_PTH}_rna/{GEX2ADT_PTH}test_mod2.h5ad",
}

ATAC2GEX = {
    "train_mod1": f"{V1_DIR}/{MULTIOME_PTH}_mod2/{ATAC2GEX_PTH}train_mod1.h5ad",
    "train_mod2": f"{V1_DIR}/{MULTIOME_PTH}_mod2/{ATAC2GEX_PTH}train_mod2.h5ad",
    "test_mod1": f"{V1_DIR}/{MULTIOME_PTH}_mod2/{ATAC2GEX_PTH}test_mod1.h5ad",
    "test_mod2": f"{V1_DIR}/{MULTIOME_PTH}_mod2/{ATAC2GEX_PTH}test_mod2.h5ad",
}

GEX2ATAC = {
    "train_mod1": f"{V1_DIR}/{MULTIOME_PTH}_rna/{GEX2ATAC_PTH}train_mod1.h5ad",
    "train_mod2": f"{V1_DIR}/{MULTIOME_PTH}_rna/{GEX2ATAC_PTH}train_mod2.h5ad",
    "test_mod1": f"{V1_DIR}/{MULTIOME_PTH}_rna/{GEX2ATAC_PTH}test_mod1.h5ad",
    "test_mod2": f"{V1_DIR}/{MULTIOME_PTH}_rna/{GEX2ATAC_PTH}test_mod2.h5ad",
}

# phase 1 v2 dataset
V2_DIR = "../output/datasets_phase1v2/predict_modality"
CITE_V2_PTH = "openproblems_bmmc_cite_phase1v2"
MULTIOME_V2_PTH = "openproblems_bmmc_multiome_phase1v2"

ADT2GEX_V2_PTH = "openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_"
GEX2ADT_V2_PTH = "openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_"
ATAC2GEX_V2_PTH = "openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_"
GEX2ATAC_V2_PTH = "openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_"

ADT2GEX_V2 = {
    "train_mod1": f"{V2_DIR}/{CITE_V2_PTH}_mod2/{ADT2GEX_V2_PTH}train_mod1.h5ad",
    "train_mod2": f"{V2_DIR}/{CITE_V2_PTH}_mod2/{ADT2GEX_V2_PTH}train_mod2.h5ad",
    "test_mod1": f"{V2_DIR}/{CITE_V2_PTH}_mod2/{ADT2GEX_V2_PTH}test_mod1.h5ad",
    "test_mod2": f"{V2_DIR}/{CITE_V2_PTH}_mod2/{ADT2GEX_V2_PTH}test_mod2.h5ad",
}

GEX2ADT_V2 = {
    "train_mod1": f"{V2_DIR}/{CITE_V2_PTH}_rna/{GEX2ADT_V2_PTH}train_mod1.h5ad",
    "train_mod2": f"{V2_DIR}/{CITE_V2_PTH}_rna/{GEX2ADT_V2_PTH}train_mod2.h5ad",
    "test_mod1": f"{V2_DIR}/{CITE_V2_PTH}_rna/{GEX2ADT_V2_PTH}test_mod1.h5ad",
    "test_mod2": f"{V2_DIR}/{CITE_V2_PTH}_rna/{GEX2ADT_V2_PTH}test_mod2.h5ad",
}

ATAC2GEX_V2 = {
    "train_mod1": f"{V2_DIR}/{MULTIOME_V2_PTH}_mod2/{ATAC2GEX_V2_PTH}train_mod1.h5ad",
    "train_mod2": f"{V2_DIR}/{MULTIOME_V2_PTH}_mod2/{ATAC2GEX_V2_PTH}train_mod2.h5ad",
    "test_mod1": f"{V2_DIR}/{MULTIOME_V2_PTH}_mod2/{ATAC2GEX_V2_PTH}test_mod1.h5ad",
    "test_mod2": f"{V2_DIR}/{MULTIOME_V2_PTH}_mod2/{ATAC2GEX_V2_PTH}test_mod2.h5ad",
}

GEX2ATAC_V2 = {
    "train_mod1": f"{V2_DIR}/{MULTIOME_V2_PTH}_rna/{GEX2ATAC_V2_PTH}train_mod1.h5ad",
    "train_mod2": f"{V2_DIR}/{MULTIOME_V2_PTH}_rna/{GEX2ATAC_V2_PTH}train_mod2.h5ad",
    "test_mod1": f"{V2_DIR}/{MULTIOME_V2_PTH}_rna/{GEX2ATAC_V2_PTH}test_mod1.h5ad",
    "test_mod2": f"{V2_DIR}/{MULTIOME_V2_PTH}_rna/{GEX2ATAC_V2_PTH}test_mod2.h5ad",
}

# phase 2 dataset
P2_DIR = "../output/datasets_phase2/predict_modality"
CITE_P2_PTH = "openproblems_bmmc_cite_phase2"
MULTIOME_P2_PTH = "openproblems_bmmc_multiome_phase2"

ADT2GEX_P2_PTH = "openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_"
GEX2ADT_P2_PTH = "openproblems_bmmc_cite_phase2_rna.censor_dataset.output_"
ATAC2GEX_P2_PTH = "openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_"
GEX2ATAC_P2_PTH = "openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_"

ADT2GEX_P2 = {
    "train_mod1": f"{P2_DIR}/{CITE_P2_PTH}_mod2/{ADT2GEX_P2_PTH}train_mod1.h5ad",
    "train_mod2": f"{P2_DIR}/{CITE_P2_PTH}_mod2/{ADT2GEX_P2_PTH}train_mod2.h5ad",
    "test_mod1": f"{P2_DIR}/{CITE_P2_PTH}_mod2/{ADT2GEX_P2_PTH}train_mod1.h5ad",
    "test_mod2": f"{P2_DIR}/{CITE_P2_PTH}_mod2/{ADT2GEX_P2_PTH}train_mod2.h5ad",
}

GEX2ADT_P2 = {
    "train_mod1": f"{P2_DIR}/{CITE_P2_PTH}_rna/{GEX2ADT_P2_PTH}train_mod1.h5ad",
    "train_mod2": f"{P2_DIR}/{CITE_P2_PTH}_rna/{GEX2ADT_P2_PTH}train_mod2.h5ad",
    "test_mod1": f"{P2_DIR}/{CITE_P2_PTH}_rna/{GEX2ADT_P2_PTH}train_mod1.h5ad",
    "test_mod2": f"{P2_DIR}/{CITE_P2_PTH}_rna/{GEX2ADT_P2_PTH}train_mod2.h5ad",
}

ATAC2GEX_P2 = {
    "train_mod1": f"{P2_DIR}/{MULTIOME_P2_PTH}_mod2/{ATAC2GEX_P2_PTH}train_mod1.h5ad",
    "train_mod2": f"{P2_DIR}/{MULTIOME_P2_PTH}_mod2/{ATAC2GEX_P2_PTH}train_mod2.h5ad",
    "test_mod1": f"{P2_DIR}/{MULTIOME_P2_PTH}_mod2/{ATAC2GEX_P2_PTH}train_mod1.h5ad",
    "test_mod2": f"{P2_DIR}/{MULTIOME_P2_PTH}_mod2/{ATAC2GEX_P2_PTH}train_mod2.h5ad",
}

GEX2ATAC_P2 = {
    "train_mod1": f"{P2_DIR}/{MULTIOME_P2_PTH}_rna/{GEX2ATAC_P2_PTH}train_mod1.h5ad",
    "train_mod2": f"{P2_DIR}/{MULTIOME_P2_PTH}_rna/{GEX2ATAC_P2_PTH}train_mod2.h5ad",
    "test_mod1": f"{P2_DIR}/{MULTIOME_P2_PTH}_rna/{GEX2ATAC_P2_PTH}train_mod1.h5ad",
    "test_mod2": f"{P2_DIR}/{MULTIOME_P2_PTH}_rna/{GEX2ATAC_P2_PTH}train_mod2.h5ad",
}

# datasets
DATASET = {
    "atac2gex": ATAC2GEX,
    "adt2gex": ADT2GEX,
    "gex2adt": GEX2ADT,
    "gex2atac": GEX2ATAC,
    "atac2gex_v2": ATAC2GEX_V2,
    "adt2gex_v2": ADT2GEX_V2,
    "gex2adt_v2": GEX2ADT_V2,
    "gex2atac_v2": GEX2ATAC_V2,
    "atac2gex_p2": ATAC2GEX_P2,
    "adt2gex_p2": ADT2GEX_P2,
    "gex2adt_p2": GEX2ADT_P2,
    "gex2atac_p2": GEX2ATAC_P2,
}
