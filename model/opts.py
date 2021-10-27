import argparse

def model_opts(parser):
    parser.add_argument("--mode", type=str, required=True, 
        choices=[
            'gex2atac', 'gex2adt', 'adt2gex', 'atac2gex',
            'gex2atac_tfidf', 'gex2adt_tfidf', 'adt2gex_tfidf', 'atac2gex_tfidf',
            'gex2atac_concat', 'gex2adt_concat', 'adt2gex_concat', 'atac2gex_concat',
            'atac2gex_5pct', 'atac2gex_2pct', 'atac2gex_5pct_tfidf', 'atac2gex_5pct_concat'])

    parser.add_argument("--arch", type=str, default='nn',
        choices=['nn', 'pairae', 'residual', 'pix2pix', 'cycle', 'decoder', 'kernelae', 'scvi', 'rec'], 
    )
    parser.add_argument("--epoch", "-e", type=int, default=50)
    parser.add_argument("--batch_size", "-bs", type=int, default=2048)

    parser.add_argument("--hid_dim", type=int, default=1000)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--dryrun", action="store_true")

    parser.add_argument("--selection", action="store_true")
    parser.add_argument("--select_dim", type=int, default=1000)
    parser.add_argument("--mod1_idx_path", type=str, default=None)
    parser.add_argument("--mod2_idx_path", type=str, default=None)

    parser.add_argument("--tfidf", type=int, default=0, choices=[0, 1, 2], 
        help='0: raw data input, 1: tfidf input, 2: concat [raw, tfidf] data input')
    parser.add_argument("--idf_path", type=str, default=None)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_decay_epoch", type=int, default=80)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--rec_loss_weight", type=float, default=10)
    parser.add_argument("--cmn_loss_weight", type=float, default=1)
    parser.add_argument("--cos_loss_weight", type=float, default=1)
    parser.add_argument("--reg_loss_weight", type=float, default=0)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pretrain_weight", type=str, default=None)
    parser.add_argument("--pretrain_epoch", type=int, default=100)

    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0, use -1 for CPU')

ADT2GEX = {
    'train_mod1': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod2.h5ad'
}

GEX2ADT = {
    'train_mod1': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad'
}

ATAC2GEX = {
    'train_mod1': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
}

GEX2ATAC = {
    'train_mod1': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad',
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad',
    'test_mod1': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad' 
}

# TODO: finish tfidf, 2pct in training script
TFIDF_ADT2GEX = {
    'train_mod1': '../output/datasets/predict_modality/tfidf_openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/tfidf_openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod2.h5ad'
}

TFIDF_GEX2ADT = {
    'train_mod1': '../output/datasets/predict_modality/tfidf_openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/tfidf_openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad'
}

TFIDF_ATAC2GEX = {
    'train_mod1': '../output/datasets/predict_modality/tfidf_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/tfidf_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
}

TFIDF_GEX2ATAC = {
    'train_mod1': '../output/datasets/predict_modality/concat_openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad',
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad',
    'test_mod1': '../output/datasets/predict_modality/concat_openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad' 
}

CONCAT_ADT2GEX = {
    'train_mod1': '../output/datasets/predict_modality/rawtfidf_openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/rawtfidf_openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod2.h5ad'
}

CONCAT_GEX2ADT = {
    'train_mod1': '../output/datasets/predict_modality/rawtfidf_openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/rawtfidf_openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad'
}

CONCAT_ATAC2GEX = {
    'train_mod1': '../output/datasets/predict_modality/rawtfidf_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/rawtfidf_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
}

CONCAT_GEX2ATAC = {
    'train_mod1': '../output/datasets/predict_modality/rawtfidf_openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad',
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad',
    'test_mod1': '../output/datasets/predict_modality/rawtfidf_openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad' 
}

PCT5_ATAC2GEX = {
    'train_mod1': '../output/datasets/predict_modality/pct5_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/pct5_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
}

PCT2_ATAC2GEX = {
    'train_mod1': '../output/datasets/predict_modality/pct2_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/pct2_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
}

CONCAT_PCT5_ATAC2GEX = {
    'train_mod1': '../output/datasets/predict_modality/rawtfidf_pct5_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/rawtfidf_pct5_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
}

TFIDF_PCT5_ATAC2GEX = {
    'train_mod1': '../output/datasets/predict_modality/tfidf_pct5_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets/predict_modality/tfidf_pct5_openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
}

DATASET = {
    "atac2gex": ATAC2GEX, 
    "adt2gex": ADT2GEX, 
    "gex2adt": GEX2ADT, 
    "gex2atac": GEX2ATAC
}