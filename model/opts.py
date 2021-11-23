import argparse

def model_opts(parser):
    parser.add_argument("--mode", type=str, required=True, 
        choices=[
            'gex2atac', 'gex2adt', 'adt2gex', 'atac2gex',
            'gex2atac_v2', 'gex2adt_v2', 'adt2gex_v2', 'atac2gex_v2',
            'gex2atac_p2', 'gex2adt_p2', 'adt2gex_p2', 'atac2gex_p2'])

    parser.add_argument("--train_batch", type=str, nargs='*', default=[],
        help="v1: train = ['s1d1', 's2d1', 's2d4', 's3d6', 's3d7'], \
            v1 test  = ['s1d2'] \
            v2 train = ['s1d1', 's1d2', 's1d3', 's2d1', 's2d4', 's2d5', 's3d6', 's3d7'] \
            v2 test  = ['s1d2', 's3d10'] \
            p3 train = ['s1d1', 's1d2', 's1d3', 's2d1', 's2d4', 's2d5', 's3d1', 's3d3', 's3d6', 's3d7', 's3d10']")
    parser.add_argument("--test_batch", type=str, nargs='*', default=[])
    parser.add_argument("--arch", type=str, default='nn',
        choices=[
            'nn', 'cycle', 'scvi', 'peakvi', 'rec', 'peakrec', 'scvirec',
            'pairae', 'residual', 'pix2pix', 'raw', 'batchgan'])
    parser.add_argument("--epoch", "-e", type=int, default=50)
    parser.add_argument("--batch_size", "-bs", type=int, default=2048)

    parser.add_argument("--dropout", "-dp", type=float, default=0.2)
    parser.add_argument("--hid_dim", type=int, default=1000)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--dryrun", action="store_true")

    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--gene_activity", action="store_true")
    parser.add_argument("--selection", action="store_true")
    parser.add_argument("--select_dim", type=int, default=1000)
    parser.add_argument("--mod1_idx_path", type=str, default=None)
    parser.add_argument("--mod2_idx_path", type=str, default=None)

    parser.add_argument("--tfidf", type=int, default=0, choices=[0, 1, 2, 3], 
        help='0: raw data input, 1: tfidf input, 2: concat [raw, tfidf] data input')
    parser.add_argument("--idf_path", type=str, default=None)

    parser.add_argument("--lr", "-lr", type=float, default=0.1)
    parser.add_argument("--lr_decay_epoch", type=int, default=80)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--save_best_from", "-best", type=int, default=50)

    parser.add_argument("--rec_loss_weight", type=float, default=10)
    parser.add_argument("--cmn_loss_weight", type=float, default=1)
    parser.add_argument("--cos_loss_weight", type=float, default=1)
    parser.add_argument("--reg_loss_weight", type=float, default=0)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pretrain_scvi", type=str, default=None)
    parser.add_argument("--pretrain_weight", type=str, default=None)
    parser.add_argument("--pretrain_epoch", type=int, default=100)

    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--seed", type=int, default=6666)
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

ADT2GEX_v2 = {
    'train_mod1': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_train_mod1.h5ad',
    'train_mod2': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_test_mod2.h5ad'
}

GEX2ADT_v2 = {
    'train_mod1': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_test_mod2.h5ad'
}

ATAC2GEX_v2 = {
    'train_mod1': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_test_mod2.h5ad'
}

GEX2ATAC_v2 = {
    'train_mod1': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_train_mod1.h5ad',
    'train_mod2': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_train_mod2.h5ad',
    'test_mod1': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_test_mod1.h5ad', 
    'test_mod2': '../output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_test_mod2.h5ad' 
}

ADT2GEX_p2 = {
    'train_mod1': '../output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod1.h5ad', 
    'test_mod2': '../output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod2.h5ad'
}

GEX2ADT_p2 = {
    'train_mod1': '../output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod1.h5ad', 
    'test_mod2': '../output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod2.h5ad' 
}

ATAC2GEX_p2 = {
    'train_mod1': '../output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_mod2/openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_train_mod1.h5ad', 
    'train_mod2': '../output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_mod2/openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_train_mod2.h5ad', 
    'test_mod1': '../output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_mod2/openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_train_mod1.h5ad', 
    'test_mod2': '../output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_mod2/openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_train_mod2.h5ad'
}

GEX2ATAC_p2 = {
    'train_mod1': '../output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod1.h5ad',
    'train_mod2': '../output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod2.h5ad',
    'test_mod1': '../output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod1.h5ad',
    'test_mod2': '../output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod2.h5ad'
}

DATASET = {
    "atac2gex": ATAC2GEX, 
    "adt2gex": ADT2GEX, 
    "gex2adt": GEX2ADT, 
    "gex2atac": GEX2ATAC,
    "atac2gex_v2": ATAC2GEX_v2, 
    "adt2gex_v2": ADT2GEX_v2, 
    "gex2adt_v2": GEX2ADT_v2, 
    "gex2atac_v2": GEX2ATAC_v2,
    "atac2gex_p2": ATAC2GEX_p2, 
    "adt2gex_p2": ADT2GEX_p2, 
    "gex2adt_p2": GEX2ADT_p2, 
    "gex2atac_p2": GEX2ATAC_p2
}