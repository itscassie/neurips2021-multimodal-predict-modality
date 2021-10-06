import argparse

def model_opts(parser):
    parser.add_argument("--mode", type=str, choices=['gex2atac', 'gex2adt', 'adt2gex', 'atac2gex'], required=True)
    parser.add_argument("--arch", type=str, choices=['nn', 'pairae'], default='nn')
    parser.add_argument("--epoch", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-bs", type=int, default=2048)

    parser.add_argument("--hid_dim", type=int, default=1000)
    parser.add_argument("--emb_dim", type=int, default=128)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_decay_epoch", type=int, default=40)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--rec_loss_weight", type=float, default=10)
    parser.add_argument("--common_loss_weight", type=float, default=1)


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

DATASET = {
    "atac2gex": ATAC2GEX, 
    "adt2gex": ADT2GEX, 
    "gex2adt": GEX2ADT, 
    "gex2atac": GEX2ATAC
}