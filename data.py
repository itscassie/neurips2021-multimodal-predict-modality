import anndata as ad

ADT2GEX = [
'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod2.h5ad'
]

GEX2ADT = [
'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad', 
'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad', 
'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad'
]

ATAC2GEX = [
'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad', 
'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad', 
'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad', 
'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
]

GEX2ATAC = [
'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad',
'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad',
'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad', 
'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad' 
]


ADT2GEX_v2 = [
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_train_mod1.h5ad', 
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_train_mod2.h5ad', 
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_test_mod1.h5ad', 
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_test_mod2.h5ad'
]

GEX2ADT_v2 = [
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_train_mod1.h5ad', 
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_train_mod2.h5ad', 
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_test_mod1.h5ad', 
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_test_mod2.h5ad'
]

ATAC2GEX_v2 = [
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_train_mod1.h5ad', 
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_train_mod2.h5ad', 
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_test_mod1.h5ad', 
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_mod2/openproblems_bmmc_multiome_phase1v2_mod2.censor_dataset.output_test_mod2.h5ad',
]

GEX2ATAC_v2 = [
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_train_mod1.h5ad',
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_train_mod2.h5ad',
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_test_mod1.h5ad', 
'output/datasets_phase1v2/predict_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_test_mod2.h5ad' 
]

ADT2GEX_p2 = [
'output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod1.h5ad', 
'output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod2.h5ad', 
]

GEX2ADT_p2 = [
'output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod1.h5ad', 
'output/datasets_phase2/predict_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod2.h5ad', 
]

ATAC2GEX_p2 = [
'output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_mod2/openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_train_mod1.h5ad', 
'output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_mod2/openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_train_mod2.h5ad', 
]

GEX2ATAC_p2 = [
'output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod1.h5ad',
'output/datasets_phase2/predict_modality/openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod2.h5ad',
]

# DATAPTH = [ADT2GEX, GEX2ADT, ATAC2GEX, GEX2ATAC]
# DATAPTH = [ADT2GEX_v2, GEX2ADT_v2, ATAC2GEX_v2, GEX2ATAC_v2]
DATAPTH = [ATAC2GEX_p2, GEX2ATAC_p2]

for (i, mode) in enumerate(DATAPTH):
    print(f"DIRECTION [{i + 1} / {len(DATAPTH)}]")
    
    train_mod1_pth = mode[0]
    train_mod2_pth = mode[1]
    # test_mod1_pth = mode[2]
    # test_mod2_pth = mode[3]

    train_mod1 = ad.read_h5ad(train_mod1_pth)
    train_mod2 = ad.read_h5ad(train_mod2_pth)
    # test_mod1 = ad.read_h5ad(test_mod1_pth)
    # test_mod2 = ad.read_h5ad(test_mod2_pth)
    print("train mod1:", train_mod1)
    print("train mod2:", train_mod2)
    # print("test mod1:", test_mod1)
    # print("test mod2:", test_mod2)
    
    print(train_mod1.obs["batch"])
    print(train_mod2.obs["batch"])
    # change here 
    print(f"MOD1: {train_mod1.var['feature_types'][0]}")
    print(f"MOD2: {train_mod2.var['feature_types'][0]}")
    print(f"MOD1_DIM: {train_mod1.X.shape[1]}, MOD2_DIM: {train_mod2.X.shape[1]}")
    print(f"TRAIN_NUM: {train_mod1.X.shape[0]}")
    print("batch: ", set(train_mod1.obs["batch"]))
    # print(f" TEST_NUM: {test_mod1.X.shape[0]}")

    # assert int(train_mod1.X.shape[1]) == int(test_mod1.X.shape[1]), "mod1 feature # train != test"
    # assert int(train_mod2.X.shape[1]) == int(test_mod2.X.shape[1]), "mod1 feature # train != test"
    assert int(train_mod1.X.shape[0]) == int(train_mod2.X.shape[0]), "train # mod1 != mod2"
    # assert int(test_mod1.X.shape[0]) == int(test_mod2.X.shape[0]), "train # mod1 != mod2"

"""
for pth in data_pths:
    print(pth)
    f = ad.read_h5ad(pth)
    print(f)
    print(f.var['feature_types'])
    # print(f.obs_names)
    print(f.obs["batch"])
    print(" ")
"""