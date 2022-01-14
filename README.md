[scJoint] 
===
## NeurIPS 2021 Competition - Multimodal Single-Cell Modality Prediction

This repo contains the training pipeline and script used for the **NeurIPS 2021 Competition - Multimodal Single-Cell Data Integration**, the **Predict Modality** task. Our team **scJoint** achieved 3rd place of the modality prediction task ([leaderboard](https://eval.ai/web/challenges/challenge-page/1111/leaderboard/2860)) in terms of the overall ranking of 4 subtasks: namely `GEX to ADT`, `ADT to GEX`, `GEX to ATAC`, and `ATAC to GEX`. Specifically, our methods ranked 3rd in `GEX to ADT` and 4th in `ATAC to GEX`.

Our solution was originally based on an autoencoder architecture that aims to predict the desired modality features given another. We incorporated various strategies of preprocessing such as extracting tf-idf features and filtering highly variable genes/features. We also applied different training techniques such as cycle consistency loss and adversarial training to further minimize the reconstruction errors. For the prediction model used in the contest, we simply ensembled predictions generated from different model architecture and dataset batches by averaging.

Full documentation for the competition, including dataset, can be found at [openproblems.bio/neurips_docs/](https://openproblems.bio/neurips_docs/).

# Table of Contents

- [Getting Started](#getting-started)
- [Folder Sturcture](#folder-sturcture)
- [Training Pipeline](#training-pipeline)
  * [Environment setting](#environment-setting)
  * [Training and Evaluation](#training-and-evaluation)
    + [Model and Data Settings](#model-and-data-settings)
    + [Optimization](#optimization)
    + [Model Architecture](#model-architecture)
    + [Loss Functions](#loss-functions)
    + [Data Preprocessing](#data-preprocessing)
    + [Save or Load Model](#save-or-load-model)
    + [Others](#others)
  * [Preprocessing (optional)](#preprocessing--optional-)

# Getting Started
* Check and download the [Starter kit contents](https://openproblems.bio/neurips_docs/submission/starter_kit_contents/) for a quickguide and script.
* Follow [Quickstart](https://openproblems.bio/neurips_docs/submission/quickstart/) on open problems official page to download desired datasets.
* (Optional) preprocessing:
    ```=bash
    $ cd preprocess/
    ```
    * Save filter gene index
    ```=bash
    $ python save_filter_genes.py
    ```
    * Save highly variable gene index
    ```=bash
    python save_highlyvar_genes.py
    ```
    * Save idf matrix
    ```=bash
    $ python save_idf_matrix.py
    ```
* Train model:
    ```=bash
    $ cd model/
    $ train.py --mode [MODE]
    ```
* (Optional) eval model only:
    ```=bash
    $ cd model/
    $ eval.py --mode [MODE] --checkpoint CHECKPOINT --train eval
    ```

# Folder Sturcture
* Contents in this repository
```
├── LICENSE                                 # MIT License
├── README.md                               # Some information and training guides
├── requirements.txt                        # Requirements packages
├── config.vsh.yaml                         # Viash configuration file
├── script_v10.5.py                         # Script containing current method
├── scripts/                                # Scripts to test, generate, and evaluate a submission
│   ├── 0_sys_checks.sh                     # Checks that necessary software installed
│   ├── 1_unit_test.sh                      # Runs the unit tests in test.py
│   ├── 2_generate_submission.sh            # Generates a submission pkg by running your method on validation data
│   ├── 3_evaluate_submission.sh            # Scores your method locally
│   ├── 4_generate_phase2_submission.sh     # Generates a submission pkg by running your method on test data
│   └── nextflow.config                     # Configurations for running Nextflow locally
├── model/                                  # Full Training/Evaluation pipeline
│   ├── train.py                            # Training/evaluating pipeline
│   ├── eval.py                             # Evaluating pipeline
│   ├── opts.py                             # The opts that model used
│   ├── preprocess/                         # Some preprocess code that can generate required preprocess file
│   │   ├── save_filter_genes.py            # Generate filtered gene index using scanpy package
│   │   ├── save_highlyvar_genes.py         # Generate highly variable gene index using scanpy package
│   │   └── save_idf_matrix.py              # Generate pre-calculated idf matrix
│   ├── idf_matrix/                         # Preprocessed idf matrix 
│   ├── indexs/                             # Preprocessed filtered/highlyvariable gene index file
│   ├── modules                             # Modules that used in training
│   │   └── model_ae.py
│   ├── trainer                             # Trainer of different archetecture
│   │   ├── __init__.py
│   │   ├── trainer_batchgan.py
│   │   ├── trainer_cycle.py
│   │   └── trainer_nn.py
│   ├── utils                               # Utilities folder
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   ├── loss.py
│   │   └── metric.py
│   └── weights_v10                         # Weights file 
└── test.py                                 # Default unit tests
```
* The below content should be download via starterkit / running quickstart
```
├── bin/                                    # Binaries needed to generate a submission
│   ├── check_format
│   ├── nextflow
│   └── viash
├── sample_data/                            # Sample datasets for unit testing and debugging
└── output/
    ├── datasets/                           # Contains phase 1 datasets
    ├── datasets_phase1v2/                  # Contains phase 1 v2 datasets
    └── datasets_phase2/                    # Contains phase 2 datasets
```
# Training Pipeline
## Environment setting
`$ pip install -r requirements.txt`

## Training and Evaluation
`$ cd model`
```
train.py
    --mode {
        gex2atac, gex2adt, adt2gex, atac2gex,
        gex2atac_v2, gex2adt_v2, adt2gex_v2, atac2gex_v2,
        gex2atac_p2, gex2adt_p2, adt2gex_p2, atac2gex_p2
    }
    --train {train, eval}
    [--arch {nn, cycle, batchgan}]
    [--train_batch [TRAIN_BATCH [TRAIN_BATCH ...]]]
    [--test_batch [TEST_BATCH [TEST_BATCH ...]]]
    [--seed SEED]
    [--epoch EPOCH]
    [--batch_size BATCH_SIZE]
    [--learning_rate LEARNING_RATE]
    [--lr_decay_epoch LR_DECAY_EPOCH]
    [--momentum MOMENTUM]
    [--dropout DROPOUT]
    [--hid_dim HID_DIM]
    [--emb_dim EMB_DIM]
    [--reg_loss_weight REG_LOSS_WEIGHT]
    [--rec_loss_weight REC_LOSS_WEIGHT]
    [--norm]
    [--gene_activity]
    [--selection]
    [--mod1_idx_path MOD1_IDX_PATH]
    [--tfidf {0,1,2,3}]
    [--idf_path IDF_PATH]
    [--dryrun]
    [--save_best_from SAVE_BEST_FROM]
    [--checkpoint CHECKPOINT]
    [--note NOTE]
    [--name NAME]
    [--gpu_ids GPU_IDS]
```
### Model and Data Settings
- `--train {train, eval}`
    Training or evaluating the model
    Default: train
- `--mode`
    Direction and dataset using for modality prediction.
    Possible choices: `gex2atac`, `gex2adt`, `adt2gex`, `atac2gex` (phase1 v1 data)
    `_v2`, `_p2` represent phase1 v2 and phase 2 data.
    In phase 1 v1 and v2 dataset, we split dataset into train/test set by batch. In phase 2 dataset, we randomly split the dataset into train/test set by the ratio of 0.9 : 0.1.
    **Reminder**: dataset should be downloaded and placed in the correct directory beforehand.

- `--arch {nn, cycle, batchgan}`
    Desired training architecture. 
    Possible choices: `nn`, `cycle`, `batchgan`
    - `nn`: a simple autoencoder architecture
    - `cycle`: add cycle consistency loss (with a cycle structure) during training
    - `batchgan`: autoencoder architecture with an advarsarial discriminator
    
    Recommended settings: gex2adt = nn/batchgan, others = cycle
    Default: `nn`
    
- `--train_batch`
    Desired training batch. 
    If `None`, the training set will include all batches in training set. 
    In phase 1 v1 and v2 dataset, we split dataset into train/test set manually by batch. In phase 2 dataset, we randomly split the dataset into train/test set by the ratio of 0.9 : 0.1.
    Batch list in default dataset:
    - phase 1 v1: 
    train: s1d1, s2d1, s2d4, s3d6, s3d7
    test: s1d2
    - phase 1 v2:
    train: s1d1, s1d2, s1d3, s2d1, s2d4, s2d5, s3d6, s3d7
    test: s1d2, s3d10
    - phase 2:
    train: s1d1, s1d2, s1d3, s2d1, s2d4, s2d5, s3d1, s3d3, s3d6, s3d7, s3d10
    
    Default: `None`

- `--test_batch`
    Desired testing batch. 
    If `None`, the testing set will include all batches in testing set.
    
    Default: `None`

- `--seed`
    Seed used for reproducibility in spliting phase 2 dataset into train/tes set.
    Useless if training on phase 1 v1 & phase 1 v2 data.
    
    Default: 6666

### Optimization
- `--epoch`, `-e`
    Number of training epochs.
    Recommended settings: nn = 200, cycle = 400, batchgan = 300
    Default: 200
- `batch_size`, `-bs`
    Number of batch size for training and testing.
    Dafault: 2048
- `--learning_rate`, `-lr`
    Starting learning rate.
    Default: 0.1
- `--lr_decay_epoch`
    Decay every lr_decay_epoch.
    Recommended settings: nn = 40, cycle = 80, batchgan = 40
    Default: 40
- `--momentum`
    Momentum factor of SGD optimizer.
    Default: 0.9

### Model Architecture
- `--dropout`, `dp`
    Dropout probability, applied in the autoencoder stacks.
    Default: 0.2
- `--hid_dim`
    The dimension of model hidden layer.
    Default: 1000 
- `--emb_dim`
    The dimension of model embedding feature.
    Default: 128

### Loss Functions
- `--reg_loss_weight`
    L1 regularization loss weight.
    Recommended settings: gex2adt = 1, others = 0
    Default: 0
- `--rec_loss_weight`
    Reconstruction loss weight of A to B model used in `cycle` architecture training.
    Default: 10

### Data Preprocessing
- `--norm`
    True for normalize mod1 input data batch-wise.
    Deafult: False
- `--gene_activity`
    True for use gene activity feature in mod1 input.
    Can be apply only on atac2gex mode (only supports`atac2gex_v2` & `atac2gex_p2` dataset).
    Default: False
- `--selection`
    True for using the selected feature index.
    Should generate the selected index txt file using codes in `preprocess/` beforehand. (`save_filter_genes.py`/`save_highlyvar_genes.py`)

    Default: False
- `--mod1_idx_path`
    The path to mod1 index path (.np), required when selection=True.
    Should generate the selected index using codes in `preprocess/save_idf_matrix.py` beforehand.
    Default: None
- `--tfidf {0, 1, 2, 3}`
    The tfidf mode for mod1 input.
    0: raw data input
    1: tfidf input
    2: concat [raw, tfidf] feature
    3: concat [gene activity, tfidf] feature
    Default: 0
- `--idf_path`
    The path to pre-calculated idf matrix, required if tfidf != 0. 
    Should generate the idf matrix file using code in `preprocess/` beforehand.
    Default: 0

### Save or Load Model
- `--dryrun`
    True: saves weights, logs, runs (tensorboard) during training.
    False: saves runs (tensorboard) only during training.
    Default: False
- `--save_best_from`, `-best`
    Desire epoch that starts to save best model that gets the lowest error from testing set.
    Default: 50
- `--checkpoint`
    Path to pre-trained model checkpoint. Required when evaluating model.
    Default: None

### Others
- `--note`
    Optional notes in log files.
    Default: None
- `--name`
    Name for certain experiments, appears in weight/log/run file. 
    Default: None
- `--gpu_ids`
    GPU ID to be used in training. e.g. 0, use -1 for CPU.
    Default: 0

## Preprocessing (optional)
`$ cd preprocess/`
- `save_filter_genes.py`
    - Generate filtered genes/features index using scanpy package.
    - parameters:
        `DATAPTH`: path to data
        `PCT`: percentage of filter gene
    - outputs:
        `index_{PCT}pct.txt`: the filtered genes/features index TXT file

- `save_highlyvar_genes.py`
    - Generate highly variable genes/features index using scanpy package.
    - parameters:
        `DATAPTH`: path to data
        `NTOP`: number of n top genes
    - outputs:
        `index_hoghly{NTOP}.txt`: the highly variable genes/features index TXT file
- `save_idf_matrix.py`
    - Generate pre-calculated mod1 idf matrixs file from the dataset.
    - parameters:
        `DATAPTH`: path to data
    - outputs:
        `mod1_idf.npy`: the mod1 idf matrix NPY file
