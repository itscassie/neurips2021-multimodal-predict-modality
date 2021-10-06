import logging
import argparse
import numpy as np
import anndata as ad
from datetime import datetime
from scipy.sparse import csc_matrix

from trainer_nn import TrainProcess as TrainProcess_NN
from trainer_pairae import TrainProcess as TrainProcess_PairAE
from trainer_residual import TrainProcess as TrainProcess_Res

from opts import DATASET, model_opts
from dataloader import get_data_dim

# config parser
parser = argparse.ArgumentParser(add_help=False)
model_opts(parser)
args = parser.parse_known_args()[0]

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(message)s',
    handlers=[
        logging.FileHandler(f"logs/train_{args.arch}_{args.mode}.log", mode='w'),
        logging.StreamHandler()
        ]
    )

# load data
mod1_dim, mod2_dim = get_data_dim(DATASET[args.mode])
parser.add_argument('--mod1_dim', default=mod1_dim)
parser.add_argument('--mod2_dim', default=mod2_dim)
args = parser.parse_args()


logging.info("\nArgument:")
for arg, value in vars(args).items():
    logging.info(f"{arg:20s}: {value}")
logging.info("\n")

if args.arch == 'nn':
    trainer = TrainProcess_NN(args)
elif args.arch == 'pairae':
    trainer = TrainProcess_PairAE(args)
elif args.arch == 'residual':
    trainer = TrainProcess_Res(args)

trainer.run()
trainer.eval()