import logging
import argparse
import numpy as np
import anndata as ad
from datetime import datetime
from scipy.sparse import csc_matrix

from opts import DATASET, model_opts
from utils.dataloader import get_data_dim

from modules.trainer_nn import TrainProcess as TrainProcess_NN
from modules.trainer_pairae import TrainProcess as TrainProcess_PairAE
from modules.trainer_residual import TrainProcess as TrainProcess_Res
from modules.trainer_pix2pix import TrainProcess as TrainProcess_Pix2Pix
from modules.trainer_cycle import TrainProcess as TrainProcess_Cycle

# config parser
parser = argparse.ArgumentParser(add_help=False)
model_opts(parser)
args = parser.parse_known_args()[0]


logging.basicConfig(
    level=logging.DEBUG, 
    format='%(message)s',
    handlers=[
        logging.FileHandler(f"../../logs/{args.checkpoint.replace('../', '').replace('weights/', '')}.log", mode='w'),
        logging.StreamHandler()
    ]
)

mod1_dim, mod2_dim = get_data_dim(DATASET[args.mode], args)
parser.add_argument('--mod1_dim', default=mod1_dim)
parser.add_argument('--mod2_dim', default=mod2_dim)
parser.add_argument('--exp_name', default="save_model")
args = parser.parse_args()


logging.info("\nArgument:")
for arg, value in vars(args).items():
    logging.info(f"{arg:20s}: {value}")
logging.info("\n")

if args.arch == 'cycle':
    trainer = TrainProcess_Cycle(args)
else:
    raise BaseException("NOT IMPLEMENTED")

trainer.load_checkpoint()
trainer.eval()
trainer.save_AtoB()