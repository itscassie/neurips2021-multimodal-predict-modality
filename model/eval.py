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
        logging.FileHandler(f"logs/eval_{args.arch}_{args.mode}_{datetime.now().strftime('%b%d-%H-%M')}.log", mode='w'),
        logging.StreamHandler()
        ]
    )

mod1_dim, mod2_dim = get_data_dim(DATASET[args.mode])
parser.add_argument('--mod1_dim', default=mod1_dim)
parser.add_argument('--mod2_dim', default=mod2_dim)
args = parser.parse_args()


logging.info("\nArgument:")
for arg, value in vars(args).items():
    logging.info(f"{arg:20s}: {value}")
logging.info("\n")

if args.arch in ['nn', 'unb_ae', 'decoder']:
    trainer = TrainProcess_NN(args)
elif args.arch == 'pairae':
    trainer = TrainProcess_PairAE(args)
elif args.arch == 'residual':
    trainer = TrainProcess_Res(args)
elif args.arch == 'pix2pix':
    trainer = TrainProcess_Pix2Pix(args)
elif args.arch == 'cycle':
    trainer = TrainProcess_Cycle(args)

trainer.load_checkpoint()
trainer.eval()