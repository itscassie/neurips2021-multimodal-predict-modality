import logging
import argparse
import numpy as np
import anndata as ad
from datetime import datetime
from scipy.sparse import csc_matrix

from modules.trainer_nn import TrainProcess as TrainProcess_NN
from modules.trainer_pairae import TrainProcess as TrainProcess_PairAE
from modules.trainer_residual import TrainProcess as TrainProcess_Res
from modules.trainer_pix2pix import TrainProcess as TrainProcess_Pix2Pix
from modules.trainer_cycle import TrainProcess as TrainProcess_Cycle
from modules.trainer_scvi import TrainProcess as TrainProcess_SCVI
from modules.trainer_rec import TrainProcess as TrainProcess_REC

from opts import DATASET, model_opts
from utils.dataloader import get_data_dim

# config parser
parser = argparse.ArgumentParser(add_help=False)
model_opts(parser)
args = parser.parse_known_args()[0]

time_now = datetime.now().strftime('%b%d-%H-%M')
exp_name = f'{args.arch}_{args.mode}_{args.name}_{time_now}'

if args.dryrun:
    handlers = [logging.StreamHandler()]
else:
    handlers = [
        logging.FileHandler(f"../../logs/train_{exp_name}.log", mode='w'),
        logging.StreamHandler()
        ]

logging.basicConfig(level=logging.DEBUG, format='%(message)s', handlers=handlers)

# load data
mod1_dim, mod2_dim = get_data_dim(DATASET[args.mode])
parser.add_argument('--mod1_dim', default=mod1_dim)
parser.add_argument('--mod2_dim', default=mod2_dim)
parser.add_argument('--exp_name', default=exp_name)
args = parser.parse_args()


logging.info("\nArgument:")
for arg, value in vars(args).items():
    logging.info(f"{arg:20s}: {value}")
logging.info("\n")

if args.arch in ['nn', 'unb_ae', 'decoder', 'kernelae']:
    trainer = TrainProcess_NN(args)
elif args.arch == 'pairae':
    trainer = TrainProcess_PairAE(args)
elif args.arch == 'residual':
    trainer = TrainProcess_Res(args)
elif args.arch == 'pix2pix':
    trainer = TrainProcess_Pix2Pix(args)
elif args.arch == 'cycle':
    trainer = TrainProcess_Cycle(args)
elif args.arch == 'scvi':
    trainer = TrainProcess_SCVI(args)
elif args.arch == 'rec':
    trainer = TrainProcess_REC(args)

trainer.run()
trainer.eval()