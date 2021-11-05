import logging
import argparse
import numpy as np
import anndata as ad
from datetime import datetime

from modules.trainer_concrete import TrainProcess


from opts import DATASET, model_opts
from utils.dataloader import get_data_dim

# config parser
parser = argparse.ArgumentParser(add_help=False)
model_opts(parser)
args = parser.parse_known_args()[0]

time_now = datetime.now().strftime('%b%d-%H-%M')
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(message)s',
    handlers=[
        logging.FileHandler(f"../../logs/concrete_{args.mode}_{args.name}_{time_now}.log", mode='w'),
        logging.StreamHandler()
        ]
    )

# load data
mod1_dim, mod2_dim = get_data_dim(DATASET[args.mode], args)
parser.add_argument('--mod1_dim', default=mod1_dim)
parser.add_argument('--mod2_dim', default=mod2_dim)
args = parser.parse_args()

logging.info("\nArgument:")
for arg, value in vars(args).items():
    logging.info(f"{arg:20s}: {value}")
logging.info("\n")


trainer = TrainProcess(args)
trainer.run()