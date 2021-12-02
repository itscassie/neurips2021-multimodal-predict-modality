""" this code select important feature index from dataset from a data persepective"""
import logging
import argparse
from datetime import datetime

from trainer.trainer_concrete import TrainProcess


from opts import DATASET, model_opts
from utils.dataloader import get_data_dim

# config parser
parser = argparse.ArgumentParser(add_help=False)
model_opts(parser)
args = parser.parse_known_args()[0]

TIME_NOW = datetime.now().strftime("%b%d-%H-%M")
LOG_FILE = f"../../logs/concrete_{args.mode}_{args.name}_{TIME_NOW}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)

# load data
MOD1_DIM, MOD2_DIM = get_data_dim(DATASET[args.mode], args)
parser.add_argument("--mod1_dim", default=MOD1_DIM)
parser.add_argument("--mod2_dim", default=MOD2_DIM)
args = parser.parse_args()

logging.info("\nArgument:")
for arg, value in vars(args).items():
    logging.info(f"{arg:20s}: {value}")
logging.info("\n")


trainer = TrainProcess(args)
trainer.run()
