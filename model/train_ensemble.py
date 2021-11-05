import logging
import argparse
import numpy as np
import anndata as ad
from datetime import datetime
from scipy.sparse import csc_matrix

from modules.trainer_ensemble import TrainProcess as TrainProcess
from opts import DATASET, model_opts
from utils.dataloader import get_data_dim, get_processed_dim

# config parser
parser = argparse.ArgumentParser(add_help=False)
model_opts(parser)
# model A
parser.add_argument("--modelA", type=str, default=None)
parser.add_argument("--A_tfidf", type=int, default=0, choices=[0, 1, 2], 
    help='0: raw data input, 1: tfidf input, 2: concat [raw, tfidf] data input')
parser.add_argument("--A_selection", action="store_true")
# model B
parser.add_argument("--modelB", type=str, default=None)
parser.add_argument("--B_tfidf", type=int, default=0, choices=[0, 1, 2], 
    help='0: raw data input, 1: tfidf input, 2: concat [raw, tfidf] data input')
parser.add_argument("--B_selection", action="store_true")
args = parser.parse_known_args()[0]

# exp name for log, weights, model
time_now = datetime.now().strftime('%b%d-%H-%M')
exp_name = f'{args.mode}'
if args.selection:
    assert args.mod1_idx_path != None, "need to specified --mod1_idx_path"
    exp_name += f'_select{args.mod1_idx_path.split("/")[-1].replace(".txt", "").replace("index_", "")}'
if args.tfidf == 1:
    exp_name += f'_tfidf'
elif args.tfidf == 2:
    exp_name += f'_tfidfconcat'
if args.name != "":
    exp_name += f'_{args.name}'
exp_name += f'_{time_now}'

# loggings and logs
if args.dryrun:
    handlers = [logging.StreamHandler()]
else:
    handlers = [
        logging.FileHandler(f"../../logs/weighttedavg_{exp_name}.log", mode='w'),
        logging.StreamHandler()
        ]

logging.basicConfig(level=logging.DEBUG, format='%(message)s', handlers=handlers)

# load data
# make selection=False, tfidf=0 in this stage to get the original mod1_dim
mod1_dim, mod2_dim = get_data_dim(DATASET[args.mode], args)
# update mod1 dim
A_mod1_dim = get_processed_dim(mod1_dim, args, selection=args.A_selection, tfidf=args.A_tfidf)
B_mod1_dim = get_processed_dim(mod1_dim, args, selection=args.B_selection, tfidf=args.B_tfidf)

parser.add_argument('--A_mod1_dim', default=A_mod1_dim)
parser.add_argument('--B_mod1_dim', default=B_mod1_dim)
parser.add_argument('--mod2_dim', default=mod2_dim)
parser.add_argument('--exp_name', default=exp_name)
args = parser.parse_args()

logging.info("\nArgument:")
for arg, value in vars(args).items():
    logging.info(f"{arg:20s}: {value}")
logging.info("\n")


trainer = TrainProcess(args)

trainer.run()
trainer.eval()