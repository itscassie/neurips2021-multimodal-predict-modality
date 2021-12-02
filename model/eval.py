""" this code eval pre-trained models """
import logging
import argparse
from datetime import datetime

from opts import DATASET, model_opts
from utils.dataloader import get_data_dim

from trainer.trainer_nn import TrainProcess as TrainProcess_NN
from trainer.trainer_pairae import TrainProcess as TrainProcess_PairAE
from trainer.trainer_residual import TrainProcess as TrainProcess_Res
from trainer.trainer_pix2pix import TrainProcess as TrainProcess_Pix2Pix
from trainer.trainer_cycle import TrainProcess as TrainProcess_Cycle
from trainer.trainer_scvi import TrainProcess as TrainProcess_SCVI
from trainer.trainer_rec import TrainProcess as TrainProcess_REC


# config parser
parser = argparse.ArgumentParser(add_help=False)
model_opts(parser)
args = parser.parse_known_args()[0]

# exp name for log, weights, model
TIME_NOW = datetime.now().strftime("%b%d-%H-%M")
exp_name = f"{args.arch}_{args.mode}"
if args.selection:
    assert args.mod1_idx_path is not None, "need to specified --mod1_idx_path"
    SELECT_NUM = (
        args.mod1_idx_path.split("/")[-1].replace(".txt", "").replace("index_", "")
    )
    exp_name += f"_select{SELECT_NUM}"
if args.tfidf == 1:
    exp_name += f"_tfidf"
elif args.tfidf == 2:
    exp_name += f"_tfidfconcat"
if args.name != "":
    exp_name += f"_{args.name}"
exp_name += f"_{TIME_NOW}"

# loggings and logs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[
        logging.FileHandler(f"../../logs/eval_{exp_name}.log", mode="w"),
        logging.StreamHandler(),
    ],
)

# load data
MOD1_DIM, MOD2_DIM = get_data_dim(DATASET[args.mode], args)

parser.add_argument("--mod1_dim", default=MOD1_DIM)
parser.add_argument("--mod2_dim", default=MOD2_DIM)
parser.add_argument("--exp_name", default=exp_name)
args = parser.parse_args()


logging.info("\nArgument:")
for arg, value in vars(args).items():
    logging.info(f"{arg:20s}: {value}")
logging.info("\n")

if args.arch in ["nn", "unb_ae", "decoder", "kernelae"]:
    trainer = TrainProcess_NN(args)
elif args.arch == "pairae":
    trainer = TrainProcess_PairAE(args)
elif args.arch == "residual":
    trainer = TrainProcess_Res(args)
elif args.arch == "pix2pix":
    trainer = TrainProcess_Pix2Pix(args)
elif args.arch == "cycle":
    trainer = TrainProcess_Cycle(args)
elif args.arch == "scvi":
    trainer = TrainProcess_SCVI(args)
elif args.arch == "rec":
    trainer = TrainProcess_REC(args)


trainer.load_checkpoint()
trainer.eval()
