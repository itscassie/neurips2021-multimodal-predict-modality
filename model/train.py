""" main training process """
import logging
import argparse
from datetime import datetime

from trainer.trainer_nn import TrainProcess as TrainProcess_NN
from trainer.trainer_cycle import TrainProcess as TrainProcess_Cycle
from trainer.trainer_batchgan import TrainProcess as TrainProcess_BATCHGAN
from trainer.trainer_pix2pix import TrainProcess as TrainProcess_Pix2Pix
from trainer.trainer_rec import TrainProcess as TrainProcess_REC
from trainer.trainer_scvi import TrainProcess as TrainProcess_SCVI
from trainer.trainer_peakvi import TrainProcess as TrainProcess_PEAKVI
from trainer.trainer_pairae import TrainProcess as TrainProcess_PairAE
from trainer.trainer_residual import TrainProcess as TrainProcess_Res

from opts import DATASET, model_opts
from utils.dataloader import get_data_dim

if __name__ == "__main__":
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
    elif args.tfidf == 3:
        exp_name += f"_tfidfconcatga"
    elif args.gene_activity:
        exp_name += f"_ga"
    if args.norm:
        exp_name += f"_norm"
    if args.dropout != 0.2:
        exp_name += f"_dropout{args.dropout}"
    if args.name != "":
        exp_name += f"_{args.name}"
    exp_name += f"_{TIME_NOW}"

    # loggings and logs
    if args.dryrun:
        handlers = [logging.StreamHandler()]
    else:
        handlers = [
            logging.FileHandler(f"../../logs/train_{exp_name}.log", mode="w"),
            logging.StreamHandler(),
        ]

    logging.basicConfig(level=logging.DEBUG, format="%(message)s", handlers=handlers)

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

    if args.arch == "nn":
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
    elif args.arch in ["rec", "peakrec", "scvirec"]:
        trainer = TrainProcess_REC(args)
    elif args.arch == "peakvi":
        trainer = TrainProcess_PEAKVI(args)
    elif args.arch == "batchgan":
        trainer = TrainProcess_BATCHGAN(args)

    trainer.run()

    if args.mode in ["gex2atac", "gex2adt", "adt2gex", "atac2gex"]:
        trainer.eval()
