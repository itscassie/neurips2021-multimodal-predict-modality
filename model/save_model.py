""" this code save the single AtoB part of a cycle model weights"""

import logging
import argparse

from opts import DATASET, model_opts
from utils.dataloader import get_data_dim
from trainer.trainer_cycle import TrainProcess as TrainProcess_Cycle

if __name__ == "__main__":

    # config parser
    parser = argparse.ArgumentParser(add_help=False)
    model_opts(parser)
    args = parser.parse_known_args()[0]

    LOG_FILE = (
        f"../../logs/{args.checkpoint.replace('../', '').replace('weights/', '')}.log"
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
    )

    MOD1_DIM, MOD2_DIM = get_data_dim(DATASET[args.mode], args)
    parser.add_argument("--mod1_dim", default=MOD1_DIM)
    parser.add_argument("--mod2_dim", default=MOD2_DIM)
    parser.add_argument("--exp_name", default="save_model")
    args = parser.parse_args()

    logging.info("\nArgument:")
    for arg, value in vars(args).items():
        logging.info(f"{arg:20s}: {value}")
    logging.info("\n")

    if args.arch == "cycle":
        trainer = TrainProcess_Cycle(args)
    else:
        raise BaseException("NOT IMPLEMENTED")

    trainer.load_checkpoint()
    trainer.eval()
    trainer.save_AtoB()
