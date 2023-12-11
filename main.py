import logging
import os
import sys

import torch
from torch_geometric import seed_everything
from exp.exp_attack import ExpAttack
from exp.exp_megu import ExpMEGU
from parameter_parser import parameter_parser
import warnings

warnings.filterwarnings("ignore")
seed_everything(2019816)


def config_logger(save_name):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')

    # create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


if __name__ == "__main__":
    args = parameter_parser()

    # config the logger
    logger_name = "_".join((args['method'], args['target_model'], args['dataset_name'],
                            args['unlearn_task'], str(args['unlearn_ratio'])))
    config_logger(logger_name)
    logging.info(logger_name)

    torch.set_num_threads(args["num_threads"])
    torch.cuda.set_device(args["cuda"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args["exp"].lower() == "unlearn":
        if args["method"].lower() == "megu":
            ExpMEGU(args)
        else:
            raise NotImplementedError
    elif args["exp"].lower() == "attack":
        ExpAttack(args)
