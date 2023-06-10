import os
import argparse
import random
from datetime import datetime
import numpy as np
import torch

from common.registry import registry
from common.logger import logger
from config.default import get_config
import imports
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "generate_eval_order"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(config: str, mode: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        config: path to config file.
        mode: "train" or "eval".
        opts: list of strings of additional config options.
    """
    torch.autograd.set_detect_anomaly(True)
    config = get_config(config, opts)
    logdir = config.LOG_DIR
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        logger.add_filehandler(
            os.path.join(
                config.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
            )
        )

    # seed = random.randrange(0x7F7F7F7F)
    seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if config.ENVIRONMENTS:
        assert len(config.ENVIRONMENTS) % 2 == 0, "Wrong number of environments!"
        for i in range(0, len(config.ENVIRONMENTS), 2):
            os.environ[config.ENVIRONMENTS[i]] = config.ENVIRONMENTS[i + 1]
    if config.DEVICE:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(v) for v in config.DEVICE])

    if (
        len(config.DEVICE) > 1 and mode == "train"
    ):  # Distributed condition: training and multiple config.DEVICE
        dist.init_process_group("nccl")
    else:  # single
        os.environ["LOCAL_RANK"] = "0"
    trainer_cls = registry.get_trainer(config.TRAINER.name)
    assert trainer_cls is not None, f"{config.TRAINER.name} is not supported"
    trainer = trainer_cls.from_config(config)

    if mode == "train":
        trainer.train()
    elif mode == "eval":
        trainer.eval()
    elif mode == "generate_eval_order":
        trainer.generate_eval_order()


if __name__ == "__main__":
    main()
