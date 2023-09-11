from copy import deepcopy
from typing import List, Optional, Union

import numpy as np

from yacs.config import CfgNode as CN

DEFAULT_CONFIG_DIR = "config/"
CONFIG_FILE_SEPARATOR = ","


# ----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# ----------------------------------------------------------------------------
_C = CN()
_C.DEVICE = [0, 1]
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/tac"
_C.CHECKPOINT_DIR = "data/checkpoints/tac"
_C.EVAL_PREFIX = "valseen"
_C.LOG_DIR = "debug"
_C.ENVIRONMENTS = ["TOKENIZERS_PARALLELISM", "false"]
_C.DEBUG = True  # If True: reduce training and evaluation samples
_C.SEED = 0


_C.TRAINER = CN()
_C.TRAINER.name = "DistTrainer"
_C.TRAINER.optimizer = "AdamW"
_C.TRAINER.adam_epsilon = 1e-6
_C.TRAINER.weight_decay = 1e-4
_C.TRAINER.learning_rate = 1e-4
_C.TRAINER.batch_size = 128  # Total batch size (containing distributed)
_C.TRAINER.epochs = 5
_C.TRAINER.loss_window = 100
_C.TRAINER.save_strategies = ["epoch", "step"]  # ["epoch","step","best"]
_C.TRAINER.save_steps = 20000  #
_C.TRAINER.save_best = "loss"  #
_C.TRAINER.loaders = ["DistRGBDLoader"]  # a loader corrensponding to a DATA.{DATASET}
_C.TRAINER.eval_metrics = ["BLEU"]

_C.DATA = CN()
_C.DATA.RGBD = CN()

#########################################################################################
# train data
#########################################################################################
_C.DATA.RGBD.data_path = []
_C.DATA.RGBD.scale_value = []
_C.DATA.RGBD.time_factor = []
_C.DATA.RGBD.is_resized = True
# shuffle the data by blocks (enable generating samples with the same path)
# the final block size is the minimum between block_size and batch_size/2
_C.DATA.RGBD.block_size = 8

## add hm3d
# for i in range(30): # fullsize is 52
#     _C.DATA.RGBD.data_path.append(
#         "/root/TAC/data/rgbd_data/hm3d_rgbd/train/{}".format(i)
#     )
#     _C.DATA.RGBD.scale_value.append(1000.0)
#     _C.DATA.RGBD.time_factor.append(87.56)  # 9 images
## add sun3d
# _C.DATA.RGBD.data_path.append("/root/TAC/data/rgbd_data/sun3d/train")
# _C.DATA.RGBD.scale_value.append(8000.0)
# _C.DATA.RGBD.time_factor.append(42.96)  # 43 images, half the original value
## add scenenet
# for i in range(16): # full size is 17
#     _C.DATA.RGBD.data_path.append(
#         "/root/TAC/data/rgbd_data/scenenet_resize/train/{}".format(i)
#     )
#     _C.DATA.RGBD.scale_value.append(1000.0)
#     _C.DATA.RGBD.time_factor.append(278.17)  # 11 images
## add diode
for split in ["indoors", "outdoor"]:
    _C.DATA.RGBD.data_path.append(
        "/root/TAC/data/rgbd_data/diode_clean_resize/train/{}".format(split)
    )
    _C.DATA.RGBD.scale_value.append(1000.0)
    _C.DATA.RGBD.time_factor.append(36.60)  # 4 images
## add tum
_C.DATA.RGBD.data_path.append("/root/TAC/data/rgbd_data/tumrgbd_clean_resize/train")
_C.DATA.RGBD.scale_value.append(5000.0)
_C.DATA.RGBD.time_factor.append(1416.31)  # 42 images half the original value
## add outdoor, only for outdoor tune
# _C.DATA.RGBD.data_path.append("/root/TAC/data/rgbd_data/outdoor_train/diml_resize/outdoor")
# _C.DATA.RGBD.scale_value.append(1000.0)
# _C.DATA.RGBD.time_factor.append(1.0)
# _C.DATA.RGBD.data_path.append("/root/TAC/data/rgbd_data/outdoor_train/rgbd1k_resize/outdoor")
# _C.DATA.RGBD.scale_value.append(200.0)
# _C.DATA.RGBD.time_factor.append(1.0)

#########################################################################################
# val data
#########################################################################################
# Eval config
_C.DATA.RGBD.EVAL = CN()
_C.DATA.RGBD.EVAL.is_resized = True
_C.DATA.RGBD.EVAL.shuffle = "noshuffle"
_C.DATA.RGBD.EVAL.block_size = 10
_C.DATA.RGBD.EVAL.seed = 25
_C.DATA.RGBD.EVAL.data_path = []
_C.DATA.RGBD.EVAL.scale_value = []
_C.DATA.RGBD.EVAL.time_factor = []
## add hm3d
_C.DATA.RGBD.EVAL.data_path.append("/root/TAC/data/rgbd_data/pretrain_val/hm3d_val")
_C.DATA.RGBD.EVAL.scale_value.append(1000.0)
_C.DATA.RGBD.EVAL.time_factor.append(87.56)
## add sun3d
_C.DATA.RGBD.EVAL.data_path.append("/root/TAC/data/rgbd_data/pretrain_val/sun3d_val")
_C.DATA.RGBD.EVAL.scale_value.append(8000.0)
_C.DATA.RGBD.EVAL.time_factor.append(42.96)
## add scenenet
_C.DATA.RGBD.EVAL.data_path.append(
    "/root/TAC/data/rgbd_data/pretrain_val/scenenet_val500"
)
_C.DATA.RGBD.EVAL.scale_value.append(1000.0)
_C.DATA.RGBD.EVAL.time_factor.append(278.17)
## add diode
_C.DATA.RGBD.EVAL.data_path.append("/root/TAC/data/rgbd_data/pretrain_val/diode_val")
_C.DATA.RGBD.EVAL.scale_value.append(1000.0)
_C.DATA.RGBD.EVAL.time_factor.append(36.60)
## add tum
_C.DATA.RGBD.EVAL.data_path.append("/root/TAC/data/rgbd_data/pretrain_val/tum_val")
_C.DATA.RGBD.EVAL.scale_value.append(5000.0)
_C.DATA.RGBD.EVAL.time_factor.append(1416.31)
## add scannet, only for out-of-domain
# _C.DATA.RGBD.EVAL.data_path.append('/root/TAC/data/rgbd_data/pretrain_val/scannet_val')
# _C.DATA.RGBD.EVAL.scale_value.append(1000.0)
# _C.DATA.RGBD.EVAL.time_factor.append(1)

_C.MODEL = CN()
_C.MODEL.name = "TAC"
_C.MODEL.modalities = ["image", "depth"]
_C.MODEL.text_len = 140
_C.MODEL.load_from_ckpt = False
_C.MODEL.ckpt_path = ""
# CLIP: CLIP, TAC, MIX
# Contrastive: C_L2, C_COS
# Triplet: T_L2, T_COS
_C.MODEL.loss_type = "CLIP"
_C.MODEL.negative_weight = 2.0
_C.MODEL.focal_gamma = 0.25
_C.MODEL.loss_margin = 0.5
_C.MODEL.block_size = min(_C.DATA.RGBD.block_size, _C.TRAINER.batch_size // 2)
_C.MODEL.learnable_time_scale = False
_C.MODEL.init_time_scale = 1 / 6
_C.MODEL.use_gaussian_coef = False

_C.MODEL.bottleneck = "vit"
_C.MODEL.IMAGE = CN()
_C.MODEL.IMAGE.model_name = "openai/clip-vit-base-patch32"
_C.MODEL.IMAGE.feature_dim = 768
_C.MODEL.IMAGE.projection_dim = 512
_C.MODEL.IMAGE.trainable = True


_C.MODEL.DEPTH = CN()
_C.MODEL.DEPTH.model_name = "openai/clip-vit-base-patch32"
_C.MODEL.DEPTH.feature_dim = 768
_C.MODEL.DEPTH.projection_dim = 512
_C.MODEL.DEPTH.trainable = True


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    """Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
