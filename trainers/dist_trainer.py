import os
from datetime import datetime
import time
import json
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from trainers.base_trainer import BaseTrainer
from common.registry import registry
from common.logger import logger
from common.utils import get_checkpoint_id, poll_checkpoint_folder

# from common.scorer import Scorer
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast

import torch.distributed as dist
import transformers


@registry.register_trainer
class DistTrainer(BaseTrainer):
    """Distributed version"""

    def __init__(self, config) -> None:
        self.config = config
        self.rank = int(os.environ["LOCAL_RANK"])
        if self.config.DEVICE:
            self.device_id = self.rank % len(config.DEVICE)
        else:
            self.device_id = torch.device("cpu")
        logger.debug(f"Start running a trainer on rank {self.rank}.")
        if self.rank > 0:
            transformers.logging.set_verbosity_error()
        if self._is_rank0:
            logger.debug(f"Config: {config}")
            logger.debug(f"Random seed: {config.SEED}")
        # self.initialize()

    @property
    def _is_distributed(self):
        return len(self.config.DEVICE) > 1

    @property
    def _is_rank0(self):
        return self.rank == 0

    def initialize(self, mode="train"):
        ## Model
        model_cls = registry.get_model(self.config.MODEL.name)
        self.model = model_cls.from_config(self.config)
        if self.config.MODEL.load_from_ckpt:
            ckpt = torch.load(self.config.MODEL.ckpt_path)
            self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device_id)
        if self._is_distributed and mode == "train":
            self.model = DDP(
                self.model,
                device_ids=[self.device_id],
                output_device=self.device_id,
                find_unused_parameters=True,
            )
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        ## Optimizer
        if mode == "train":
            no_decay = ["bias", "LayerNorm.weight", "temperature", "time_scale"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.config.TRAINER.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.TRAINER.learning_rate,
                eps=self.config.TRAINER.adam_epsilon,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.TRAINER.epochs
            )
        ## Data loader, load different sub-sets accroding to the mode
        loader_cls = registry.get_dataloader(self.config.TRAINER.loaders[0])
        self.loader = loader_cls(
            self.config,
            mode=mode,
            num_workers=6,
        )
        ## Folder preparation
        if mode == "train":
            os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
            if "step" in self.config.TRAINER.save_strategies:
                os.makedirs(
                    os.path.join(self.config.CHECKPOINT_DIR, "step"), exist_ok=True
                )
            if "best" in self.config.TRAINER.save_strategies:
                os.makedirs(
                    os.path.join(self.config.CHECKPOINT_DIR, "best"), exist_ok=True
                )
        else:
            os.makedirs(
                os.path.join(self.config.CHECKPOINT_DIR, "evals"), exist_ok=True
            )

    def save_checkpoint(
        self, file_name: str, checkpoint_dir=None, mean_loss=-1
    ) -> None:
        """Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint
        """
        checkpoint = {
            "state_dict": self.model.module.state_dict()
            if self._is_distributed
            else self.model.state_dict(),
            "config": self.config,
            "mean_loss": mean_loss,
        }
        if checkpoint_dir is None:
            checkpoint_dir = self.config.CHECKPOINT_DIR
        torch.save(checkpoint, os.path.join(checkpoint_dir, file_name))

    def log_model_info(self):
        tmp_model = self.model.module if self._is_distributed else self.model
        params = sum(param.numel() for param in tmp_model.parameters())
        params_t = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
        params_d = sum(
            param.numel() for param in tmp_model.depth_transformer.parameters()
        )
        logger.debug(
            f"Agent parameters: {params}. Trainable: {params_t}. Depth encoder: {params_d}"
        )
        params_name = [k for k, p in tmp_model.named_parameters() if p.requires_grad]
        logger.debug(f"Trainable names: {params_name}")

    def train(self):
        self.initialize(mode="train")
        if self._is_rank0:
            self.log_model_info()
            writer = SummaryWriter(
                os.path.join(
                    self.config.TENSORBOARD_DIR,
                    self.config.MODEL.name + datetime.now().strftime("_%Y%m%d-%H%M%S"),
                )
            )
        iter_num = 0
        losses = []
        min_loss = 0x7F7F7F7F
        loss_window = self.config.TRAINER.loss_window
        epochs_iter = (
            tqdm.trange(self.config.TRAINER.epochs, dynamic_ncols=True)
            if self._is_rank0
            else range(self.config.TRAINER.epochs)
        )
        for epoch in epochs_iter:
            # if self._is_distributed:
            self.loader.sampler.set_epoch(epoch)
            batch_num = len(self.loader.dataset) // self.config.TRAINER.batch_size
            batch_bar = (
                tqdm.tqdm(
                    self.loader,
                    total=batch_num,
                    leave=False,
                    dynamic_ncols=True,
                )
                if self._is_rank0
                else self.loader
            )
            for batch in batch_bar:
                batch = {
                    k: (v.to(self.device_id) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                outputs = self.model(batch)
                loss = outputs["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self._is_distributed:
                    self.model.module.clamp_param()
                else:
                    self.model.clamp_param()

                losses.append(loss.item())
                smooth_loss = np.mean(
                    losses[max(0, len(losses) - loss_window) : len(losses)]
                )
                if (
                    "best" in self.config.TRAINER.save_strategies
                    and smooth_loss < min_loss
                    and self._is_rank0
                ):
                    # save the best checkpoint and the smooth loss
                    min_loss = smooth_loss
                    self.save_checkpoint(
                        f"ckpt.{self.config.MODEL.name}.best.0.pth",
                        os.path.join(self.config.CHECKPOINT_DIR, "best"),
                        mean_loss=smooth_loss,
                    )

                if self._is_rank0:
                    batch_bar.set_description(f"E {epoch}.")
                    batch_bar.set_postfix(
                        {
                            "loss": "%2.4f" % (loss.item()),
                            "smooth": "%2.4f" % (smooth_loss),
                        }
                    )
                    writer.add_scalar("loss/total", loss, iter_num)
                    writer.add_scalar("loss/smooth", smooth_loss, iter_num)
                if (
                    "step" in self.config.TRAINER.save_strategies
                    and iter_num % self.config.TRAINER.save_steps == 0
                    and self._is_rank0
                ):
                    # save step checkpoints and the mean loss
                    self.save_checkpoint(
                        f"ckpt.{self.config.MODEL.name}.{iter_num:0>6d}.pth",
                        os.path.join(self.config.CHECKPOINT_DIR, "step"),
                        mean_loss=np.mean(
                            losses[
                                max(
                                    0, len(losses) - self.config.TRAINER.save_steps
                                ) : len(losses)
                            ]
                        ),
                    )
                iter_num += 1
            self.scheduler.step()
            if "epoch" in self.config.TRAINER.save_strategies and self._is_rank0:
                # save epoch checkpoints and the mean loss
                self.save_checkpoint(
                    f"ckpt.{self.config.MODEL.name}.{epoch:0>2d}.pth",
                    mean_loss=np.mean(
                        losses[max(0, len(losses) - batch_num) : len(losses)]
                    ),
                )

    def eval(self):
        writer = SummaryWriter(
            os.path.join(
                self.config.TENSORBOARD_DIR,
                self.config.MODEL.name + datetime.now().strftime("_%Y%m%d-%H%M%S_eval"),
            )
        )
        if os.path.isfile(self.config.CHECKPOINT_DIR):
            logger.info(f"Evaluating {self.config.CHECKPOINT_DIR}")
            current_ckpt = self.config.CHECKPOINT_DIR
            index = get_checkpoint_id(current_ckpt)
            self.config.defrost()
            self.config.CHECKPOINT_DIR = os.path.dirname(current_ckpt)
            self.config.MODEL.load_from_ckpt = True
            self.config.MODEL.ckpt_path = current_ckpt
            self.config.freeze()
            self.initialize(mode="eval")
            res = self._eval_model(self.model, index=index)
            # write to tensorboard
            for k, v in res.items():
                writer.add_scalar(f"eval/{k}", v, index)

            # write to json
            current_ckpt = os.path.basename(current_ckpt)
            current_ckpt = (
                current_ckpt.replace("ckpt.", "").replace(".pth", "").replace(".", "_")
                + ".json"
            )
            if self.config.EVAL_PREFIX:
                current_ckpt = self.config.EVAL_PREFIX + "_" + current_ckpt
            result_file = os.path.join(
                self.config.CHECKPOINT_DIR, "evals", current_ckpt
            )
            with open(result_file, "w") as f:
                f.write(json.dumps(res))
        else:
            # evaluate multiple checkpoints in order
            prev_ckpt_ind = -1
            time_cnt = 0
            while True:
                current_ckpt = None
                while current_ckpt is None:
                    current_ckpt = poll_checkpoint_folder(
                        self.config.CHECKPOINT_DIR, prev_ckpt_ind
                    )
                    time.sleep(1)
                    time_cnt += 1
                    if time_cnt >= 3:
                        break
                if time_cnt >= 3:
                    break
                logger.info(f"Evaluating {current_ckpt}")
                prev_ckpt_ind += 1
                self.config.defrost()
                self.config.MODEL.load_from_ckpt = True
                self.config.MODEL.ckpt_path = current_ckpt
                self.config.freeze()
                self.initialize(mode="eval")
                index = get_checkpoint_id(current_ckpt)
                res = self._eval_model(self.model, index=index)
                # write to tensorboard
                for k, v in res.items():
                    if isinstance(v, float):
                        writer.add_scalar(f"eval/{k}", v, index)
                # write to json
                current_ckpt = os.path.basename(current_ckpt)
                current_ckpt = (
                    current_ckpt.replace("ckpt.", "")
                    .replace(".pth", "")
                    .replace(".", "_")
                    + ".json"
                )
                if self.config.EVAL_PREFIX:
                    current_ckpt = self.config.EVAL_PREFIX + "_" + current_ckpt
                result_file = os.path.join(
                    self.config.CHECKPOINT_DIR, "evals", current_ckpt
                )
                with open(result_file, "w") as f:
                    logger.info(res)
                    f.write(json.dumps(res, indent=2))

    def _eval_model(self, model=None, loader=None, index=0):
        if model is None:
            model = self.model
        if loader is None:
            loader = self.loader
        model.eval()
        iter_num = 0
        batch_bar = tqdm.tqdm(
            loader,
            total=len(loader.dataset) // loader.batch_size,
            leave=False,
            dynamic_ncols=True,
        )
        all_predictions = []
        all_predictions_top5 = []
        all_targets = []
        all_filepaths = []
        with torch.no_grad():
            for batch in batch_bar:
                batch = {
                    k: (v.to(self.device_id) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                outputs = model(batch)
                try:
                    all_predictions_top5.extend(
                        torch.topk(outputs["logits"], 5)[1].cpu().tolist()
                    )
                    all_predictions_top5.extend(
                        torch.topk(outputs["logits"].T, 5)[1].cpu().tolist()
                    )
                except RuntimeError:
                    break
                all_predictions.extend(outputs["predictions"].cpu().tolist())
                all_targets.extend(outputs["targets"].cpu().tolist())
                all_filepaths.extend(batch["file_path"])
                all_filepaths.extend(batch["file_path"])

                batch_bar.set_description(f"CKPT {index}")
                batch_bar.set_postfix(
                    {
                        "acc": np.sum(
                            np.array(all_predictions) == np.array(all_targets)
                        )
                        / len(all_targets),
                    }
                )
                iter_num += 1
        all_predictions = np.array(all_predictions)
        all_predictions_top5 = np.array(all_predictions_top5)
        all_targets = np.array(all_targets)
        acc = np.sum(all_predictions == all_targets) / len(all_targets)
        i2d_idx = np.zeros_like(all_targets, dtype=bool)
        bs = loader.batch_size
        for i in range(0, len(i2d_idx), bs):
            left = i
            right = min(len(i2d_idx), i + bs)
            assert (right - left) % 2 == 0
            i2d_idx[left : (left + (right - left) // 2)] = True
        d2i_idx = np.logical_not(i2d_idx)
        assert np.sum(i2d_idx) == np.sum(d2i_idx)
        acc_i2d = np.sum((all_predictions == all_targets)[i2d_idx]) / np.sum(i2d_idx)
        acc_d2i = np.sum((all_predictions == all_targets)[d2i_idx]) / np.sum(d2i_idx)
        # Caution, near accuracy is meaningful only when the eval data is NOT shuffled
        acc_near1 = acc
        acc_near5 = np.sum(np.abs(all_predictions - all_targets) < 5) / len(all_targets)
        acc_near10 = np.sum(np.abs(all_predictions - all_targets) < 10) / len(
            all_targets
        )
        # Caution, top accuracy is meaningful only when the eval data is shuffled
        all_targets_expand = np.repeat(np.expand_dims(all_targets, axis=1), 5, axis=1)
        acc_top5 = np.logical_or.reduce(
            all_predictions_top5 == all_targets_expand, axis=1
        ).sum() / len(all_targets_expand)

        acc_split = {}
        acc_near5_split = {}
        keys = {
            "HM3D": "hm3d_val",
            "SceneNet": "scenenet_val500",
            "SUN3D": "sun3d_val",
            "TUM": "tum_val",
            "DIODE": "diode_val",
        }
        for k, v in keys.items():
            split_idx = [(v in path) for path in all_filepaths]
            split_idx = np.array(split_idx, dtype=bool)
            if np.sum(split_idx) > 0:
                acc_split[k] = np.sum(
                    all_predictions[split_idx] == all_targets[split_idx]
                ) / np.sum(split_idx)
                acc_near5_split[k] = np.sum(
                    np.abs(all_predictions[split_idx] - all_targets[split_idx]) < 5
                ) / np.sum(split_idx)
        return {
            "acc": acc,
            "acc_i2d": acc_i2d,
            "acc_d2i": acc_d2i,
            "acc_near1": acc_near1,
            "acc_near5": acc_near5,
            "acc_near10": acc_near10,
            "acc_top5": acc_top5,
            "acc_split": acc_split,
            "acc_near5_split": acc_near5_split,
        }

    def generate_eval_order(self):
        mode = "generate_eval_order"
        loader_cls = registry.get_dataloader(self.config.TRAINER.loaders[0])
        loader = loader_cls(
            self.config,
            mode=mode,
            num_workers=6,
        )
        batch_bar = tqdm.tqdm(
            loader,
            total=len(loader.dataset) // loader.batch_size,
            leave=False,
            dynamic_ncols=True,
        )
        all_rgbs = []
        all_depths = []
        for batch in batch_bar:
            all_rgbs.extend(batch["image"])
            all_depths.extend(batch["depth"])

        def long_str(li):
            result = ""
            for i in zip(*li):
                if len(set(i)) == 1:
                    result += i[0]
                else:
                    break
            return result

        root_folder = long_str(all_rgbs[0 : min(50, len(all_rgbs))])
        all_rgbs_relative = [v.replace(root_folder, "") for v in all_rgbs]
        all_depths_relative = [v.replace(root_folder, "") for v in all_depths]
        with open(
            f"eval_{self.config.DATA.RGBD.EVAL.shuffle}_rgb_order.json", "w"
        ) as f:
            f.write(json.dumps(all_rgbs_relative, indent=2))
        with open(
            f"eval_{self.config.DATA.RGBD.EVAL.shuffle}_depth_order.json", "w"
        ) as f:
            f.write(json.dumps(all_depths_relative, indent=2))
