import os
import math
import re
import time
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTImageProcessor,
    ViTMAEConfig,
    ViTMAEForPreTraining,
)


logger = logging.getLogger(__name__)
MIN_DEPTH = 0.0
MAX_DEPTH = 10.0


def list_sort(data_path, onlydigit=True):
    if onlydigit:
        return sorted(os.listdir(data_path), key=lambda f: int(re.sub(r"\D", "", f)))
    else:
        return sorted(os.listdir(data_path))


class RGBDDataset(Dataset):
    def __init__(self, config, mode="train"):
        if mode == "train":
            self.data_path = config.DATA.RGBD.data_path
            self.scale_value = config.DATA.RGBD.scale_value
            self.time_factor = config.DATA.RGBD.time_factor
            self.is_resized = config.DATA.RGBD.is_resized
        elif mode == "eval":
            self.data_path = config.DATA.RGBD.EVAL.data_path
            self.scale_value = config.DATA.RGBD.EVAL.scale_value
            self.time_factor = config.DATA.RGBD.EVAL.time_factor
            self.is_resized = config.DATA.RGBD.EVAL.is_resized
        self.config = config
        self.image_samples = []
        self.depth_samples = []
        self.depth_scales = []
        self.time_factors = []
        # data_path / episodes / image / sequence
        assert isinstance(self.data_path, list), "The data_path must be list!"
        for i, data_path in enumerate(self.data_path):
            for episode in list_sort(data_path, onlydigit=False):
                image_files = list_sort(os.path.join(data_path, episode, "rgb"))
                depth_files = list_sort(os.path.join(data_path, episode, "depth"))
                assert len(image_files) == len(
                    depth_files
                ), "Please ensure all rgbs and depths in %s are aligned!" % (
                    os.path.join(data_path, episode)
                )

                self.image_samples.extend(
                    [os.path.join(data_path, episode, "rgb", v) for v in image_files]
                )
                self.depth_samples.extend(
                    [os.path.join(data_path, episode, "depth", v) for v in depth_files]
                )
                self.depth_scales.extend([self.scale_value[i]] * len(depth_files))
                self.time_factors.extend([self.time_factor[i]] * len(depth_files))
        logger.debug(f"Sample number: {len(self.image_samples)}")
        self.processor = ViTImageProcessor.from_pretrained(
            config.MODEL.IMAGE.model_name
        )

    def read_image(self, image_path):
        """Return a image tensor from image_path"""
        image = Image.open(image_path).convert("RGB")
        if self.is_resized:
            image = self.processor(
                image, do_resize=False, do_center_crop=False, return_tensors="pt"
            ).pixel_values.squeeze()
            # test2 = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        else:
            image = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        return image

    def read_depth(self, depth_path, depth_scale=5000.0):
        """Return a depth tensor from depth_path"""
        depth = Image.open(depth_path)
        depth = np.array(depth).astype("float32") / depth_scale  # to meters
        MIN_DEPTH = depth.min()
        depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)
        depth = (depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
        depth = np.expand_dims(depth, axis=2).repeat(3, axis=2)
        if self.is_resized:
            depth = self.processor(
                depth,
                do_resize=False,
                do_center_crop=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.squeeze()
            # test2 = self.processor(depth, return_tensors="pt").pixel_values.squeeze() # numeric error because of [0,1]->[0,255]->[0,1]
        else:
            depth = self.processor(depth, return_tensors="pt").pixel_values.squeeze()
        return depth

    def __len__(self):
        return len(self.image_samples)

    def __getitem__(self, i):
        image_path = self.image_samples[i]
        depth_path = self.depth_samples[i]
        # print(image_path, depth_path)
        depth_scale = self.depth_scales[i]
        image_tensor = self.read_image(image_path)
        depth_tensor = self.read_depth(depth_path, depth_scale)
        # episode = hash(os.path.dirname(image_path))
        # index = int(
        #     os.path.basename(image_path).replace(".jpg", "").replace(".png", "")
        # )

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            # "episode": episode,
            # "index": index,
            # "time_factor": self.time_factors[i],
            # "file_path": self.image_samples[i],
        }
