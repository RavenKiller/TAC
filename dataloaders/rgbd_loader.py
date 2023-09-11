import os
import math
import re
import time
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import T5TokenizerFast
from transformers import CLIPImageProcessor
from common.registry import registry
from dataloaders.base_loader import BaseLoader
from dataloaders.base_loader import BlockShuffleDistSampler
from common.logger import logger

# from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from natsort import natsorted

MIN_DEPTH = 0.0
MAX_DEPTH = 65.535


def list_sort(data_path, onlydigit=True):
    if onlydigit:
        return sorted(os.listdir(data_path), key=lambda f: int(re.sub(r"\D", "", f)))
    else:
        return sorted(os.listdir(data_path))


@registry.register_dataloader
class RGBDLoader(BaseLoader):
    def __init__(self, config, mode="train", *args, **kwargs):
        ds = RGBDDataset(config, mode=mode)
        super().__init__(ds, *args, **kwargs)


@registry.register_dataloader
class DistRGBDLoader(BaseLoader):
    def __init__(self, config, mode="train", *args, **kwargs):
        if mode == "train":  # block shuffle indicies
            ds = RGBDDataset(config, mode=mode)
            if len(config.DEVICE) > 1:  # distributed
                kwargs["shuffle"] = False
                ws = dist.get_world_size()
                assert (
                    config.TRAINER.batch_size % ws == 0
                ), "The batch size must be divisible by the world size!"
                kwargs["batch_size"] = config.TRAINER.batch_size // ws
                block_size = min(
                    config.DATA.RGBD.block_size, config.TRAINER.batch_size // (2 * ws)
                )
                kwargs["sampler"] = BlockShuffleDistSampler(
                    dataset=ds, block_size=block_size
                )
            else:  # single
                kwargs["shuffle"] = False
                ws = 1
                kwargs["batch_size"] = config.TRAINER.batch_size
                block_size = min(
                    config.DATA.RGBD.block_size, config.TRAINER.batch_size // (2 * ws)
                )
                kwargs["sampler"] = BlockShuffleDistSampler(
                    dataset=ds,
                    block_size=block_size,
                    num_replicas=1,
                    rank=int(os.environ["LOCAL_RANK"]),
                )
        elif mode == "eval":
            ds = RGBDDataset(config, mode=mode)
            kwargs["batch_size"] = config.TRAINER.batch_size
            if config.DATA.RGBD.EVAL.shuffle == "shuffle":
                kwargs["shuffle"] = True
            elif config.DATA.RGBD.EVAL.shuffle == "non-shuffle":
                kwargs["shuffle"] = False
            elif config.DATA.RGBD.EVAL.shuffle == "block-shuffle":
                kwargs["shuffle"] = False
                block_size = config.DATA.RGBD.EVAL.block_size
                kwargs["sampler"] = BlockShuffleDistSampler(
                    dataset=ds,
                    block_size=block_size,
                    num_replicas=1,
                    rank=int(os.environ["LOCAL_RANK"]),
                )
            torch.manual_seed(config.DATA.RGBD.EVAL.seed)
        elif mode == "generate_eval_order":
            ds = OrderRGBDDataset(config, mode=mode)
            kwargs["batch_size"] = config.TRAINER.batch_size
            if config.DATA.RGBD.EVAL.shuffle == "shuffle":
                kwargs["shuffle"] = True
            elif config.DATA.RGBD.EVAL.shuffle == "non-shuffle":
                kwargs["shuffle"] = False
            elif config.DATA.RGBD.EVAL.shuffle == "block-shuffle":
                kwargs["shuffle"] = False
                block_size = config.DATA.RGBD.EVAL.block_size
                kwargs["sampler"] = BlockShuffleDistSampler(
                    dataset=ds,
                    block_size=block_size,
                    num_replicas=1,
                    rank=int(os.environ["LOCAL_RANK"]),
                )
            torch.manual_seed(config.DATA.RGBD.EVAL.seed)

        super().__init__(ds, *args, **kwargs)


class RGBDDataset(Dataset):
    def __init__(self, config, mode="train", processor=None):
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
        if processor is None:
            self.processor = CLIPImageProcessor.from_pretrained(
                config.MODEL.IMAGE.model_name
            )
        else:
            self.processor = processor

    def read_image(self, image_path):
        """Return a image tensor from image_path"""
        image = Image.open(image_path).convert("RGB")
        if self.config.MODEL.name != "EDGE":
            if self.is_resized:
                image = self.processor(
                    image, do_resize=False, do_center_crop=False, return_tensors="pt"
                ).pixel_values.squeeze()
                # test2 = self.processor(image, return_tensors="pt").pixel_values.squeeze()
            else:
                image = self.processor(
                    image, return_tensors="pt"
                ).pixel_values.squeeze()
        else:
            image = torch.tensor(np.array(image))
        return image

    def read_depth(self, depth_path, depth_scale=5000.0):
        """Return a depth tensor from depth_path"""
        depth = Image.open(depth_path)
        depth = np.array(depth).astype("float32") / depth_scale  # to meters
        MIN_DEPTH = depth.min()
        depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)
        depth = (depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
        depth = np.expand_dims(depth, axis=2).repeat(3, axis=2)
        if self.config.MODEL.name != "EDGE":
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
                depth = self.processor(
                    depth, return_tensors="pt"
                ).pixel_values.squeeze()
        else:
            depth = torch.tensor(depth)
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
        episode = hash(os.path.dirname(image_path))
        index = int(
            os.path.basename(image_path).replace(".jpg", "").replace(".png", "")
        )

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "episode": episode,
            "index": index,
            "time_factor": self.time_factors[i],
            "file_path": self.image_samples[i],
        }


class OrderRGBDDataset(Dataset):
    def __init__(self, config, mode="generate_eval_order", processor=None):
        assert mode == "generate_eval_order"
        self.data_path = config.DATA.RGBD.EVAL.data_path
        self.scale_value = config.DATA.RGBD.EVAL.scale_value
        self.time_factor = config.DATA.RGBD.EVAL.time_factor
        self.is_resized = config.DATA.RGBD.EVAL.is_resized

        self.image_samples = []
        self.depth_samples = []
        self.depth_scales = []
        self.time_factors = []
        # data_path / episodes / image / sequence
        assert isinstance(self.data_path, list), "The data_path must be list!"
        for i, data_path in enumerate(self.data_path):
            for episode in list_sort(data_path):
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
        if processor is None:
            self.processor = CLIPImageProcessor.from_pretrained(
                config.MODEL.IMAGE.model_name
            )
        else:
            self.processor = processor

    def __len__(self):
        return len(self.image_samples)

    def __getitem__(self, i):
        image_path = self.image_samples[i]
        depth_path = self.depth_samples[i]

        return {
            "image": image_path,
            "depth": depth_path,
        }


if __name__ == "__main__":
    import requests
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    dog_path = "/hy-tmp/dog.png"
    cat_path = "/hy-tmp/cat.png"
    noise_path = "/root/UniSpeaker/data/speaker_data/vlnce_train/10821/depth/000.png"
    dog_img = Image.open(dog_path).convert("RGB")
    cat_img = Image.open(cat_path).convert("RGB")
    noise_img = Image.open(noise_path)
    dog_img = transforms.ToTensor()(dog_img)
    cat_img = transforms.ToTensor()(cat_img)
    noise_img = transforms.ToTensor()(noise_img).repeat(3, 1, 1)
    print(dog_img.shape, cat_img.shape)
    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=[dog_img, cat_img, noise_img],
        return_tensors="pt",
        padding=True,
    )

    outputs = model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities
    print("Open-totensor-process", probs)

    dog_path = "/hy-tmp/dog.png"
    cat_path = "/hy-tmp/cat.png"
    noise_path = "/root/UniSpeaker/data/speaker_data/vlnce_train/10821/depth/000.png"
    dog_img = Image.open(dog_path)
    cat_img = Image.open(cat_path)
    noise_img = Image.open(noise_path)
    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=[dog_img, cat_img, noise_img],
        return_tensors="pt",
        padding=True,
    )

    outputs = model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities
    print("Open-process", probs)
