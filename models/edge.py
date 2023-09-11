from typing import Any, Mapping
import torch
import sys
import os
import math
import requests
import cv2
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel  # noqa: E402
from common.registry import registry  # noqa: E402


@registry.register_model
class EDGE(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # No need to load ckpt
        pass

    def edge_iou(self, edge1, edge2):
        mask1 = edge1 > 0
        mask2 = edge2 > 0
        return np.logical_and(mask1, mask2).sum() / (
            np.logical_or(mask1, mask2).sum() + 1e-8
        )

    def template_match(self, edge1, edge2):
        # !opencv implementation is too slow
        # res = cv2.matchTemplate(edge2,edge1,cv2.TM_CCORR_NORMED)
        # return float(res[0][0])
        norm1 = np.sqrt((edge1**2).sum())
        norm2 = np.sqrt((edge2**2).sum())
        corr = (edge1 * edge2).sum() / (norm1 * norm2 + 1e-8)
        return float(corr)

    def forward(self, batch):
        image = batch["image"]
        depth = batch["depth"]
        bs = image.shape[0]
        image = image.cpu().numpy()
        depth = depth.cpu().numpy()

        edges_image = []
        edges_depth = []
        logits = torch.zeros((bs, bs))
        edge_alg, metric = self.config.MODEL.loss_type.split("_")
        if edge_alg == "CANNY":
            for i in image:
                # Canny edge detection
                edges_image.append(cv2.Canny(i, 100, 200).astype(np.float32))
            for i in depth:
                # normalize
                i = (i - i.min()) / (i.max() - i.min() + 1e-8)
                i = (i * 255).astype(np.uint8)
                # Canny edge detection
                edges_depth.append(cv2.Canny(i, 100, 200).astype(np.float32))
        elif edge_alg == "SOBEL":
            for i in image:
                # normalize
                i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                # Sobel edge detection
                edges_image.append(
                    np.abs(cv2.Sobel(i, cv2.CV_32F, 1, 1, ksize=3))
                )  # ignoring the edge direction
            for i in depth:
                # normalize
                i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                i = (i - i.min()) / (i.max() - i.min() + 1e-8)
                i = (i * 255).astype(np.uint8)
                # Sobel edge detection
                edges_depth.append(np.abs(cv2.Sobel(i, cv2.CV_32F, 1, 1, ksize=3)))
        elif edge_alg == "LAPLACIAN":
            for i in image:
                # normalize
                i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                # Sobel edge detection
                edges_image.append(
                    cv2.convertScaleAbs(cv2.Laplacian(i, cv2.CV_8U))
                )  # ignoring the edge direction
            for i in depth:
                # normalize
                i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                i = (i - i.min()) / (i.max() - i.min() + 1e-8)
                i = (i * 255).astype(np.uint8)
                # Sobel edge detection
                edges_depth.append(cv2.convertScaleAbs(cv2.Laplacian(i, cv2.CV_8U)))

        if metric == "IOU":
            for i in range(bs):
                for j in range(bs):
                    logits[i, j] = self.edge_iou(edges_image[i], edges_depth[j])
        elif metric == "CORR":
            for i in range(bs):
                for j in range(bs):
                    logits[i, j] = self.template_match(edges_image[i], edges_depth[j])

        targets = torch.arange(bs).to(logits.device)
        loss_i2d = F.cross_entropy(logits, targets)
        loss_d2i = F.cross_entropy(logits.T, targets)
        loss = (loss_i2d + loss_d2i) / 2

        # for evaluation
        i2d_prediction = logits.argmax(dim=1)
        d2i_prediction = logits.argmax(dim=0)
        predictions = torch.cat([i2d_prediction, d2i_prediction], dim=0)
        targets = torch.cat([targets, targets], dim=0)

        return {
            "logits": logits,
            "loss": loss,
            "predictions": predictions,
            "targets": targets,
        }
