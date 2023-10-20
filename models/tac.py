import torch
import sys
import os
import math
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from PIL import Image
import requests
from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig
from transformers import ViTConfig, ViTImageProcessor, ViTModel
from config.default import get_config
from models.base_model import BaseModel
from common.registry import registry
from common.logger import logger
from common.utils import cross_entropy_focal
from models.encoders.clip_encoders import CLIPResEncoder
import torch.nn.functional as F
from torch.cuda.amp import autocast

# import timm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WrapModule(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        res = self.model(pixel_values)
        if len(res.shape) == 2:
            res = res.unsqueeze(1)
        return {"last_hidden_state": res}


@registry.register_model
class TAC(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # RGB image model
        self.image_transformer = CLIPVisionModel.from_pretrained(
            config.MODEL.IMAGE.model_name
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(
            config.MODEL.IMAGE.model_name
        )
        self.image_feature_dim = config.MODEL.IMAGE.feature_dim
        self.image_projection_dim = config.MODEL.IMAGE.projection_dim
        self.image_fc = torch.nn.Linear(
            self.image_feature_dim, self.image_projection_dim, bias=False
        )

        # Depth model
        if config.MODEL.bottleneck == "vit":
            self.depth_transformer = CLIPVisionModel.from_pretrained(
                config.MODEL.DEPTH.model_name
            )
            self.depth_processor = CLIPImageProcessor.from_pretrained(
                config.MODEL.DEPTH.model_name
            )
        elif config.MODEL.bottleneck == "vits":
            # Deprecated
            # self.depth_transformer = WrapModule(
            #     timm.create_model(
            #         "vit_small_patch32_224.augreg_in21k_ft_in1k",
            #         pretrained=True,
            #         num_classes=0,  # remove classifier nn.Linear
            #     )
            # )
            cfg = CLIPVisionConfig.from_pretrained(config.MODEL.DEPTH.model_name)
            cfg.num_hidden_layers = 8
            cfg.hidden_size = 512
            cfg.num_attention_heads = 8
            cfg.intermediate_size = 1536
            self.depth_transformer = CLIPVisionModel(cfg)
            self.depth_processor = CLIPImageProcessor.from_pretrained(
                config.MODEL.DEPTH.model_name
            )
        elif config.MODEL.bottleneck == "vitl":
            cfg = CLIPVisionConfig.from_pretrained(config.MODEL.DEPTH.model_name)
            cfg.num_hidden_layers = 16
            cfg.hidden_size = 1024
            cfg.num_attention_heads = 16
            cfg.intermediate_size = 4096
            self.depth_transformer = CLIPVisionModel(cfg)
            self.depth_processor = CLIPImageProcessor.from_pretrained(
                config.MODEL.DEPTH.model_name
            )
        elif config.MODEL.bottleneck == "rn50":
            self.depth_transformer = CLIPResEncoder("RN50")
        else:
            self.depth_transformer = CLIPResEncoder("RN101")
        self.depth_feature_dim = config.MODEL.DEPTH.feature_dim
        self.depth_projection_dim = config.MODEL.DEPTH.projection_dim
        self.depth_fc = torch.nn.Linear(
            self.depth_feature_dim, self.depth_projection_dim, bias=False
        )

        # Control similarity matrix scale
        self.temperature = torch.nn.Parameter(torch.tensor([0.07]), requires_grad=True)

        # Classification projection
        if self.config.MODEL.loss_type == "SUP":  # clip loss with soft label
            self.classification = torch.nn.Sequential(
                torch.nn.Linear(
                    self.image_projection_dim + self.depth_projection_dim,
                    self.image_projection_dim,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    self.image_projection_dim,
                    1,
                ),
            )
        else:
            self.classification = torch.nn.Sequential(
                torch.nn.Linear(
                    self.image_projection_dim + self.depth_projection_dim,
                    2,
                ),
            )
        self.block_size = config.DATA.RGBD.block_size
        self.margin = config.MODEL.loss_margin
        self.time_scale = self.time_scale = torch.nn.Parameter(
            torch.tensor([config.MODEL.init_time_scale]),
            requires_grad=config.MODEL.learnable_time_scale,
        )

        # Set trainable
        for param in self.image_transformer.parameters():
            param.requires_grad_(config.MODEL.IMAGE.trainable)
        for param in self.depth_transformer.parameters():
            param.requires_grad_(config.MODEL.DEPTH.trainable)

        # Init params
        self.init_param()

    def embed_image(self, image_batch, fc=True):
        """Embed a batch of image."""
        outputs = self.image_transformer(pixel_values=image_batch)
        outputs = outputs["last_hidden_state"][:, 0, :]
        if fc:
            outputs = self.image_fc(outputs)
        return outputs

    def embed_depth(self, depth_batch, fc=True):
        """Embed a batch of depth."""
        outputs = self.depth_transformer(pixel_values=depth_batch)
        outputs = outputs["last_hidden_state"][:, 0, :]
        if fc:
            outputs = self.depth_fc(outputs)
        return outputs

    def init_param(self):
        pass

    def clamp_param(self):
        self.temperature.data.clamp_(-2, 5)
        self.time_scale.data.clamp_(1 / 20, 1)

    def forward(self, batch, **kwargs):
        image_embeddings = self.embed_image(batch["image"])
        depth_embeddings = self.embed_depth(batch["depth"])
        bs = image_embeddings.shape[0]
        if self.config.MODEL.loss_type == "CLIP":  # clip loss
            # calculate loss
            image_embeddings = F.normalize(image_embeddings)
            depth_embeddings = F.normalize(depth_embeddings)
            logits = torch.matmul(image_embeddings, depth_embeddings.T) * torch.exp(
                self.temperature
            )
            targets = torch.arange(bs).to(image_embeddings.device)
            loss_i2d = F.cross_entropy(logits, targets)
            loss_d2i = F.cross_entropy(logits.T, targets)
            loss = (loss_i2d + loss_d2i) / 2

            # for evaluation
            i2d_prediction = logits.argmax(dim=1)
            d2i_prediction = logits.argmax(dim=0)
            predictions = torch.cat([i2d_prediction, d2i_prediction], dim=0)
            targets = torch.cat([targets, targets], dim=0)
        elif self.config.MODEL.loss_type == "TAC":  # tac loss
            # calculate loss
            image_embeddings = F.normalize(image_embeddings)
            depth_embeddings = F.normalize(depth_embeddings)
            logits = torch.matmul(image_embeddings, depth_embeddings.T) * torch.exp(
                self.temperature
            )
            # episode = batch["episode"].float()
            index = batch["index"].float()
            # time factor is 3sigma
            with torch.no_grad():
                targets = torch.arange(bs).to(image_embeddings.device)
                time_factor = batch["time_factor"].float() * self.time_scale
                # episode_mat = episode.unsqueeze(1).repeat(1, bs)
                index_mat = index.unsqueeze(1).repeat(1, bs)
                sigma_mat = time_factor.unsqueeze(1).repeat(1, bs)
                gt_mat = -(((index_mat - index_mat.T) / sigma_mat) ** 2) / 2
                gt_mat = torch.exp(gt_mat)
                if self.config.MODEL.use_gaussian_coef:
                    gt_mat = gt_mat / (math.sqrt(2 * math.pi) * sigma_mat)
                # gt_mat[episode_mat == episode_mat.T] = gt_mat_[episode_mat == episode_mat.T]
                targets_mat = targets.unsqueeze(1).repeat(1, bs)
                gt_mat[
                    (targets_mat / self.block_size).int()
                    != (targets_mat.T / self.block_size).int()
                ] = 0
            # TODO: different curves for gt. the current is gaussian + linear prob
            loss_i2d = F.cross_entropy(logits, F.normalize(gt_mat, p=1))
            loss_d2i = F.cross_entropy(logits.T, F.normalize(gt_mat.T, p=1))
            loss = (loss_i2d + loss_d2i) / 2

            # for evaluation
            with torch.no_grad():
                i2d_prediction = logits.argmax(dim=1)
                d2i_prediction = logits.argmax(dim=0)
                predictions = torch.cat([i2d_prediction, d2i_prediction], dim=0)
                targets = torch.cat([targets, targets], dim=0)
        elif (
            self.config.MODEL.loss_type == "MULI_LABEL"
        ):  # soft label to multi hard positive label
            # calculate loss
            image_embeddings = F.normalize(image_embeddings)
            depth_embeddings = F.normalize(depth_embeddings)
            logits = torch.matmul(image_embeddings, depth_embeddings.T) * torch.exp(
                self.temperature
            )
            # episode = batch["episode"].float()
            index = batch["index"].float()
            # time factor is 3sigma
            with torch.no_grad():
                targets = torch.arange(bs).to(image_embeddings.device)
                time_factor = batch["time_factor"].float() * self.time_scale
                # episode_mat = episode.unsqueeze(1).repeat(1, bs)
                index_mat = index.unsqueeze(1).repeat(1, bs)
                sigma_mat = time_factor.unsqueeze(1).repeat(1, bs)
                gt_mat = -(((index_mat - index_mat.T) / sigma_mat) ** 2) / 2
                gt_mat = torch.exp(gt_mat)
                if self.config.MODEL.use_gaussian_coef:
                    gt_mat = gt_mat / (math.sqrt(2 * math.pi) * sigma_mat)
                # gt_mat[episode_mat == episode_mat.T] = gt_mat_[episode_mat == episode_mat.T]
                targets_mat = targets.unsqueeze(1).repeat(1, bs)
                gt_mat[
                    (targets_mat / self.block_size).int()
                    != (targets_mat.T / self.block_size).int()
                ] = 0
            # half threshold
            gt_mat[gt_mat >= 0.5] = 1
            gt_mat[gt_mat < 0.5] = 0
            # TODO: different curves for gt. the current is gaussian + linear prob
            loss_i2d = F.multilabel_soft_margin_loss(logits, gt_mat)
            loss_d2i = F.multilabel_soft_margin_loss(logits.T, gt_mat.T)
            loss = (loss_i2d + loss_d2i) / 2

            # for evaluation
            with torch.no_grad():
                i2d_prediction = logits.argmax(dim=1)
                d2i_prediction = logits.argmax(dim=0)
                predictions = torch.cat([i2d_prediction, d2i_prediction], dim=0)
                targets = torch.cat([targets, targets], dim=0)
        elif self.config.MODEL.loss_type == "C_L2":  # contrastive loss with L2 distance
            # calculate loss
            image_embeddings = F.normalize(image_embeddings)
            depth_embeddings = F.normalize(depth_embeddings)
            dp = torch.norm(image_embeddings - depth_embeddings, dim=1)
            dn = torch.norm(
                image_embeddings
                - torch.roll(depth_embeddings, self.block_size, dims=0),
                dim=1,
            )
            loss = (dp**2).mean() / 2 + (torch.relu(self.margin - dn) ** 2).mean() / 2

            # for evaluation
            # small distance -> large logits
            with torch.no_grad():
                logits = -torch.norm(
                    image_embeddings.unsqueeze(1).expand(-1, bs, -1)
                    - depth_embeddings.unsqueeze(0).expand(bs, -1, -1),
                    dim=2,
                )
                targets = torch.arange(bs).to(image_embeddings.device)
                i2d_prediction = logits.argmax(dim=1)
                d2i_prediction = logits.argmax(dim=0)
                predictions = torch.cat([i2d_prediction, d2i_prediction], dim=0)
                targets = torch.cat([targets, targets], dim=0)
        elif (
            self.config.MODEL.loss_type == "C_COS"
        ):  # contrastive loss with cosine similarity
            # calculate loss
            image_embeddings = F.normalize(image_embeddings)
            depth_embeddings = F.normalize(depth_embeddings)
            dp = (image_embeddings * depth_embeddings).sum(dim=1) - 1  # [-2, 0]
            dn = (
                image_embeddings * torch.roll(depth_embeddings, self.block_size, dims=0)
            ).sum(dim=1) - 1
            loss = (-dp).mean() / 2 + torch.relu(self.margin + dn).mean() / 2

            # for evaluation
            logits = torch.matmul(image_embeddings, depth_embeddings.T)
            targets = torch.arange(bs).to(image_embeddings.device)
            i2d_prediction = logits.argmax(dim=1)
            d2i_prediction = logits.argmax(dim=0)
            predictions = torch.cat([i2d_prediction, d2i_prediction], dim=0)
            targets = torch.cat([targets, targets], dim=0)
        elif self.config.MODEL.loss_type == "T_L2":  # triplet loss with L2 distance
            # calculate loss
            image_embeddings = F.normalize(image_embeddings)
            depth_embeddings = F.normalize(depth_embeddings)
            dp = torch.norm(image_embeddings - depth_embeddings, dim=1)
            dn = torch.norm(
                image_embeddings
                - torch.roll(depth_embeddings, self.block_size, dims=0),
                dim=1,
            )
            loss = (torch.relu(dp**2 - dn**2 + self.margin)).mean()

            # for evaluation
            # small distance -> large logits
            with torch.no_grad():
                logits = -torch.norm(
                    image_embeddings.unsqueeze(1).expand(-1, bs, -1)
                    - depth_embeddings.unsqueeze(0).expand(bs, -1, -1),
                    dim=2,
                )
                targets = torch.arange(bs).to(image_embeddings.device)
                i2d_prediction = logits.argmax(dim=1)
                d2i_prediction = logits.argmax(dim=0)
                predictions = torch.cat([i2d_prediction, d2i_prediction], dim=0)
                targets = torch.cat([targets, targets], dim=0)
        elif (
            self.config.MODEL.loss_type == "T_COS"
        ):  # triplet loss with cosine similarity
            # calculate loss
            image_embeddings = F.normalize(image_embeddings)
            depth_embeddings = F.normalize(depth_embeddings)
            dp = (image_embeddings * depth_embeddings).sum(dim=1) - 1
            dn = (
                image_embeddings * torch.roll(depth_embeddings, self.block_size, dims=0)
            ).sum(dim=1) - 1
            loss = torch.relu(-dp + dn + self.margin).mean()

            # for evaluation
            logits = torch.matmul(image_embeddings, depth_embeddings.T)
            targets = torch.arange(bs).to(image_embeddings.device)
            i2d_prediction = logits.argmax(dim=1)
            d2i_prediction = logits.argmax(dim=0)
            predictions = torch.cat([i2d_prediction, d2i_prediction], dim=0)
            targets = torch.cat([targets, targets], dim=0)
        elif self.config.MODEL.loss_type == "SUP":  # supervised loss
            # calculate loss
            fp = torch.cat((image_embeddings, depth_embeddings), dim=1)
            fn = torch.cat(
                (
                    image_embeddings,
                    torch.roll(depth_embeddings, self.block_size, dims=0),
                ),
                dim=1,
            )
            dp = self.classification(fp)
            dn = self.classification(fn)
            logits_ = torch.cat((dp, dn), dim=0)
            targets_ = torch.cat((torch.ones_like(dp), torch.zeros_like(dn)), dim=0)
            loss = F.binary_cross_entropy_with_logits(logits_, targets_)

            # for evaluation
            with torch.no_grad():
                cat_embeddings = torch.cat(
                    [
                        image_embeddings.unsqueeze(1).expand(-1, bs, -1),
                        depth_embeddings.unsqueeze(0).expand(bs, -1, -1),
                    ],
                    dim=2,
                )
                logits = self.classification(cat_embeddings).squeeze()
                gt_mat = torch.diag(torch.ones(bs))

                targets = torch.arange(bs).to(image_embeddings.device)
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


if __name__ == "__main__":
    config = get_config()
    speaker = TAC(config)
    s = "hello world!"
    a = torch.randn((2, 512))
    b = torch.randn((2, 512))
    res = speaker.get_similarity(a, b)
