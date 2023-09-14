import torch
import sys
import os
import math
import torch.nn.functional as F
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTImageProcessor,
    ViTMAEConfig,
    ViTMAEModel,
    ViTMAEForPreTraining,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.default import get_config  # noqa: E402
from models.base_model import BaseModel  # noqa: E402
from common.registry import registry  # noqa: E402


@registry.register_model
class MAE(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # RGB image model
        self.image_transformer = ViTMAEModel.from_pretrained(
            config.MODEL.IMAGE.model_name
        )
        self.image_processor = ViTImageProcessor.from_pretrained(
            config.MODEL.IMAGE.model_name
        )
        self.image_feature_dim = config.MODEL.IMAGE.feature_dim
        self.image_projection_dim = config.MODEL.IMAGE.projection_dim
        self.image_fc = torch.nn.Linear(
            self.image_feature_dim, self.image_projection_dim, bias=False
        )

        # Depth model
        self.depth_transformer = ViTMAEModel.from_pretrained(
            config.MODEL.DEPTH.model_name
        )
        self.depth_processor = ViTImageProcessor.from_pretrained(
            config.MODEL.DEPTH.model_name
        )
        self.depth_feature_dim = config.MODEL.DEPTH.feature_dim
        self.depth_projection_dim = config.MODEL.DEPTH.projection_dim
        self.depth_fc = torch.nn.Linear(
            self.depth_feature_dim, self.depth_projection_dim, bias=False
        )

        self.block_size = config.DATA.RGBD.block_size
        # bs = self.config.TRAINER.batch_size
        # self.wp = 1
        # self.wn = config.MODEL.negative_weight
        # self.gamma = config.MODEL.focal_gamma
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

    def embed_image(self, image_batch, fc=True):
        """Embed a batch of image."""
        # batch_size = image_batch.shape[0]
        # device = image_batch.device
        outputs = self.image_transformer(pixel_values=image_batch)
        outputs = outputs["last_hidden_state"][:, 0, :]
        if fc:
            outputs = self.image_fc(outputs)
        return outputs

    def embed_depth(self, depth_batch, fc=True):
        """Embed a batch of depth."""
        # batch_size = depth_batch.shape[0]
        # device = depth_batch.device
        outputs = self.depth_transformer(pixel_values=depth_batch)
        outputs = outputs["last_hidden_state"][:, 0, :]
        if fc:
            outputs = self.depth_fc(outputs)
        return outputs

    def forward(self, batch):
        image_embeddings = self.embed_image(batch["image"], fc=False)
        depth_embeddings = self.embed_depth(batch["depth"], fc=False)
        bs = image_embeddings.shape[0]
        image_embeddings = F.normalize(image_embeddings)
        depth_embeddings = F.normalize(depth_embeddings)
        logits = torch.matmul(image_embeddings, depth_embeddings.T) * math.exp(0.5)
        targets = torch.arange(bs).to(image_embeddings.device)
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


if __name__ == "__main__":
    config = get_config("/root/TAC/config/v2/v2_mae.yaml")
    model = MAE(config)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict()},
        os.path.join(config.CHECKPOINT_DIR, "ckpt.MAE.00.pth"),
    )
