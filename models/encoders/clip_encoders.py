import clip
import numpy as np
import torch
import torch.nn as nn
from gym import Space, spaces
from torch import Tensor

from transformers import DistilBertModel, DistilBertTokenizer, AutoModel, AutoTokenizer


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        trainable: bool = False,
        rgb_level: int = -1,
    ) -> None:
        super().__init__()
        self.model, self.preprocessor = clip.load(model_name)
        for param in self.model.parameters():
            param.requires_grad_(trainable)
        self.normalize_visual_inputs = True
        self.normalize_mu = torch.FloatTensor([0.48145466, 0.4578275, 0.40821073])
        self.normalize_sigma = torch.FloatTensor([0.26862954, 0.26130258, 0.27577711])
        self.rgb_embedding_seq = None
        # self.ln_rgb = nn.LayerNorm(768)
        # self.ln_text = nn.LayerNorm(512)
        self.use_mean = True
        if rgb_level == -1:
            self.model.visual.transformer.register_forward_hook(self._vit_hook)
            self.model.transformer.register_forward_hook(self._t_hook)
        else:
            self.model.visual.transformer.resblocks[rgb_level].register_forward_hook(
                self._vit_hook
            )
            self.model.transformer.resblocks[rgb_level].register_forward_hook(
                self._t_hook
            )
        self.sub_embedding_seq = None

    def _normalize(self, imgs: Tensor) -> Tensor:
        if self.normalize_visual_inputs:
            device = imgs.device
            if self.normalize_sigma.device != imgs.device:
                self.normalize_sigma = self.normalize_sigma.to(device)
                self.normalize_mu = self.normalize_mu.to(device)
            imgs = (imgs / 255.0 - self.normalize_mu) / self.normalize_sigma
            imgs = imgs.permute(0, 3, 1, 2)
            return imgs
        else:
            return imgs

    def _vit_hook(self, m, i, o):
        self.rgb_embedding_seq = o.float()

    def encode_image(self, pixel_values) -> Tensor:
        rgb_observations = pixel_values
        _ = self.model.encode_image(rgb_observations).float()
        # LND -> NLD
        rgb_embedding_seq = self.rgb_embedding_seq.float().permute(1, 0, 2)
        return {"last_hidden_states": rgb_embedding_seq}

    def forward(self, pixel_values):
        return self.encode_image(pixel_values=pixel_values)


def convert_weights_float(model: nn.Module):
    """Convert applicable model parameters back to fp32"""

    def _convert_weights_to_fp32(md):
        if isinstance(md, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            md.weight.data = md.weight.data.float()
            if md.bias is not None:
                md.bias.data = md.bias.data.float()

        if isinstance(md, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(md, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(md, name):
                attr = getattr(md, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)


class CLIPResEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        rgb_level: int = -1,
    ) -> None:
        super().__init__()
        self.model, self.preprocessor = clip.load(model_name)
        self.rgb_embedding_seq = None
        self.model.visual.attnpool.register_forward_hook(self._vit_hook)
        self.sub_embedding_seq = None
        convert_weights_float(self.model)
        self.model.train()

    def _vit_hook(self, m, i, o):
        self.rgb_embedding_seq = o.float()

    def encode_image(self, pixel_values) -> Tensor:
        rgb_observations = pixel_values
        _ = self.model.encode_image(rgb_observations).float()
        # LND -> NLD
        rgb_embedding_seq = self.rgb_embedding_seq.float().unsqueeze(1)
        return {"last_hidden_state": rgb_embedding_seq}

    def forward(self, pixel_values):
        return self.encode_image(pixel_values=pixel_values)
