import torch


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, config):
        return cls(config=config)
