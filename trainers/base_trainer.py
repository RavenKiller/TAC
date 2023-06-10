class BaseTrainer:
    @classmethod
    def from_config(cls, config):
        return cls(config=config)

    def initialize(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError
