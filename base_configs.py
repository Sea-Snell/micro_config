from dataclasses import dataclass, field, asdict
from micro_config import ConfigScript, ConfigScriptModel, convert_path
from src.data import WikitextDataset
from src.lm import LMModel
import torch

# configs define a parameter schema, defaults, and a method of loading the object from the config

# data config
@dataclass
class WikiDataConfig(ConfigScript):
    f_path: str='data/wikitext-2-raw/wiki.train.raw'
    max_len: int=256

    def unroll(self, metaconfig):
        return WikitextDataset(convert_path(self.f_path), self.max_len)

# model config
@dataclass
class TransformerConfig(ConfigScript):
    max_length: int=1024
    heads: int=12
    hidden_dim: int=768
    attn_dim: int=64
    intermediate_dim: int=3072
    num_blocks: int=12
    block_repeats: int=1
    dropout: float=0.1
    pre_norm: bool=True

    def unroll(self, metaconfig):
        return asdict(self)

@dataclass
class LMModelConfig(ConfigScriptModel):
    dataset: WikiDataConfig=field(default_factory=lambda: WikiDataConfig())
    transformer_config: TransformerConfig=field(default_factory=lambda: TransformerConfig())

    def unroll(self, metaconfig):
        dataset = self.dataset.unroll(metaconfig)
        transformer_config = self.transformer_config.unroll(metaconfig)
        return LMModel(dataset, transformer_config, self.device)

# optimizer config
@dataclass
class AdamWConfig(ConfigScript):
    lr: float=1e-4
    betas: tuple=(0.9, 0.999)
    weight_decay: float=0.01

    def unroll(self, metaconfig):
        return lambda model: torch.optim.AdamW(model.parameters(), lr=self.lr, 
                                               weight_decay=self.weight_decay, 
                                               betas=self.betas)
