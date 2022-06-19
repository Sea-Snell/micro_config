from dataclasses import dataclass, field, asdict
from typing import Optional, Union
from micro_config import ConfigScript
from src.data import WikitextDataset
from src.lm import LMModel
import torch
import os

project_root = os.path.dirname(__file__)

# configs define a parameter schema, defaults, and a method of loading the object from the config

# implements some default functionality for loading model checkpoints with the config_script.
@dataclass
class ConfigScriptModel(ConfigScript):
    checkpoint_path: Optional[str]
    strict_load: bool
    device: Union[torch.device, str]

    # You can always override self.meta_unroll to modify lower-level implementation details.
    def meta_unroll(unroll):
        def new_unroll(self, metaconfig):
            # load model in from cache
            if metaconfig.unrolled is None:
                metaconfig.unrolled = {}
            if id(self) in metaconfig.unrolled:
                if metaconfig.verbose:
                    print(f'fetching {self.__class__.__name__} from cache: {id(self)}')
                return metaconfig.unrolled[id(self)]
            if metaconfig.verbose:
                print(f'unrolling {self.__class__.__name__}: {id(self)}')
            
            # load model from self.checkpoint_path and place on self.device
            model = unroll(self, metaconfig)
            model = model.to(self.device)
            if self.checkpoint_path is not None:
                if metaconfig.verbose:
                    print('loading state dict from: %s' % metaconfig.convert_path(self.checkpoint_path))
                model.load_state_dict(torch.load(metaconfig.convert_path(self.checkpoint_path), map_location='cpu'), strict=self.strict_load)
                if metaconfig.verbose:
                    print('loaded.')
            
            # save model to cache and return
            metaconfig.unrolled[id(self)] = model
            if metaconfig.verbose:
                print(f'unrolled and {self.__class__.__name__} cached: {id(self)}')
            return model
        return new_unroll

# data config
@dataclass
class WikiDataConfig(ConfigScript):
    f_path: str='data/wikitext-2-raw/wiki.train.raw'
    max_len: int=256

    def unroll(self, metaconfig):
        return WikitextDataset(metaconfig.convert_path(self.f_path), self.max_len)

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
