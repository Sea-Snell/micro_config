from __future__ import annotations
from dataclasses import dataclass, field, replace
from micro_config import MetaConfig, ConfigScript, ConfigScriptModel, deep_replace, parse_args, convert_path
from src.data import WikitextDataset
import pickle as pkl

@dataclass
class DataConfig(ConfigScript):
    f_path: str='data/wikitext-2-raw/wiki.train.raw'
    seq_len: int=256

    def unroll(self, metaconfig):
        # return WikitextDataset(convert_path(self.f_path), self.seq_len)
        return 420

@dataclass
class ModelConfig(ConfigScript):
    data: DataConfig = field(default_factory=lambda: DataConfig())

    def unroll(self, metaconfig):
        dataset = self.data.unroll(metaconfig)
        return {'data': dataset, 'gpt2': None}

@dataclass
class TrainConfig(ConfigScript):
    epochs: int=10
    lr: float=1e-4
    bsize: int=12
    data: DataConfig = field(default_factory=lambda: DataConfig())
    model: ModelConfig = field(default_factory=lambda: ModelConfig())

    def unroll(self, metaconfig):
        dataset = self.data.unroll(metaconfig)
        model = self.model.unroll(metaconfig)
        return {'dataset': dataset, 'model': model}

if __name__ == "__main__":
    metaconfig = MetaConfig()
    data = DataConfig(
                        f_path='data/wikitext-2-raw/wiki.test.raw'
                     )
    train_config = TrainConfig(epochs=100, 
                               lr=1e-5, 
                               data=data, 
                               model=ModelConfig(data=data), 
                              )
    print(train_config)
    # print(parse_args())
    # print(deep_replace(train_config, **parse_args()))
    # print(deep_replace(train_config, lr=0.1, model=dict(
    #                                                     data=dict(
    #                                                               f_path='data/wikitext-2-raw/wiki.train.raw'
    #                                                              )
    #                                                    )))
    # print(train_config)
    # train_config.lr = 0.001
    # print(pkl.dumps(train_config))
    # train_config.unroll(metaconfig)
    # print(metaconfig)
