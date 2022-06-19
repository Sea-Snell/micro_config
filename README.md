# micro_config.py
*an opinionated python dataclass config framework for deep learning*



I hope this approach to configurations can make your life easier, but if it doesn't, please submit a pull request or an issue and I'll see what I can do. This config system is certaintly under development, so open to any new ideas or suggestions.

# Installation

pip install:

``` shell
pip install micro-config
```

or install from source:

> Place `micro_config.py` at the root of your project.

# Repo Guide

The config framework is defined in `micro_config.py`.

The rest of the repo provides a demo for how one might actually want to use `micro_config.py` in a deep learning project. Specifically, I implement transformer language model training on wikitext in pytorch.

To run the demo:
1. navigate to the root directory
2. `pip install -r requirements.txt`
3. `export PYTHONPATH="$PWD"`
4. `cd scripts`
5. `python train_lm.py`

Optionally you can define commandline arguments to `train_lm.py` like:
``` shell
python train_lm.py epochs=1 bsize=16 model.transformer_config.hidden_dim=256
```

overview of demo project code:
* `scripts/train_lm.py` defines the training configuration and script execution.
* `base_config.py` defines config schema and defaults for all main config objects: `WikiDataConfig`, `TransformerConfig`, `LMModelConfig`, `AdamWConfig`
* `general_train_loop.py` defines the config schema and script for training models.
* `src/` defines all of the core demo project code.

# Quick Start / Walkthrough

*Most demo code in this section is adopted from the demo project provided in the repo.*

## Python dataclasses provide a more natural and flexible config definition interface than `.yaml` files.

* All config schema should be defined as an instance of `ConfigScript` or `ConfigScriptModel` and include a `@dataclass` decorator
* ConfigScripts firstly define a parameter schema and optionally default config values.

For example, a simple dataset object configuration:
``` python
from dataclasses import dataclass, adsict
from micro_config import ConfigScript

# data config
@dataclass
class WikiDataConfig(ConfigScript):
    f_path: str='data/wikitext-2-raw/wiki.train.raw'
    max_len: int=256
```

## `ConfigScript`s load associated objects or functions.
* To do this, all `ConfigScript`s implement `unroll(self, metaconfig: MetaConfig)`.
* The `metaconfig` parameter is another dataclass which specifies configs for the config framework. Feel free to subclass `MetaConfig`.

For example, loading the dataset from the config:
``` python
from dataclasses import dataclass, adsict
from micro_config import ConfigScript, MetaConfig
from src.data import WikitextDataset
import torch
import os

# data config
@dataclass
class WikiDataConfig(ConfigScript):
    f_path: str='data/wikitext-2-raw/wiki.train.raw'
    max_len: int=256

    def unroll(self, metaconfig: MetaConfig):
        # metaconfig.convert_path converts paths reletive to metaconfig.project_root into absolute paths
        return WikitextDataset(metaconfig.convert_path(self.f_path), self.max_len)

if __name__ == "__main__":
    metaconfig = MetaConfig(project_root=os.path.dirname(__file__), 
                            verbose=True)
    
    data_config = WikiDataConfig(max_len=512)
    data = data_config.unroll(metaconfig)
```

## Configurations can be defined hierarchically.
* You can define `ConfigScripts` as paremeters of other `ConfigScripts`
* You can define lists or dictionaries of `ConfigScript`s as parameters of a `ConfigScript` by wrapping your list or dict in `ConfigScriptList` or `ConfigScriptDict` respectively.

For example, the LM model config below defines `ConfigScript`s for both a dataset and a `transformer_config` as parameters:
``` python
from micro_config import MetaConfig
from base_configs import ConfigScriptModel
from dataclasses import field
from src.lm import LMModel
import os

# model config
@dataclass
class LMModelConfig(ConfigScriptModel):
    dataset: WikiDataConfig=field(default_factory=lambda: WikiDataConfig())
    transformer_config: TransformerConfig=field(default_factory=lambda: TransformerConfig(max_len=256))

    def unroll(self, metaconfig: MetaConfig):
        dataset = self.dataset.unroll(metaconfig)
        transformer_config = self.transformer_config.unroll(metaconfig)
        return LMModel(dataset, transformer_config, self.device)

if __name__ == "__main__":
    metaconfig = MetaConfig(project_root=os.path.dirname(__file__), 
                            verbose=True)

    model_config = LMModelConfig(
        checkpoint_path=None, 
        strict_load=True, 
        device='cpu', 
        dataset=WikiDataConfig(f_path='data/wikitext-2-raw/wiki.train.raw', max_len=256), 
        transformer_config=TransformerConfig(
            max_length=256, 
            heads=12, 
            hidden_dim=768, 
            attn_dim=64, 
            intermediate_dim=3072, 
            num_blocks=12, 
            dropout=0.1
        )
    )
    model = model_config.unroll(metaconfig)
```

`ConfigScriptModel` (not provided with `micro_config` out of the box), as used above, is a subclass of `ConfigScript` which defines some default functionality for loading a pytorch module returned by unroll and placing it on a specified device. You can look inside `base_configs.py` to see how to implement special functionality like this.

## Configs and scripts are unified: a config is to a script as a script is to a config.
* `unroll(self, metaconfig: MetaConfig)` can not only be used to load objects, but also to define script logic.

For example, let's define a simple configurable training loop:
``` python
from src.utils import combine_logs
from micro_config import ConfigScript, MetaConfig
from base_configs import ConfigScriptModel

@dataclass
class TrainLoop(ConfigScript):
    train_dataset: ConfigScript
    eval_dataset: ConfigScript
    model: ConfigScriptModel
    optim: ConfigScript
    epochs: int=10
    bsize: int=32
    
    def unroll(self, metaconfig: MetaConfig):
        print('using config:', asdict(self))
        device = metaconfig.device
        train_dataset = self.train_dataset.unroll(metaconfig)
        eval_dataset = self.eval_dataset.unroll(metaconfig)
        model = self.model.unroll(metaconfig)
        model.train()
        train_dataloader = DataLoader(train_dataset, batch_size=self.bsize)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.bsize)
        optim = self.optim.unroll(metaconfig)(model)
        for epoch in range(epochs):
            for x in tqdm(train_dataloader):
                loss, logs = model.get_loss(x.to(device))
                optim.zero_grad()
                loss.backward()
                optim.step()
                model.eval()
                val_x = next(iter(eval_dataloader))
                _, val_logs = model.get_loss(val_x.to(device))
                out_log = print({'train': combine_logs([logs]), 'val': combine_logs([val_logs]), 'step': (step+1)})
                model.train()
        return model
```

## Objects returned by `unroll(self, metaconfig: MetaConfig)` respect the reference structure of the config hierarchy.

* If the same config object is referenced multiple times in a config hierarchy, the object's `unroll(self, metaconfig: MetaConfig)` method will only be called once and its output cached, subsequent calls will return the cached output. If you don't want this caching behavior, you can subclass `ConfigScriptNoCache` instead.

For example, `train_dataset` is referenced twice in `train_config_script`:
``` python
import torch
import os

train_dataset = WikiDataConfig(f_path='data/wikitext-2-raw/wiki.train.raw', max_len=256)
eval_dataset = WikiDataConfig(f_path='data/wikitext-2-raw/wiki.valid.raw', max_len=256)

model = LMModelConfig(
            checkpoint_path=None, 
            strict_load=True, 
            device='cpu', 
            dataset=train_dataset, 
            transformer_config=TransformerConfig(
                max_length=256, 
                heads=12, 
                hidden_dim=768, 
                attn_dim=64, 
                intermediate_dim=3072, 
                num_blocks=12, 
                dropout=0.1
            )
        )

train_config_script = TrainLoop(
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset, 
    model=model, 
    optim=AdamWConfig(lr=1e-4, weight_decay=0.01), 
    epochs=10, 
    bsize=16, 
)

if __name__ == "__main__":
    metaconfig = MetaConfig(project_root=os.path.dirname(__file__), 
                            verbose=True)
    # run the script
    train_config_script.unroll(metaconfig)
```

The dataset object configured by `train_dataset` will only be loaded once in the above hiararchy, even though both `LMModelConfig` and `TrainLoop` take it in as input.

## A method for parsing commandline args is provided.

* `parse_args(config)` parses the command line arguments into a dictionary
* `deep_replace(config, **kwargs)` implements a nested version of the standard `dataclasses.replace` function

``` python
from micro_config import parse_args, deep_replace, MetaConfig
import os

if __name__ == "__main__":
    metaconfig = MetaConfig(project_root=os.path.dirname(__file__), 
                            verbose=True)
    train_config_script = deep_replace(train_config_script, **parse_args())
    # run the script
    train_config_script.unroll(metaconfig)
```

To edit any arguments in the hierarchy through the commandline, call the script like so:

``` shell
python train_lm.py epochs=1 bsize=16 model.transformer_config.hidden_dim=256
```
