from __future__ import annotations
from copy import deepcopy
import os
import sys
from typing import Any, Dict, Optional, Union
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass

# defines parameters for the configuration process.
@dataclass
class MetaConfig:
    project_root: str=''
    verbose: bool=True
    device: Union[torch.device, str]='cpu'
    unrolled: Optional[Dict[int, Any]]=None

    # convert reletive path to absolute path
    def convert_path(self, path):
        if path is None:
            return None
        return os.path.join(self.project_root, path)

# standard config script super class.
# To keep config hierarchy references consistent, caches output of previously unrolled config_scripts by memory id.
@dataclass
class ConfigScript(ABC):
    def __init_subclass__(cls):
        cls.unroll = cls.meta_unroll(cls.unroll)
    
    def meta_unroll(unroll):
        def new_unroll(self, metaconfig):
            if metaconfig.unrolled is None:
                metaconfig.unrolled = {}
            if id(self) in metaconfig.unrolled:
                if metaconfig.verbose:
                    print(f'fetching {self.__class__.__name__} from cache: {id(self)}')
                return metaconfig.unrolled[id(self)]
            if metaconfig.verbose:
                print(f'unrolling {self.__class__.__name__}: {id(self)}')
            result = unroll(self, metaconfig)
            metaconfig.unrolled[id(self)] = result
            if metaconfig.verbose:
                print(f'unrolled {self.__class__.__name__} and cached: {id(self)}')
            return result
        return new_unroll
    
    @abstractmethod
    def unroll(self, metaconfig: MetaConfig) -> Any:
        pass

# if a ConfigScript contains a list/dict of ConfigScripts as a parameter, wrap the list/dict in these.
class ConfigScriptList(list):
    pass

class ConfigScriptDict(dict):
    pass

# implements some default functionality for loading model checkpoints with the config_script.
@dataclass
class ConfigScriptModel(ConfigScript):
    checkpoint_path: Optional[str]=None
    strict_load: bool=True
    device: Optional[Union[torch.device, str]]=None

    def meta_unroll(unroll):
        def new_unroll(self, metaconfig):
            if self.device is None:
                self.device = metaconfig.device
            if metaconfig.unrolled is None:
                metaconfig.unrolled = {}
            if id(self) in metaconfig.unrolled:
                if metaconfig.verbose:
                    print(f'fetching {self.__class__.__name__} from cache: {id(self)}')
                return metaconfig.unrolled[id(self)]
            if metaconfig.verbose:
                print(f'unrolling {self.__class__.__name__}: {id(self)}')
            model = unroll(self, metaconfig)
            model = model.to(self.device)
            if self.checkpoint_path is not None:
                if metaconfig.verbose:
                    print('loading state dict from: %s' % metaconfig.convert_path(self.checkpoint_path))
                model.load_state_dict(torch.load(metaconfig.convert_path(self.checkpoint_path), map_location='cpu'), strict=self.strict_load)
                if metaconfig.verbose:
                    print('loaded.')
            metaconfig.unrolled[id(self)] = model
            if metaconfig.verbose:
                print(f'unrolled and {self.__class__.__name__} cached: {id(self)}')
            return model
        return new_unroll

# deep version of dataclasses.replace
def deep_replace(cfg: ConfigScript, **overrides: Dict[str, Any]) -> ConfigScript:
    def inner_replace(base_config, **overrides_):
        for k, v in overrides_.items():
            base_config_item = getattr(base_config, k)
            if isinstance(base_config_item, ConfigScript):
                assert isinstance(v, dict)
                inner_replace(base_config_item, **v)
            elif isinstance(base_config_item, ConfigScriptList):
                assert isinstance(v, dict)
                for i, v2 in v.items():
                    assert isinstance(v2, dict)
                    inner_replace(base_config_item[int(i)], **v2)
            elif isinstance(base_config_item, ConfigScriptDict):
                assert isinstance(v, dict)
                for k2, v2 in v.items():
                    assert isinstance(v2, dict)
                    inner_replace(base_config_item[k2], **v2)
            else:
                setattr(base_config, k, v)
        return base_config
    return inner_replace(deepcopy(cfg), **overrides)

# custom method for parsing commandline arguments specified like: python script.py a.b.c=some_value a.b.d=some_other_value ...
def parse_args():
    structure = {}
    commandline_args = sys.argv[1:]
    for arg in commandline_args:
        if '=' in arg:
            k, v = arg.split('=')
            config_path = k.split('.')
            curr_element = structure
            for path_element in config_path[:-1]:
                if path_element not in curr_element:
                    curr_element[path_element] = {}
                curr_element = curr_element[path_element]
            if v in ['True', 'False']:
                v = (v == 'True')
            elif (v.replace('.','',1).isdigit() and '.' in v) or (v.replace('e', '', 1).isdigit() and 'e' in v) or (v.replace('e-', '', 1).isdigit() and 'e-' in v):
                v = float(v)
            elif v.isdigit():
                v = int(v)
            curr_element[config_path[-1]] = v
        else:
            raise NotImplementedError
    return structure
