import torch
from abc import ABC, abstractmethod
from typing import Optional
from transformers import PreTrainedTokenizerBase
from typing import List, Union
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset

class LMGeneralDataset(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_len: Optional[int]):
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def collate(self, items: List[List[int]], device):
        tokens = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x), tokens)), 
                                                 batch_first=True, padding_value=self.tokenizer.pad_token_id)
        tokens = tokens[:, :self.max_len]
        return tokens

class LMListDataset(LMGeneralDataset, Dataset):
    @abstractmethod
    def __getitem__(self, i: int) -> List[int]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

class LMIterableDataset(LMGeneralDataset, IterableDataset):
    @abstractmethod
    def __next__(self):
        pass

    def __iter__(self):
        return self

class BaseModel(ABC, nn.Module):
    def __init__(self, 
                 dataset: LMGeneralDataset, 
                 device: Union[torch.device, str]) -> None:
        super().__init__()
        self.dataset = dataset
        self.device = device

    @abstractmethod
    def get_loss(self, items: torch.Tensor, **kwargs):
        pass
