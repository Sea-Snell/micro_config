from dataclasses import asdict
from typing import Union, Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base import BaseModel, LMGeneralDataset
from src.transformer import Transformer
from src.utils import causal_attn_mask

class LMModel(BaseModel):
    def __init__(self, dataset: LMGeneralDataset, 
                 transformer_config: Dict[str, Any], 
                 device: Union[torch.device, str]):
        super().__init__(dataset, device)
        self.n_tokens = len(self.dataset.tokenizer)
        self.transformer = Transformer(**transformer_config, 
                                       vocab_size=self.n_tokens, 
                                       output_size=self.n_tokens)
    
    def forward(self, x):
        attn_mask = causal_attn_mask(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1, 1).to(self.device)
        predictions, _, _ = self.transformer(x, attn_mask)
        return predictions
    
    def get_loss(self, x):
        predictions = self(x)
        loss = F.cross_entropy(predictions[:, :-1, :].reshape(-1, self.n_tokens), x[:, 1:].reshape(-1))
        return loss, {'loss': (loss.item(), x.shape[0])}
