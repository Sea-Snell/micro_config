import random
from transformers import GPT2Tokenizer
import torch
from src.base import LMIterableDataset

class WikitextDataset(LMIterableDataset):
    def __init__(self, f_path, max_len):
        super().__init__(GPT2Tokenizer.from_pretrained('gpt2'), 
                         max_len)
        with open(f_path, 'r') as f:
            d = f.read()
        self.tokens = self.tokenizer.encode(d)
    
    def __next__(self):
        s = random.randint(0, len(self.tokens)-self.max_len)
        e = s + self.max_len
        return torch.tensor(self.tokens[s:e])
