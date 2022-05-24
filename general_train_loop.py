from dataclasses import dataclass, asdict
from typing import Optional, Any
import torch
from src.utils import combine_logs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from micro_config import  ConfigScript, convert_path
from collections import deque
import os
import json
import pickle as pkl

# scripts and configs are unified. unroll() defines the scipt.
@dataclass
class TrainLoop(ConfigScript):
    use_wandb: bool=False
    wandb_project: str=''
    train_dataset: Any=None
    eval_dataset: Any=None
    model: Any=None
    optim: Any=None
    epochs: int=10
    max_steps: Optional[int]=None
    n_dloader_workers: int=0
    bsize: int=32
    eval_batches: int=16
    grad_accum_steps: int=1
    eval_every: int=1
    log_every: int=1
    save_every: Optional[int]=None
    save_checkpoint_dir: str=''
    optim_state_path: Optional[str]=None
    max_checkpoints: int=1

    # implement __post_init__ to handle any final edits to config parameters
    def __post_init__(self):
        # convert_path converts paths reletive to project root into absolute paths
        self.save_checkpoint_dir = convert_path(self.save_checkpoint_dir)
        self.optim_state_path = convert_path(self.optim_state_path)

    def out_log(self, logs):
        print(logs)
        if self.use_wandb:
            wandb.log(logs)
        return logs
    
    def unroll(self, metaconfig):
        print('using config:', asdict(self))
        print('using device:', metaconfig.device)
        # save config as json or pickle or both
        if not os.path.exists(self.save_checkpoint_dir):
            os.makedirs(self.save_checkpoint_dir)
        with open(os.path.join(self.save_checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(asdict(self), f)
        with open(os.path.join(self.save_checkpoint_dir, 'config.pkl'), 'wb') as f:
            pkl.dump(self, f)
        if self.use_wandb:
            wandb.init(project=self.wandb_project, config=asdict(self))
        device = metaconfig.device
        train_dataset = self.train_dataset.unroll(metaconfig)
        eval_dataset = self.eval_dataset.unroll(metaconfig)
        model = self.model.unroll(metaconfig)
        model.train()
        train_dataloader = DataLoader(train_dataset, num_workers=self.n_dloader_workers, batch_size=self.bsize)
        eval_dataloader = DataLoader(eval_dataset, num_workers=self.n_dloader_workers, batch_size=self.bsize)
        optim = self.optim.unroll(metaconfig)(model)
        step = 0
        train_logs_accum = []
        best_loss = float('inf')
        saved_checkpoints = deque([])
        for epoch in range(self.epochs):
            for x in tqdm(train_dataloader):
                loss, logs = model.get_loss(x.to(device))
                (loss / self.grad_accum_steps).backward()
                train_logs_accum.append(logs)
                if (step + 1) % self.grad_accum_steps == 0:
                    optim.step()
                    optim.zero_grad()
                if (step + 1) % self.log_every == 0:
                    out_log = self.out_log({'train': combine_logs(train_logs_accum), 'step': (step+1)})
                    train_logs_accum = []
                if (step+1) % self.eval_every == 0:
                    model.eval()
                    eval_logs_accum = []
                    with torch.no_grad():
                        for i, val_x in tqdm(enumerate(eval_dataloader)):
                            if i >= self.eval_batches:
                                break
                            _, val_logs = model.get_loss(val_x.to(device))
                            eval_logs_accum.append(val_logs)
                    out_log = self.out_log({'val': combine_logs(eval_logs_accum), 'step': (step+1)})
                    eval_logs_accum = []
                    if out_log['val']['loss'] < best_loss:
                        print('new best eval loss! Saving ...')
                        if not os.path.exists(self.save_checkpoint_dir):
                            os.makedirs(self.save_checkpoint_dir)
                        torch.save(model.state_dict(),
                                    os.path.join(self.save_checkpoint_dir, 'model.pkl'))
                        torch.save(optim.state_dict(), os.path.join(self.save_checkpoint_dir, 'optim.pkl'))
                        print('saved.')
                        best_loss = out_log['val']['loss']
                    model.train()
                if self.save_every is not None and (step + 1) % self.save_every == 0:
                    print('saving checkpoint...')
                    if not os.path.exists(self.save_checkpoint_dir):
                        os.makedirs(self.save_checkpoint_dir)
                    if (self.max_checkpoints is not None) and (len(saved_checkpoints) >= self.max_checkpoints):
                        os.system('rm -rf %s' % (saved_checkpoints.popleft()))
                    torch.save(model.state_dict(),
                                os.path.join(self.save_checkpoint_dir, 'model_%d.pkl' % (step)))
                    saved_checkpoints.append(os.path.join(self.save_checkpoint_dir, 'model_%d.pkl' % (step)))
                    print('saved.')
                step += 1
                if self.max_steps is not None and step >= self.max_steps:
                    break
        return model
