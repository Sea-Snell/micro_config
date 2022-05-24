from micro_config import MetaConfig, parse_args, deep_replace
from base_configs import AdamWConfig, LMModelConfig, TransformerConfig, WikiDataConfig
from general_train_loop import TrainLoop
import torch

# returns from unroll respect config dataclass references
# (i.e. if train_dataset is referenced twice in a hierarchy, the object is only loaded once) 
train_dataset = WikiDataConfig(f_path='data/wikitext-2-raw/wiki.train.raw', max_len=256)
eval_dataset = WikiDataConfig(f_path='data/wikitext-2-raw/wiki.valid.raw', max_len=256)

model = LMModelConfig(
            checkpoint_path=None, 
            strict_load=True, 
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
    use_wandb=False, 
    wandb_project='wikitext-2-lm', 
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset, 
    model=model, 
    optim=AdamWConfig(lr=3e-4, weight_decay=0.01), 
    epochs=1,
    max_steps=None,
    n_dloader_workers=0,
    bsize=16,
    eval_batches=8,
    grad_accum_steps=1,
    eval_every=1024,
    log_every=256,
    save_every=16384,
    save_checkpoint_dir='outputs/lm_checkpoints',
    optim_state_path=None,
    max_checkpoints=1,
)

if __name__ == "__main__":
    # metaconfig defines parameters for the configuration process
    # it could also include default parameters that you want to share broadly across objects
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metaconfig = MetaConfig(verbose=True, device=device)
    # parse_args parses the command line arguments into a dictionary
    # deep_replace implements a nested version of the standard dataclasses replace method
    train_config_script = deep_replace(train_config_script, **parse_args())
    # run the script
    train_config_script.unroll(metaconfig)
