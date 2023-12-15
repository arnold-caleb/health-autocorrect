import wandb
from omegaconf import DictConfig

def init_wandb(cfg: DictConfig):
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=dict(cfg.main))
