import torch
from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from logger import setup_logger
from utils.environment import setup_environment, create_checkpoint_folder
from utils.wandb_setup import init_wandb
from utils.optimizer import setup_optimizer_scheduler

from utils.training.text_correction import text_correction_training

from dataloaders.mimic_dataloader import create_dataloaders_error_identification
from models.text_correction import Correction

import warnings
warnings.filterwarnings("ignore")

@hydra.main(config_path="./configs/models", config_name="error_classification")
def main(cfg: DictConfig) -> None:
    seed = cfg.main.seed
    setup_environment(seed)

    checkpoint_folder = cfg.main.checkpoint_folder
    create_checkpoint_folder(checkpoint_folder)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.main.biomegatron)
    dataloaders = create_dataloaders_error_identification(cfg.main.batch_size, cfg.data_transform, tokenizer)

    init_wandb(cfg)

    model = Correction.from_config(cfg, device)

    text_correction_training(model, dataloaders, 200, device)

if __name__ == "__main__":
    main()
