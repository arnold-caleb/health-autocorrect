import torch
from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from logger import setup_logger
from utils.environment import setup_environment, create_checkpoint_folder
from utils.wandb_setup import init_wandb
from utils.optimizer import setup_optimizer_scheduler
from utils.training import run_training_token_classifier, save_best_model

from dataloaders.mimic_dataloader import create_dataloaders_error_identification
from models import ImageTextCorrection, ImageTextCorrection2

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

    # model = ImageTextCorrection.from_config(cfg, device) # concat approach
    model = ImageTextCorrection2.from_config(cfg, device) # pool and append approach depending on what you uncomment
    
    optimizer, scheduler = setup_optimizer_scheduler(cfg, model)

    # init_wandb(cfg)

    best_weights = run_training_token_classifier(model, dataloaders, optimizer, scheduler, device, tokenizer, cfg)
    # save_best_model(model, best_weights, cfg) 

if __name__ == "__main__":
    main()
