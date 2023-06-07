import torch
import time
import copy
import os
import argparse

import random
import numpy as np
import wandb

from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cosine_similarity

from torchvision.transforms import transforms

from dataset import DataTransform, CustomDataset, RadiographDataset
from model import ImageTextGroundingModelHierarchical
from loss import NTXentLoss, TripletLoss

from transformers import AutoTokenizer

from utils import preprocess_image, get_top_k_tokens, train, validate

def create_dataloaders(batch_size):
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    data_transform = DataTransform(data_transform)
    dataset = CustomDataset("locations.csv", data_transform)

    train_set, val_set = torch.utils.data.random_split(dataset, [round(len(dataset) * 0.75), round(len(dataset) * 0.25)])

    dataloaders = {
        "train": torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8),
        "val": torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8)}

    return dataloaders


def initialize_model(device):
    model = ImageTextGroundingModelHierarchical(128).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 3, 4, 5, 6, 7])
    # model.load_state_dict(torch.load("best_model_weights_grounding_cv12.pth"))

    return model


def setup_optimizer_scheduler(model, learning_rate, num_epochs):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    return optimizer, scheduler


def main(args):
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    # Create the checkpoint folder if it doesn't exist
    checkpoint_folder = "/proj/vondrick/aa4870/checkpoints_cv12"
    os.makedirs(checkpoint_folder, exist_ok=True)

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    patience = args.patience
    checkpoint_interval = args.checkpoint_interval

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    dataloaders = create_dataloaders(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = initialize_model(device)

    ntxent_loss = NTXentLoss(temperature=.1, device=device)
    optimizer, scheduler = setup_optimizer_scheduler(model, learning_rate, num_epochs)

    wandb.init(project="Image Text Grounding Project Summer 2023", config=args)

    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())

    no_improvement_counter = 0

    for epoch in range(24, 24 + num_epochs):
        total_loss = 0.0

        model.train()
        total_loss = train(model, dataloaders['train'], optimizer, device, tokenizer, ntxent_loss, epoch)

        model.eval()
        val_loss = validate(model, dataloaders['val'], device, tokenizer, ntxent_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= patience:
            print("Early stopping triggered, no improvement for {} consecutive epochs.".format(patience))
            break

        # Save model checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_folder, f"grounding_model_checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        scheduler.step()

    model.load_state_dict(best_weights)
    torch.save(best_weights, "best_model_weights_grounding_cv_12.pth")
    wandb.save("best_model_weights_grounding_cv_12.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=40, help="Number of epochs")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--checkpoint_interval', type=int, default=4, help="Model checkpoint interval in epochs")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args)
