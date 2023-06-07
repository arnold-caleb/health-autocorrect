import os
import argparse
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import timm
from torch.optim.lr_scheduler import StepLR
from torch.optim.swa_utils import AveragedModel, update_bn
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
import wandb
import time

from datasets.datasets import ChestXpertDataset, ChestXpertDataTransform

def one_hot_encode(labels, num_classes, missing_label_strategy="positive"):
    one_hot_labels = torch.zeros(labels.size(0), num_classes)

    if missing_label_strategy == "positive":
        labels[labels == -1] = 1
    elif missing_label_strategy == "negative":
        labels[labels == -1] = 0

    labels = labels.to(torch.int64)
    return labels.float()


def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    batch_counter = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        start = time.time()
        inputs = inputs.to(device)
        labels = one_hot_encode(labels, num_classes, "negative").to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        loss = criterion(outputs, labels)

        time_forward = time.time()

        loss.backward()
        optimizer.step()

        time_backward = time.time()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum((preds == labels.data).all(axis=1)).item()

        batch_accuracy = torch.sum((preds == labels.data).all(axis=1)).item() / inputs.size(0)
        wandb.log({'train Batch': batch_counter + 1, 'Batch Acc': batch_accuracy})
        time_end = time.time()

        # print(f'train Batch {batch_counter + 1} Acc: {batch_accuracy:.4f} Times --> End: {(time_end - start):.2f}, Forward pass: {(time_forward - start):.2f}, Backward_pass: {(time_backward - time_forward):.2f}')
        batch_counter += 1

    return running_loss, running_corrects


def val_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    batch_counter = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            start = time.time()
            inputs = inputs.to(device)
            labels = one_hot_encode(labels, num_classes, "negative").to(device)

            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            loss = criterion(outputs, labels)

            time_forward = time.time()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum((preds == labels.data).all(axis=1)).item()

            batch_accuracy = torch.sum((preds == labels.data).all(axis=1)).item() / inputs.size(0)
            wandb.log({'val Batch': batch_counter + 1, 'Batch Acc': batch_accuracy})
            time_end = time.time()

            # print(f'val Batch {batch_counter + 1} Acc: {batch_accuracy:.4f} Times --> End: {(time_end - start):.2f}, Forward pass: {(time_forward - start):.2f}')
            batch_counter += 1

    return running_loss, running_corrects


def train_val(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, swa_start, model_path):
    best_acc = 0.0
    best_weights = None
    num_classes = 14
    swa_model = AveragedModel(model)

    # Create the checkpoint folder if it doesn't exist
    checkpoint_folder = "/proj/vondrick/aa4870/vit_checkpoints"
    os.makedirs(checkpoint_folder, exist_ok=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        train_loss, train_corrects = train_epoch(model, dataloaders['train'], criterion, optimizer, device, num_classes)
        scheduler.step()
        if epoch >= swa_start:
            swa_model.update_parameters(model)

        val_loss, val_corrects = val_epoch(model, dataloaders['val'], criterion, device, num_classes)

        train_epoch_loss = train_loss / len(dataloaders['train'].dataset)
        train_epoch_acc = float(train_corrects) / len(dataloaders['train'].dataset)

        wandb.log({'train Loss': train_epoch_loss, 'Acc': train_epoch_acc})
        print(f'train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.4f}')

        val_epoch_loss = val_loss / len(dataloaders['val'].dataset)
        val_epoch_acc = float(val_corrects) / len(dataloaders['val'].dataset)

        wandb.log({'val Loss': val_epoch_loss, 'Acc': val_epoch_acc})
        print(f'val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_weights = swa_model.state_dict()

        # Save model checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_folder, f"vit_checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    update_bn(dataloaders['train'], swa_model, device=device)
    print(f'Best val Acc: {best_acc:.4f}')
    torch.save(best_weights, model_path)

def main(args):
    wandb.init(project="From Scratch (ChestXPert) Encoder Classifier for Image-Text Grounding problem", config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_transform = ChestXpertDataTransform(data_transforms['train'])
    full_dataset = ChestXpertDataset(args.data_csv, data_transform)

    train_set, val_set = torch.utils.data.random_split(full_dataset,
        [round(len(full_dataset) * 0.75), round(len(full_dataset) * 0.25)])

    sample_weights_train = train_set.dataset.get_sample_weights(train_set.indices)
    sample_weights_val = val_set.dataset.get_sample_weights(val_set.indices)

    sampler_train = torch.utils.data.WeightedRandomSampler(sample_weights_train, len(train_set))
    sampler_val = torch.utils.data.WeightedRandomSampler(sample_weights_val, len(val_set))

    dataloaders = {
        "train": torch.utils.data.DataLoader(train_set, sampler=sampler_train, batch_size=args.batch_size, num_workers=8),
        "val": torch.utils.data.DataLoader(val_set,sampler=sampler_val, batch_size=args.batch_size, num_workers=8)}

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=14)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # BCE(y, p) = -[y * log(p) + (1 - y) * log(1 - p)]
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_learning_rate)

    model.load_state_dict(torch.load("/proj/vondrick/aa4870/vit_checkpoints/vit_checkpoint_epoch_32.pth"))

    # Train and evaluate the model
    train_val(model, dataloaders, criterion, optimizer, scheduler, device, args.epochs, args.swa_start, args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the ViT model on a the chexpert dataset.")
    parser.add_argument("--data-csv", type=str, default="/proj/vondrick/aa4870/chestXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/train.csv", help="Path to the csv containing the files paths and their classes")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR scheduler.")
    parser.add_argument("--max-lr", type=float, default=1e-3, help="Maximum learning rate for OneCycleLR scheduler.")
    parser.add_argument("--swa-start", type=int, default=int(0.75 * 30), help="Epoch to start SWA.")
    parser.add_argument("--swa-lr", type=float, default=1e-3, help="SWA learning rate.")
    parser.add_argument("--model-path", type=str, default="vit_base_all_best_encoder_weights_fresh.pth", help="Path to save the best model weights.")
    parser.add_argument("--checkpoint-interval", type=int, default=4, help="Model checkpoint interval in epochs.")

    args = parser.parse_args()
    main(args)
