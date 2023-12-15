import torch
from torchvision.transforms import transforms
from dataset import (
    DataTransform, 
    MimicCxrDataset, 
    MimicCxrDatasetNegatives, 
    DataTransformNegatives,
    DataTransformImageTextDataset,
    TextImageDataset
    )

def get_transforms(data_transform_cfg):
    return transforms.Compose([
        transforms.Resize(data_transform_cfg.resize, interpolation=transforms.InterpolationMode.BOX),
        transforms.CenterCrop(data_transform_cfg.center_crop),
        transforms.RandomRotation(degrees=(0, 20)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(data_transform_cfg.normalize.mean, data_transform_cfg.normalize.std),
    ])

def split_dataset(full_dataset, train_ratio=0.75):
    train_len = round(len(full_dataset) * train_ratio)
    val_len = len(full_dataset) - train_len
    return torch.utils.data.random_split(full_dataset, [train_len, val_len])

def create_dataloader(dataset, batch_size, shuffle, num_workers=8):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def create_dataloaders(batch_size, data_transform_cfg):
    data_transform = DataTransform(get_transforms(data_transform_cfg))
    full_dataset = MimicCxrDataset(is_train=True, transform=data_transform)
    train_set, val_set = split_dataset(full_dataset)

    return {
        "train": create_dataloader(train_set, batch_size, shuffle=True),
        "val": create_dataloader(val_set, batch_size, shuffle=True),
    }

def create_dataloaders_negatives(batch_size, data_transform_cfg, negative_pool_size=5):
    data_transform = DataTransformNegatives(get_transforms(data_transform_cfg))
    full_dataset = MimicCxrDatasetNegatives(is_train=True, transform=data_transform, negative_pool_size=negative_pool_size)
    train_set, val_set = split_dataset(full_dataset)

    return {
        "train": create_dataloader(train_set, batch_size, shuffle=True),
        "val": create_dataloader(val_set, batch_size, shuffle=False),
    }


def create_dataloaders_error_identification(batch_size, data_transform_cfg, tokenizer):
    data_transform = DataTransformImageTextDataset(get_transforms(data_transform_cfg))
    full_dataset = TextImageDataset(tokenizer, data_transform)

    train_set, val_set = split_dataset(full_dataset)

    return {
        "train": create_dataloader(train_set, batch_size, shuffle=True),
        "val": create_dataloader(val_set, batch_size, shuffle=True)
    }