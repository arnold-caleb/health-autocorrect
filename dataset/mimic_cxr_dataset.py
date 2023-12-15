import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from PIL import Image

class MimicCxrDataset(Dataset):
    def __init__(self, is_train, transform=None):
        self.is_train = is_train
        self.transform = transform
        if is_train:
            self.df = pd.read_csv("/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/data/train_csv_script.csv")
        else:
            self.df = pd.read_csv("/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/data/error_test.csv") # won't use this for now

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.df.iloc[idx]['image'] if self.is_train else self.df.iloc[idx]['image_path']
        image = Image.open(image_path).convert("RGB")

        text = self.df.iloc[idx]["text"] if self.is_train else self.df.iloc[idx]['error_desc']

        # When is_train is False, also return error location
        if not self.is_train:
            error_loc = self.df.iloc[idx]["error_loc"]
            return self.transform({"image": image, "text": text, "error_loc": error_loc})

        return self.transform({"image_path": image_path,"image": image, "text": text})

    def remove_extra_whitespace(self, text):
        import re

        stripped_text = text.strip(' ').replace("_", "")
        no_extra_whitespace_text = re.sub('\s+', ' ', stripped_text)
        return no_extra_whitespace_text

    def extract_findings(self, text):
        lines = text.replace('_', '').strip().split("\n")
        findings = False
        findings_lines = []

        for line in lines:
            if "FINDINGS:" in line:
                findings = True
            if findings:
                findings_lines.append(line.strip())

        return " ".join(findings_lines)

class DataTransform(object):
    def __init__(self, transform):
        self.transform_image = transform

    def __call__(self, sample):
        image_key = "image" if "image" in sample else "image_path"
        text_key = "text" if "text" in sample else "error_desc"

        image_path = sample["image_path"]
        image = self.transform_image(sample[image_key])
        text = sample[text_key]

        if "error_loc" in sample:
            return image, text, sample["error_loc"]

        return image_path, image, text


class MimicCxrDatasetNegatives(Dataset):

    def __init__(self, is_train, transform=None, negative_pool_size=5):
        self.is_train = is_train
        self.transform = transform
        self.df = pd.read_csv("/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/data/train_csv_script.csv")
        self.negative_pool_size = negative_pool_size
        self.negative_pools = self.build_negative_pools()

    def build_negative_pools(self):
        negative_pools = {}
        for idx in range(len(self.df)):
            negative_pools[idx] = self.draw_negative_pool(idx)
        return negative_pools

    def draw_negative_pool(self, idx):
        neg_idxs = []
        while len(neg_idxs) < self.negative_pool_size:
            neg_idx = torch.randint(len(self.df), (1,)).item()
            if neg_idx != idx:
                neg_idxs.append(neg_idx)
        return neg_idxs

    def update_negative_pool(self, idx):
        self.negative_pools[idx] = self.draw_negative_pool(idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.df.iloc[idx]['image']
        image = Image.open(image_path).convert("RGB")

        text = self.df.iloc[idx]["text"]

        neg_idx = np.random.choice(self.negative_pools[idx])
        neg_text = self.df.iloc[neg_idx]["text"]

        return self.transform({"image": image, "text": text, "neg_text": neg_text})

    # After each epoch, update the negative pools
    def update_negative_pools(self):
        for idx in range(len(self.df)):
            self.update_negative_pool(idx)

class DataTransformNegatives(object):
    def __init__(self, transform):
        self.transform_image = transform

    def __call__(self, sample):
        
        image = self.transform_image(sample["image"])
        text = sample["text"]
        neg_text = sample["neg_text"]

        return image, text, neg_text
