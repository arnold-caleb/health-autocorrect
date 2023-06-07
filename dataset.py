import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from PIL import Image

class RadiographDataset(Dataset):
    def __init__(self, csv, transform=None):
        self.df = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.df.iloc[idx]['image']).convert("RGB")
        text = self.df.iloc[idx]['text']

        sample = {"image": image, "text": text}
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

class CustomDataset(Dataset):
    def __init__(self, csv, transform=None):
        self.df = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.df.iloc[idx]['images']).convert("RGB")
            
        text_loc = self.df.iloc[idx]["text"] 

        with open(text_loc) as t:
            text = t.read()
            text = self.extract_findings(text)

        return self.transform({"image": image, "text": text})

    def extract_findings(self, text):
        lines = text.strip(' ').replace("\n", "").replace('_', '').strip(' ').split(" ")
        findings = False
        findings_lines = []
        better_finding_lines = []

        for line in lines:
            if "FINDINGS:" in line:
                findings = True
            if findings:
                findings_lines.append(line)

        for finding in findings_lines:
            better_finding_lines.append(finding.strip(' '))

        return " ".join(better_finding_lines)

class DataTransform(object):
    def __init__(self, transform):
        self.transform_image = transform

    def __call__(self, sample):
        image = self.transform_image(sample["image"])
        text = sample["text"]

        return image, text

