import torch 
import time
import copy
import os
import argparse

import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import cosine_similarity

from torchvision.transforms import transforms

from datasets import DataTransform, CustomDataset
from model import Model
from loss import NTXentLoss

from transformers import AutoTokenizer 

os.environ['TOKENIZERS_PARALLELISM'] = "false"

parser = argparse.ArgumentParser(description="Training Script")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate")
parser.add_argument('--epochs', type=int, default=150, help="Number of epochs")

args = parser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate 
epochs = args.epochs

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

data_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

data_transform = DataTransform(data_transform)
dataset = CustomDataset("p10-trunc.csv", data_transform)

train_set, val_set = torch.utils.data.random_split(dataset,
        [round(len(dataset) * 0.75), round(len(dataset) * 0.25)])

dataloaders = {
        "train": torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8),
        "val": torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8)}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model(200).to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

criterion = NTXentLoss(device, batch_size, temperature=.2, alpha_weight=.5) 
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

def train_loop(dataloader, model, criterion, optimizer):

    model.train()

    size = len(dataloader.dataset)
    for batch, (xis, xls) in enumerate(dataloader):
        xls = tokenizer(list(xls), return_tensors="pt", truncation=True, padding=True, max_length=512)
        xls = {key: value.to(device) for key, value in xls.items()}
        xis = xis.to(device)

        zis, zjs = model(xis, xls)

        loss, _ = criterion(zis, zjs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(xis)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, criterion, best_acc, best_model_wts):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for xis, xls in dataloader:
            xls = tokenizer(xls, return_tensors="pt", truncation=True, padding=True, max_length=512)
            xis = xis.to(device)
            xls = {key: value.to(device) for key, value in xls.items()}

            zis, zjs = model(xis, xls)
            loss, _ = criterion(zis, zjs)
            test_loss += loss.item()

            # similarities = torch.matmul(zis, zjs.T) #cosine_similarity(zis, zjs)
            # predicted_indices = torch.argmax(similarities, dim=1)

            # correct += (predicted_indices == torch.arange(xis.size(0), device=device).sum().item())

    test_loss /= num_batches
    accuracy = correct / size

    # print(f"Validation Loss: {test_loss: .4f}, Accuracy: {accuracy:.4f}")
    
    if accuracy > best_acc:
        best_acc = accuracy 
        best_model_wts = copy.deepcopy(model.state_dict())

    return best_acc, best_model_wts

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

since = time.time()

try:    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloaders['train'], model.to(device), criterion, optimizer)
        best_acc, best_model_wts = test_loop(dataloaders['val'], model.to(device), criterion,
                                               best_acc, best_model_wts)
except KeyboardInterrupt:
    print("Interrupted, saving current model weights.")
    torch.save(model.state_dict(), "interrupted_model_weights.pth")

time_elapsed = time.time() - since

torch.save(best_model_wts, "best_model_weights.pth")

print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
print("Done!")
