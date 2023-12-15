import torch
import wandb

from tqdm import tqdm
import numpy as np

from sklearn.metrics import classification_report
import itertools

import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()

def compute_neg_weight(dataloader, n_batches=10):
    total_positives = 0
    total_negatives = 0

    for _, labels, _, _, _ in itertools.islice(dataloader, n_batches):
        total_negatives += (labels == 0).sum() 
        total_positives += (labels == 1).sum() 

    neg_weight = total_positives.float() / total_negatives.float()
    return neg_weight

def train_token_classifier(cfg, model, dataloaders, epoch, device, optimizer, tokenizer):
    model.train()
    running_loss = 0.0

    all_preds = []
    all_labels = []

    neg_weight = compute_neg_weight(dataloaders).to(device)
    criterion = FocalLoss()

    clip_value = 1.0  

    for batch_idx, (images, labels, neg_reports, reports, _, _) in enumerate(tqdm(dataloaders)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        encoded_neg_reports = tokenizer.batch_encode_plus(neg_reports, padding="max_length", truncation=True, max_length=200, return_tensors="pt")
        encoded_neg_reports = {key: value.to(device) for key, value in encoded_neg_reports.items()}

        optimizer.zero_grad()

        raw_output = model(images, encoded_neg_reports)
        output = torch.sigmoid(raw_output)  # Ensure output is between [0, 1]
        
        # Check for any invalid outputs and labels
        if torch.any(output < 0) or torch.any(output > 1):
            print("Invalid output detected!")
            print(output)
        if torch.any((labels != 0) & (labels != 1)):
            print("Invalid label detected!")
            print(labels)

        loss = criterion(output.view(-1), labels.view(-1).float())
        loss.backward()

        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = (output > 0.9).float()
        all_preds.extend(preds.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

        if (batch_idx % 50) == 0:
            wandb.log({"Step loss": loss})
            # print("Step Loss: ", loss)

    epoch_loss = running_loss / len(dataloaders.dataset)

    print('Training Loss: {:.4f}'.format(epoch_loss))
    print('Classification Report:')

    report = classification_report(all_labels, all_preds, target_names=['error', 'no error'], output_dict=True)
    print(report)

    return epoch_loss, report





































## this is really old code, i don't remember what it was for 

def train(model, dataloader, optimizer, device, tokenizer, ntxent_loss, epoch, logger):
    running_loss = 0.0

    for batch_idx, (_, xis, xls) in enumerate(tqdm(dataloader)):
        xis = xis.to(device)
        text_batch = xls
        text_encoded = tokenizer.batch_encode_plus(list(text_batch), padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        text_encoded = {key: value.to(device) for key, value in text_encoded.items()}

        zis, zjs = model(xis, text_encoded) 
        loss = ntxent_loss(zis, zjs) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)

    logger.info(f"Epoch {epoch}, Avg Loss: {avg_loss}")
    wandb.log({"train_loss": avg_loss})

    return avg_loss

def train_negatives(cfg, model, dataloaders, epoch, device, optimizer, tokenizer, classifier_ntxent_loss, logger):
    model.train()
    losses = []
    pos = []
    neg = []
    correct_predictions = 0
    total_predictions = 0

    binary_criterion = torch.nn.BCEWithLogitsLoss()

    for step, (images, reports, neg_reports) in enumerate(tqdm(dataloaders)):
        images = images.to(device)
        batch_size = images.size(0)
        
        # Generate targets for positive and negative pairs
        pos_target = torch.ones(batch_size, dtype=torch.float32).to(device) 
        neg_target = torch.zeros(batch_size, dtype=torch.float32).to(device)
        targets = torch.cat([pos_target, neg_target], dim=0)  # concatenate along batch dimension

        encoded_reports = tokenizer.batch_encode_plus(reports, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded_reports = {key: value.to(device) for key, value in encoded_reports.items()}
        encoded_neg_reports = tokenizer.batch_encode_plus(neg_reports, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded_neg_reports = {key: value.to(device) for key, value in encoded_neg_reports.items()}

        optimizer.zero_grad()

        # Positive pairs
        pos_output, pos_zis, pos_zjs = model(images, encoded_reports)
        pos_loss = classifier_ntxent_loss(pos_zis, pos_zjs) + binary_criterion(pos_output.squeeze(), pos_target)
        pos.append(pos_output.detach().cpu())
        
        predictions = torch.sigmoid(pos_output) >= 0.5
        correct_predictions += (predictions == pos_target.float()).sum().item()
        total_predictions += predictions.numel()

        # Negative pairs
        neg_output, _ , _ = model(images, encoded_neg_reports)  # We don't need the embeddings for negative pairs
        neg_loss = binary_criterion(neg_output.squeeze(), neg_target)  # No contrastive loss for negative pairs
        neg.append(neg_output.detach().cpu())
        
        predictions = torch.sigmoid(neg_output.squeeze()) < 0.5
        correct_predictions += (predictions == neg_target.float()).sum().item()
        total_predictions += predictions.numel()

        loss = pos_loss + neg_loss  # Total loss is sum of losses for positive and negative pairs
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % cfg.main.checkpoint_interval == 0:
            logger.info(f"Train Epoch: {epoch} [{step * len(images)}/{len(dataloaders.dataset)} ({100. * step / len(dataloaders):.0f}%)]\tLoss: {loss.item():.6f}")

    accuracy = correct_predictions / total_predictions
    logger.info(f'====> Epoch: {epoch} Average loss: {np.mean(losses):.4f}')
    logger.info(f'====> Epoch: {epoch} Accuracy: {accuracy:.4f}')

    return np.mean(losses), torch.cat(pos).numpy(), torch.cat(neg).numpy(), accuracy
