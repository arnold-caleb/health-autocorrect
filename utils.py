import torch
from tqdm import tqdm
from PIL import Image

import torch.nn.functional as F
import wandb

from transformers import AutoTokenizer

from helpers import extract_findings, preprocess_image
from distance_metrics import *
  
def encode_text_and_image(text, image_tensor, model, tokenizer):
    text = extract_findings(text)
    image_tensor = image_tensor.to("cuda:0")

    model.eval()
    encoded_text = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        token_embeddings = model.module.text_encoder(encoded_text, visualization=True) # [B, l, 128]
        image_embeddings = model.module.image_encoder(image_tensor) # [B, 128]

    return encoded_text, token_embeddings, image_embeddings


def calculate_similarities(token_embeddings, image_embeddings):
    k = token_embeddings.shape[1]

    # similarities = cosine_similarity(image_embeddings, token_embeddings)
    # similarities = euclidean_similarity(image_embeddings, token_embeddings)
    similarities = manhattan_similarity(image_embeddings, token_embeddings) # seems to perform better than the others
    # similarities = chebyshev_similarity(image_embeddings, token_embeddings)
    # similarities = minkowski_similarity(image_embeddings, token_embeddings)

    top_k_values, top_k_indices = torch.topk(similarities, k, dim=1) 
    top_k_values = F.softmax(top_k_values, dim=1)
    return top_k_values.tolist(), top_k_indices.tolist()


def combine_subwords(encoded_text, top_k_values, tokenizer):
    combined_tokens, combined_indices, combined_probs = [], [], []
    subword_probs = []
    
    for i in range(len(encoded_text['input_ids'][0])):
        token = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0][i].item())
        
        if token.startswith("##"):
            combined_tokens[-1] += token[2:]
            subword_probs.append(top_k_values[0][i])
            combined_probs[-1] = sum(subword_probs) / len(subword_probs)  # Average probabilities of subwords
        else:
            combined_tokens.append(token)
            combined_indices.append(i)  # Save the original index of each combined token
            subword_probs = [top_k_values[0][i]]
            combined_probs.append(subword_probs[0])

    return combined_tokens, combined_indices, combined_probs


def get_top_k_tokens(image_tensor, text, model, tokenizer, k=10):
    encoded_text, token_embeddings, image_embeddings = encode_text_and_image(text, image_tensor, model, tokenizer)
    top_k_values, top_k_indices = calculate_similarities(token_embeddings, image_embeddings)
    
    # Unnecessary tokens
    unneccessary_tokens = ['and', 'provided', ':', 'these', 'e', ',', '.', 'the', 'of', 'or', 'in', 'for', 'findings', 'there', 'at', 'is', 'a', 'are', 'as', '[PAD]', '[SEP]', '[CLS]']

    # Get the original order of the tokens
    original_order_indices = sorted(range(len(top_k_indices[0])), key=lambda x: top_k_indices[0][x])

    # Pre-combine subwords into words for the whole text
    combined_tokens, combined_indices, combined_probs = combine_subwords(encoded_text, top_k_values, tokenizer)
    
    # Get the tokens in original order and remove unnecessary tokens
    final_tokens, final_probs = [], []
    for i in original_order_indices:
        # Find the index of i in combined_indices
        if i in combined_indices:
            idx = combined_indices.index(i)
            token = combined_tokens[idx]
            prob = combined_probs[idx]
            
            if token not in unneccessary_tokens:
                final_tokens.append(token)
                final_probs.append(prob)

    # Sort tokens by their corresponding probabilities
    sorted_tokens_probs = sorted(zip(final_tokens, final_probs), key=lambda x: x[1], reverse=True)

    # Unzip tokens and probabilities
    sorted_tokens, sorted_probs = zip(*sorted_tokens_probs)

    return text, sorted_tokens, sorted_probs


def train(model, dataloader, optimizer, device, tokenizer, ntxent_loss, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (xis, xls) in enumerate(tqdm(dataloader)):
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

        # if batch_idx % 10 == 0:
        #     print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    avg_loss = running_loss / len(dataloader)

    print(f"Epoch {epoch}, Avg Loss: {avg_loss}")
    wandb.log({"train_loss": avg_loss, "epoch": epoch})

    return avg_loss

def validate(model, dataloader, device, tokenizer, ntxent_loss):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (xis, xls) in enumerate(tqdm(dataloader)):
            xis = xis.to(device)
            text_batch = xls
            text_encoded = tokenizer.batch_encode_plus(list(text_batch), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            text_encoded = {key: value.to(device) for key, value in text_encoded.items()}

            zis, zjs = model(xis, text_encoded)

            loss = ntxent_loss(zis, zjs)

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)

    print(f"Validation Loss: {avg_loss}")
    wandb.log({"val_loss": avg_loss})

    return avg_loss

