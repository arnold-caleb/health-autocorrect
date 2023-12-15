import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from utils.predictions.prediction_utils import calculate_similarities, combine_subwords, encode_text_and_image

from models import ImageTextGroundingModelHierarchical

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel
import timm

import math

from utils import fix_state_dict_keys
from hydra.utils import instantiate

data = pd.read_csv('/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/data/train_csv_script.csv')
report_descriptions = data['text'].tolist()

def cosine_similarity(image_embeddings, token_embeddings):
    image_embeddings_normalized = F.normalize(image_embeddings, p=2, dim=1) # [B, 128]
    token_embeddings_normalized = F.normalize(token_embeddings, p=2, dim=2) # [B, l, 128]

    return (image_embeddings_normalized.unsqueeze(1) * token_embeddings_normalized).sum(-1) # [B, l]

def encode_all_reports(texts, model, tokenizer, device="cuda:0"):
    model = ImageTextGroundingModelHierarchical.from_config(cfg, device)
    embeddings = []
    for text in texts:
        _, _, text_embedding = encode_text_and_image(text, torch.zeros((1, 3, 224, 224)).to(device), model, tokenizer)  # Placeholder image tensor
        embeddings.append(text_embedding.squeeze().cpu().numpy())
    return embeddings

all_report_embeddings = encode_all_reports(report_descriptions, model, tokenizer)


# class ReportRetrievalSystem:
#     def __init__(self, model, tokenizer, all_report_embeddings, report_descriptions):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.all_report_embeddings = all_report_embeddings
#         self.report_descriptions = report_descriptions

#     def retrieve_top_reports(self, query_image, top_k=5):
#         _, _, image_embedding = encode_text_and_image("", query_image, self.model, self.tokenizer)  # Placeholder text
#         image_embedding = image_embedding.cpu().numpy().squeeze()
        
#         similarity_scores = cosine_similarity(image_embedding.reshape(1, -1), self.all_report_embeddings).squeeze()
#         top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
#         top_reports = [self.report_descriptions[idx] for idx in top_indices]
#         return top_reports

# # Create a retrieval system
# retrieval = ReportRetrievalSystem(model, tokenizer, all_report_embeddings, report_descriptions)
