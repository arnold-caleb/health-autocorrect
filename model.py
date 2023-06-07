import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel
import timm

import math

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
        
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.context_vector = nn.Parameter(torch.randn(self.head_dim))

#     def forward(self, input, attention_mask=None):
#         batch_size, seq_length, _ = input.size()

#         query = self.query(input).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
#         key = self.key(input).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
#         value = self.value(input).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

#         scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         if attention_mask is not None:
#             attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # expand dims for heads and head_dim
#             scores = scores.masked_fill(attention_mask == 0, -1e9)
        
#         weights = torch.softmax(scores, dim=-1)
#         output = (weights @ value).transpose(1, 2).reshape(batch_size, seq_length, self.embed_dim)
        
#         return output

class ImageTextGroundingModelHierarchical(nn.Module):
    def __init__(self, out_dim):
        super(ImageTextGroundingModelHierarchical, self).__init__()
        self.out_dim = out_dim 

        # self.attention = nn.Linear(768, 1)  # Attention weights for each token
        self.attention = nn.MultiheadAttention(768, num_heads=8) # For if there is any change in the performance of the model or not
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.vit.head = torch.nn.Linear(self.vit.head.in_features, 14) 
        
        pretrained_weights = torch.load("/proj/vondrick/aa4870/vit_checkpoints/vit_checkpoint_epoch_32.pth")
        fixed_weights = fix_state_dict_keys(pretrained_weights)
        self.vit.load_state_dict(fixed_weights)
        # self.vit.load_state_dict(torch.load("vit_base_all_best_encoder_weights.pth"), strict=False) # trained on CheXpert dataset

        self.vit.head = nn.Identity()
        self.vit_l1 = nn.Linear(self.vit.embed_dim, self.vit.embed_dim)
        self.vit_l2 = nn.Linear(self.vit.embed_dim, self.out_dim)

        self.text_llm = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert_l1 = nn.Linear(768, 768)
        self.bert_l2 = nn.Linear(768, self.out_dim) 

    def attention_pooling(self, outputs, attention_mask):
        token_embeds = outputs[0].permute(1, 0, 2)  # seq_len x B x 768
        attn_output, attn_output_weights = self.attention(token_embeds, token_embeds, token_embeds, key_padding_mask=(attention_mask==0))
        sum_embeds = attn_output.sum(dim=0)  # B x 768
        
        return sum_embeds.permute(0, 1)  # B x 768

    def mean_pooling(self, outputs, attention_mask):
        token_embeds = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeds.size())
        sum_embeds = torch.sum(token_embeds * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeds / sum_mask

    def image_encoder(self, image):
        # Multi-scale feature extraction for images
        feature_maps = self.vit.forward_features(image) # [B, 197, 768]
        h = torch.mean(feature_maps, dim=1) # [B, 768]
        
        x = self.vit_l1(h) # [B, 768]
        x = F.relu(x)
        x = self.vit_l2(x) # [B, out_dim]

        return x

    def text_encoder(self, encoded_text, visualization=False):
        outputs = self.text_llm(**encoded_text)  # [B, N, 768] 

        with torch.no_grad():
            if visualization:
                sentence_embeddings = outputs['last_hidden_state'] # [B, N, 768]
                x = self.bert_l1(sentence_embeddings) # [B, N, 768]
                x = F.relu(x) # [B, N, 768]
                text_emb = self.bert_l2(x) # [B, N, out_dim]
                return text_emb # [B, N, out_dim]
            else:
                sentence_embeddings = self.attention_pooling(outputs, encoded_text["attention_mask"])  # [B, 768] # was initially using mean_pooling
            x = self.bert_l1(sentence_embeddings)  # [B, 768]   (at visualization: [B, N, 768])
            x = F.relu(x)  # [B, 768]   (at visualization: [B, N, out_dim])
            text_emb = self.bert_l2(x)  # [B, out_dim]   (at visualization: [B, N, out_dim])

        return text_emb  # [B, out_dim]   (at visualization: [B, N, out_dim])

    def forward(self, image, encoded_text):
        zis = self.image_encoder(image)
        zjs = self.text_encoder(encoded_text)
        
        return zis, zjs

def fix_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == 'n_averaged':
            continue  # Skip the unwanted key
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

