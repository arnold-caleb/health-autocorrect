import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel
import timm

import math

# from utils import fix_state_dict_keys
from hydra.utils import instantiate

def fix_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == 'n_averaged':
            continue  # Skip the unwanted key
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

# seems to work well overall...
class ImageTextGroundingModelHierarchical(nn.Module):
    def __init__(self, embed_dim, llm):
        super(ImageTextGroundingModelHierarchical, self).__init__()
        self.embed_dim = embed_dim # 2560 -> because of gatortron embed dims
        self.llm = llm

        self.text_attention = nn.MultiheadAttention(embed_dim, num_heads=8) 
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.vit.head = torch.nn.Linear(self.vit.head.in_features, 14) 
        self.image_attention = nn.MultiheadAttention(self.vit.embed_dim, num_heads=8) # self.vit.embed_dim = 768
        
        pretrained_weights = torch.load("/proj/vondrick/aa4870/vit_checkpoints/vit_checkpoint_epoch_32.pth")
        # pretrained_weights = torch.load("/proj/vondrick/aa4870/vit_checkpoints/binary_classifier.pth")
        
        fixed_weights = fix_state_dict_keys(pretrained_weights)
        self.vit.load_state_dict(fixed_weights)
        # self.vit.load_state_dict(torch.load("vit_base_all_best_encoder_weights.pth"), strict=False) # trained on CheXpert dataset

        self.vit.head = nn.Identity()
        self.vit_l1 = nn.Linear(self.vit.embed_dim, self.vit.embed_dim)
        self.vit_l2 = nn.Linear(self.vit.embed_dim, self.embed_dim) # project from 768 -> 2560

        self.text_llm = AutoModel.from_pretrained(llm)
        self.bert_l1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.bert_l2 = nn.Linear(self.embed_dim, self.embed_dim) # probably no need for this...

    def attention_pooling(self, outputs, attention_mask):
        # Apply multi-head attention to the text embeddings and sum the output.

        token_embeds = outputs[0].permute(1, 0, 2)  # seq_len x B x embed_dim
        attn_output, attn_output_weights = self.text_attention(token_embeds, token_embeds, token_embeds, key_padding_mask=(attention_mask==0))
        sum_embeds = attn_output.sum(dim=0)  # B x embed_dim
        
        return sum_embeds  # B x embed_dim

    def image_encoder(self, image):
        # Multi-scale feature extraction for images
        feature_maps = self.vit.forward_features(image) # [B, 197, 768]
        
        # Attention pooling
        feature_maps = feature_maps.permute(1, 0, 2)  # [197, B, 768]
        attn_output, _ = self.image_attention(feature_maps, feature_maps, feature_maps)  # [197, B, 768]
        h = attn_output.sum(dim=0)  # [B, 768]
        
        # proj to same dimension as that of text embedding 768 -> 2560
        x = self.vit_l1(h) # [B, 768]
        x = F.relu(x)
        x = self.vit_l2(x) # [B, embed_dim] ... for gatortron [B, 2560]

        return x

    def text_encoder(self, encoded_text, visualization=False):
        outputs = self.text_llm(**encoded_text)  # [B, N, 768] where N is number of tokens...
       
        if visualization: 
            with torch.no_grad(): # No gradient flow at visualization
                sentence_embeddings = outputs['last_hidden_state'] # [B, N, embed_dim]
                x = self.bert_l1(sentence_embeddings) # [B, N, embed_dim]
                x = F.relu(x) # [B, N, embed_dim]
                text_emb = self.bert_l2(x) # [B, N, embed_dim]
                return text_emb # [B, N, embed_dim]
        else:
            sentence_embeddings = self.attention_pooling(outputs, encoded_text["attention_mask"])  # [B, embed_dim] 
            x = self.bert_l1(sentence_embeddings)  # [B, embed_dim]   
            x = F.relu(x)  # [B, embed_dim]   
            text_emb = self.bert_l2(x)  # [B, embed_dim] ... for gatortron [B, 2560]

        return text_emb  # [B, embed_dim]   

    def forward(self, image, encoded_text):
        zis = self.image_encoder(image)
        zjs = self.text_encoder(encoded_text)
        
        return zis, zjs

    @classmethod
    def from_config(cls, cfg, device):
        model = instantiate(cfg.model).to(device)
        model = torch.nn.DataParallel(model, device_ids=cfg.main.device_ids)
        model.load_state_dict(torch.load("/proj/vondrick/aa4870/checkpoints_original_biomegatron/grounding_model_checkpoint_epoch_original_biomegatron_40.pth")) 

        return model