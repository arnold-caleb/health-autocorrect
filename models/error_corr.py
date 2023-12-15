import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
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

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class ImageTextCorrection(nn.Module):
    def __init__(self, embed_dim, llm, num_self_attention_layers=2, num_classes=1):
        super(ImageTextCorrection, self).__init__()
        self.embed_dim = embed_dim 
        self.llm = llm
        self.num_classes = num_classes

        self.text_attention = nn.MultiheadAttention(embed_dim, num_heads=8) 
        self.text_attention.apply(init_weights)
        
        # for the image encoder
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.vit.head = torch.nn.Linear(self.vit.head.in_features, 14) 
        self.image_attention = nn.MultiheadAttention(self.vit.embed_dim, num_heads=8) 
        self.image_attention.apply(init_weights)
        
        pretrained_weights = torch.load("/proj/vondrick/aa4870/vit_checkpoints/vit_checkpoint_epoch_32.pth")
        
        fixed_weights = fix_state_dict_keys(pretrained_weights)
        self.vit.load_state_dict(fixed_weights)
        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.head = nn.Identity()
        self.vit_l1 = nn.Linear(self.vit.embed_dim, self.vit.embed_dim)
        # self.vit_ln1 = nn.LayerNorm(self.vit.embed_dim)  
        self.vit_l2 = nn.Linear(self.vit.embed_dim, self.embed_dim) 
        # self.vit_ln2 = nn.LayerNorm(self.embed_dim)  

        # for the text encoder
        self.text_llm = AutoModel.from_pretrained(llm)
        for param in self.text_llm.parameters():
            param.requires_grad = False

        self.bert_l1 = nn.Linear(self.embed_dim, self.embed_dim)
        # self.bert_ln1 = nn.LayerNorm(self.embed_dim)  
        self.bert_l2 = nn.Linear(self.embed_dim, self.embed_dim) 
        # self.bert_ln2 = nn.LayerNorm(self.embed_dim)  

        # For attention on fused embeddings
        self.self_attention_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim * 2, num_heads=4) for _ in range(num_self_attention_layers)])
        for layer in self.self_attention_layers:
            layer.apply(init_weights)

        # For final per token classification
        self.classifier = nn.Linear(embed_dim * 2, num_classes)
        self.classifier.apply(init_weights)

    def attention_pooling(self, outputs, attention_mask):

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
        # x = self.vit_ln1(x)
        x = F.relu(x)
        x = self.vit_l2(x) # [B, embed_dim] ... for gatortron [B, 2560]
        # x = self.vit_ln2(x)

        return x

    def text_encoder(self, encoded_text, visualization=False):
        outputs = self.text_llm(**encoded_text)  # [B, N, 768] where N is number of tokens...
       
        if visualization: 
            # with torch.no_grad(): # No gradient flow at visualization
            sentence_embeddings = outputs['last_hidden_state'] # [B, N, embed_dim]
            x = self.bert_l1(sentence_embeddings) # [B, N, embed_dim]
            # x = self.bert_ln1(x)
            x = F.relu(x) # [B, N, embed_dim]
            x = self.bert_l2(x) # [B, N, embed_dim]
            # text_emb = self.bert_ln2(x)
            return x # [B, N, embed_dim]
        else:
            sentence_embeddings = self.attention_pooling(outputs, encoded_text["attention_mask"])  # [B, embed_dim] 
            x = self.bert_l1(sentence_embeddings)  # [B, embed_dim]   
            # x = self.bert_ln1(x)
            x = F.relu(x)  # [B, embed_dim]   
            x = self.bert_l2(x)  # [B, embed_dim] ... for gatortron [B, 2560]
            # text_emb = self.bert_ln2(x)

        return text_emb  # [B, embed_dim]   

    def forward(self, image, encoded_text):
        
        # Get image and token embeddings
        zis = self.image_encoder(image) # [B, 2560]
        zjs = self.text_encoder(encoded_text, visualization=True)  # [B, N, 2560]

        # Normalize embeddings
        zis = F.normalize(zis, p=2, dim=-1)
        zjs = F.normalize(zjs, p=2, dim=-1)  

        # Concatenate each token embedding with the image embedding
        zis_exp = zis.unsqueeze(1).expand(-1, zjs.shape[1], -1) 
        fused_embeddings = torch.cat([zjs, zis_exp], dim=-1) # [b, N, embed_dim * 2])
        
        # Self-Attention on the concatenated embeddings
        for layer in self.self_attention_layers:
            fused_embeddings, fused_attention_weights = layer(fused_embeddings, fused_embeddings, fused_embeddings)

        # Per token classification
        output = self.classifier(fused_embeddings)
        
        return output

    @classmethod
    def from_config(cls, cfg, device):
        model = instantiate(cfg.model).to(device)
        model = torch.nn.DataParallel(model, device_ids=cfg.main.device_ids)
        model.load_state_dict(torch.load("/proj/vondrick/aa4870/error_identification_checkpoints_200_512_focal_loss/experiment_0_30.pth")) 
        return model


        # /proj/vondrick/aa4870/error_identification_checkpoints_200_512_focal_loss/curriculum_2_20.pth


class ImageTextCorrection2(nn.Module):
    def __init__(self, embed_dim, llm, num_self_attention_layers=2, num_classes=1):
        super(ImageTextCorrection2, self).__init__()
        self.embed_dim = embed_dim 
        self.llm = llm
        self.num_classes = num_classes

        self.text_attention = nn.MultiheadAttention(embed_dim, num_heads=8) 
        self.text_attention.apply(init_weights)
        
        # for the image encoder
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.vit.head = torch.nn.Linear(self.vit.head.in_features, 14) 
        self.image_attention = nn.MultiheadAttention(self.vit.embed_dim, num_heads=8) 
        self.image_attention.apply(init_weights)
        
        pretrained_weights = torch.load("/proj/vondrick/aa4870/vit_checkpoints/vit_checkpoint_epoch_32.pth")
        
        fixed_weights = fix_state_dict_keys(pretrained_weights)
        self.vit.load_state_dict(fixed_weights)
        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.head = nn.Identity()
        self.vit_l1 = nn.Linear(self.vit.embed_dim, self.vit.embed_dim)
        # self.vit_ln1 = nn.LayerNorm(self.vit.embed_dim)  
        self.vit_l2 = nn.Linear(self.vit.embed_dim, self.embed_dim) 
        # self.vit_ln2 = nn.LayerNorm(self.embed_dim)  

        # for the text encoder
        self.text_llm = AutoModel.from_pretrained(llm)
        for param in self.text_llm.parameters():
            param.requires_grad = False

        self.bert_l1 = nn.Linear(self.embed_dim, self.embed_dim)
        # self.bert_ln1 = nn.LayerNorm(self.embed_dim)  
        self.bert_l2 = nn.Linear(self.embed_dim, self.embed_dim) 
        # self.bert_ln2 = nn.LayerNorm(self.embed_dim)  

        # For attention on fused embeddings
        self.self_attention_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim * 2, num_heads=4) for _ in range(num_self_attention_layers)])
        for layer in self.self_attention_layers:
            layer.apply(init_weights)

        # For final per token classification
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.classifier.apply(init_weights)

    def attention_pooling(self, outputs, attention_mask):

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

        # project attention output to the gatortron embedding space
        x = self.vit_l1(attn_output)
        x = F.relu(x)
        patch_embeds = self.vit_l2(x)

        h = attn_output.sum(dim=0)
        y = self.vit_l1(h) # [B, 768]
        y = F.relu(y)
        image_embed = self.vit_l2(y) # [B, embed_dim] ... for gatortron [B, 2560]

        return patch_embeds, image_embed

    def text_encoder(self, encoded_text, visualization=False):
        outputs = self.text_llm(**encoded_text)  # [B, N, 768] where N is number of tokens...
       
        if visualization: 
            # with torch.no_grad(): # No gradient flow at visualization
            sentence_embeddings = outputs['last_hidden_state'] # [B, N, embed_dim]
            x = self.bert_l1(sentence_embeddings) # [B, N, embed_dim]
            # x = self.bert_ln1(x)
            x = F.relu(x) # [B, N, embed_dim]
            x = self.bert_l2(x) # [B, N, embed_dim]
            # text_emb = self.bert_ln2(x)
            return x # [B, N, embed_dim]
        else:
            sentence_embeddings = self.attention_pooling(outputs, encoded_text["attention_mask"])  # [B, embed_dim] 
            x = self.bert_l1(sentence_embeddings)  # [B, embed_dim]   
            # x = self.bert_ln1(x)
            x = F.relu(x)  # [B, embed_dim]   
            x = self.bert_l2(x)  # [B, embed_dim] ... for gatortron [B, 2560]
            # text_emb = self.bert_ln2(x)

        return text_emb  # [B, embed_dim]   

    def forward(self, image, encoded_text):
        
        # Get image and token embeddings
        patch_embeds, image_embed = self.image_encoder(image) # [N, B, 2560], [B, 2560]
        zjs = self.text_encoder(encoded_text, visualization=True)  # [B, N, 2560]

        # Normalize embeddings
        patch_embeds = F.normalize(patch_embeds, p=2, dim=-1)
        image_embed = F.normalize(image_embed, p=2, dim=-1)
        zjs = F.normalize(zjs, p=2, dim=-1) 

        patch_embeds = patch_embeds.permute(1, 0, 2) # Now [2, 197 + 100, 2560]
        patch_zjs_embeds =  torch.cat([patch_embeds, zjs], dim=1) # Now [2, 197 + 200, 2560]
        image_zjs_embeds = torch.cat([image_embed, zjs], dim=1) # Now [2, 1 + 200, 2560]

        # Self-Attention on the concatenated embeddings
        for layer in self.self_attention_layers:
            # TODO: experiment 1, find out the effect of concatenating with patch embedding
            fused_embeddings, fused_attention_weights = layer(patch_zjs_embeds, patch_zjs_embeds, patch_zjs_embeds)
           
            # TODO: experiment 2, find out the effect of concatenating with the single image embedding
            # fused_emb, _ = layer(image_zjs_embeds, image_zjs_embeds, image_zjs_embeds)

        # Per token classification
        # TODO: for experiment 1
        output = self.classifier(fused_embeddings[:, -200:, :])

        # TODO: for experiment 2
        # output = self.classifier(fused_embed[:, -200, :])
        
        return output

    @classmethod
    def from_config(cls, cfg, device):
        model = instantiate(cfg.model).to(device)
        model = torch.nn.DataParallel(model, device_ids=cfg.main.device_ids)
        model.load_state_dict(torch.load("/proj/vondrick/aa4870/error_identification_checkpoints_200_512_focal_loss/experiment_1_30.pth")) 
        return model