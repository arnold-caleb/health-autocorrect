import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import GPT2LMHeadModel
# from utils import fix_state_dict_keys

def fix_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == 'n_averaged':
            continue  # Skip the unwanted key
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

class Correction(nn.Module):
    def __init__(self, embed_dim):
        super(Correction, self).__init__()
        self.llm = "gpt2-medium" # healx/gpt-2-pubmed-medium
        self.num_self_attention_layers = 2
        self.vocab_size = 30522
        self.embed_dim = embed_dim 
        
        # Vision Transformer Encoder
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.vit.head = torch.nn.Linear(self.vit.head.in_features, 14) 
        self.image_attention = nn.MultiheadAttention(self.vit.embed_dim, num_heads=8) 
        
        pretrained_weights = torch.load("/proj/vondrick/aa4870/vit_checkpoints/vit_checkpoint_epoch_32.pth")
        
        fixed_weights = fix_state_dict_keys(pretrained_weights)
        self.vit.load_state_dict(fixed_weights)

        self.vit.head = nn.Identity()
        self.vit_l1 = nn.Linear(self.vit.embed_dim, self.vit.embed_dim)
        self.vit_l2 = nn.Linear(self.vit.embed_dim, self.embed_dim) 

        # GPT2 Text Encoder
        self.text_model = GPT2LMHeadModel.from_pretrained(self.llm)
        self.text_model.resize_token_embeddings(51000)
        self.text_attention = nn.MultiheadAttention(self.embed_dim, num_heads=8) 

        # Projection layer
        self.projection = nn.Linear(self.embed_dim, self.text_model.config.n_embd) 

    def image_encoder(self, image):
        feature_maps = self.vit.forward_features(image) 
        
        feature_maps = feature_maps.permute(1, 0, 2) 
        attn_output, attn_weights = self.image_attention(feature_maps, feature_maps, feature_maps)  
        # h = attn_output.sum(dim=0) 
        
        x = self.vit_l1(attn_output)
        x = F.relu(x)
        x = self.vit_l2(x) 

        return x, attn_weights

    def text_encoder(self, encoded_text):
        outputs = self.text_model.transformer(encoded_text, output_attentions=True)
        return outputs.last_hidden_state, outputs.attentions

    def forward(self, image, encoded_text):

        zis, attn_weights_image = self.image_encoder(image) 
        zjs, attn_weights_text = self.text_encoder(encoded_text) 

        # print("Patch embeddings shape: ", zis.shape) # [197, 2, 1024]
        # print("Encoded_text shape: ", zjs.shape) # [2, 100, 1024]

        zis = zis.permute(1, 0, 2)  # Now [2, 197, 1024]

        concatenated_embeddings = torch.cat([zjs, zis], dim=1)  # Now [2, 197 + 100, 1024]

        # print("concatenated embeddings: ")
        # print(concatenated_embeddings.shape)
        # projected_embeddings = self.projection(concatenated_embeddings)  # Now [2, 297, GPT_dim]
        
        # print("projected embeddings: ")
        # print(projected_embeddings.shape)
        predictions = self.text_model(inputs_embeds=concatenated_embeddings).logits

        print("Image attention weights shape:", attn_weights_image.shape)
        # Convert the attention weights to numpy array and then to string

        # Save the string representation of attention weights to a text file
        torch.save(attn_weights_image, '/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/attn_weights_image.pt')
        torch.save(attn_weights_text, '/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/attn_weights_text.pt')
        
        # print("Text attention weights shape:", [attn.shape for attn in attn_weights_text])
        # print(predictions.shape)

        return predictions
        
    @classmethod
    def from_config(cls, cfg, device):
        model = cls(cfg.model.embed_dim).to(device)
        model = torch.nn.DataParallel(model, device_ids=cfg.main.device_ids)
        model.load_state_dict(torch.load("/proj/vondrick/aa4870/correction_checkpoints/correction_model_epoch_40.pth"), strict=False)
        return model
