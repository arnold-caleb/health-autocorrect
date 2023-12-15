import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
# from similarity_metrics import manhattan_similarity, cosine_similarity

def cosine_similarity(image_embeddings, token_embeddings):
    image_embeddings_normalized = F.normalize(image_embeddings, p=2, dim=1) # [B, 128]
    token_embeddings_normalized = F.normalize(token_embeddings, p=2, dim=2) # [B, l, 128]

    return (image_embeddings_normalized.unsqueeze(1) * token_embeddings_normalized).sum(-1) # [B, l]


def encode_text_and_image(text, image_tensor, model, tokenizer):

    image_tensor = image_tensor.to("cuda:0")
    encoded_text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        token_embeddings = model.module.text_encoder(encoded_text, visualization=True) # [B, l, 128]
        image_embeddings = model.module.image_encoder(image_tensor) # [B, 128]

    return encoded_text, token_embeddings, image_embeddings


def calculate_similarities(token_embeddings, image_embeddings):
    k = token_embeddings.shape[1] # return all the tokens for now

    similarities = cosine_similarity(image_embeddings, token_embeddings)
    # similarities = manhattan_similarity(image_embeddings, token_embeddings) # seems to perform better than the others

    top_k_values, top_k_indices = torch.topk(similarities, k, dim=1) 
    # top_k_values = F.softmax(top_k_values, dim=1)
    return top_k_values.tolist(), top_k_indices.tolist() # similarity scores of the top_k_values and their locations


def combine_subwords(encoded_text, top_k_values, tokenizer):
    combined_tokens, combined_indices, combined_probs = [], [], []
    subword_probs = []
    
    for i in range(len(encoded_text['input_ids'][0])):
        token = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0][i].item())
        
        if token.startswith("##"):
            combined_tokens[-1] += token[2:]
            subword_probs.append(top_k_values[i])
            combined_probs[-1] = sum(subword_probs) / len(subword_probs)  # Average probabilities of subwords, we could not ge the probabilities of the sentences
        else:
            combined_tokens.append(token)
            combined_indices.append(i)  # Save the original index of each combined token
            subword_probs = [top_k_values[i]]
            combined_probs.append(subword_probs[0])

    return combined_tokens, combined_indices, combined_probs
