import torch 
import torch.nn.functional as F

def cosine_similarity(image_embeddings, token_embeddings):
    image_embeddings_normalized = F.normalize(image_embeddings, p=2, dim=1) # [B, 128]
    token_embeddings_normalized = F.normalize(token_embeddings, p=2, dim=2) # [B, l, 128]

    return (image_embeddings_normalized.unsqueeze(1) * token_embeddings_normalized).sum(-1) # [B, l]

def euclidean_similarity(image_embeddings, token_embeddings):
    image_embeddings_normalized = F.normalize(image_embeddings, p=2, dim=1)  # [B, 128]
    token_embeddings_normalized = F.normalize(token_embeddings, p=2, dim=2)  # [B, l, 128]

    distances = torch.cdist(image_embeddings_normalized.unsqueeze(1), token_embeddings_normalized).squeeze(1) # [B, l]
    max_distance = distances.max() # scalar

    return max_distance - distances # [B, l]

def manhattan_similarity(image_embeddings, token_embeddings):
    image_embeddings_normalized = F.normalize(image_embeddings, p=1, dim=1)  # [B, 128]
    token_embeddings_normalized = F.normalize(token_embeddings, p=1, dim=2)  # [B, l, 128]

    manhattan_distances = torch.cdist(image_embeddings_normalized.unsqueeze(1), token_embeddings_normalized, p=1).squeeze(1) # [B, l]
    max_distance = manhattan_distances.max() # scalar

    return max_distance - manhattan_distances # [B, l]

def chebyshev_similarity(image_embeddings, token_embeddings):
    image_embeddings_normalized = F.normalize(image_embeddings, p=float('inf'), dim=1)  # [B, 128]
    token_embeddings_normalized = F.normalize(token_embeddings, p=float('inf'), dim=2)  # [B, l, 128]

    chebyshev_distances = torch.cdist(image_embeddings_normalized.unsqueeze(1), token_embeddings_normalized, p=float('inf')).squeeze(1) # [B, l]
    max_distance = chebyshev_distances.max() # scalar

    return max_distance - chebyshev_distances # [B, l]

def minkowski_similarity(image_embeddings, token_embeddings, p=3):
    image_embeddings_normalized = F.normalize(image_embeddings, p=p, dim=1)  # [B, 128]
    token_embeddings_normalized = F.normalize(token_embeddings, p=p, dim=2)  # [B, l, 128]

    minkowski_distances = torch.cdist(image_embeddings_normalized.unsqueeze(1), token_embeddings_normalized, p=p).squeeze(1)  # [B, l]
    max_distance = minkowski_distances.max() # scalar
    
    return max_distance - minkowski_distances # [B, l]