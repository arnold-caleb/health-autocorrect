import torch
from transformers import AutoTokenizer
from .prediction_utils import calculate_similarities, combine_subwords, encode_text_and_image

def get_top_k_tokens(image_tensor, text, model, tokenizer, device="cuda:0"):
    encoded_text, token_embeddings, image_embeddings = encode_text_and_image(text, image_tensor, model, tokenizer)
    top_k_values, top_k_indices = calculate_similarities(token_embeddings, image_embeddings)
    
    # Unnecessary tokens
    unneccessary_tokens = ['and', 'provided', ':', 'these', 'e', ',', '.', 'the', 'of', 'or', 'in', 'for', 'findings', 'there', 'at', 'is', 'a', 'are', 'as', '[PAD]', '[SEP]', '[CLS]']

    # Get the original order of the tokens
    original_order_indices = sorted(range(len(top_k_indices[0])), key=lambda x: top_k_indices[0][x])
    top_k_values = [top_k_values[0][i] for i in original_order_indices]

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


def get_top_k_sentences(image_tensor, text, model, tokenizer, device="cuda:0"):
    # Split the text into sentences
    k = 10 
    sentences = text.split('. ')
    encoded_sentences = [tokenizer.encode_plus(sentence, padding='longest', truncation=True, max_length=512, return_tensors="pt").to(device) for sentence in sentences]

    with torch.no_grad():
        sentence_embeddings = []
        for encoded_sentence in encoded_sentences:
            word_embeddings = model.module.text_encoder(encoded_sentence, visualization=True) # [B, l, 128]
            # Calculate the average of the word embeddings
            sentence_embedding = torch.mean(word_embeddings, dim=1) # [B, 128]
            sentence_embeddings.append(sentence_embedding)
        
        sentence_embeddings = torch.stack(sentence_embeddings) # [N, B, 128]
        
        image_embeddings = model.module.image_encoder(image_tensor.to("cuda:0")) # [B, 128]
        
    sentence_embeddings = sentence_embeddings.permute(1, 0, 2) # [B, N, 128]

    top_k_values, top_k_indices = calculate_similarities(sentence_embeddings, image_embeddings)

    # Get the sentences in the order of their similarity
    final_sentences = [sentences[idx] for idx in top_k_indices[0]]

    # Sort sentences by their corresponding probabilities
    # sorted_sentences_probs = sorted(zip(final_sentences, top_k_values[0]), key=lambda x: x[1], reverse=True)

    # Unzip sentences and probabilities
    # sorted_sentences, sorted_probs = zip(*sorted_sentences_probs)

    return text, final_sentences[:k], top_k_values[0][:k], top_k_indices[0][:k]
