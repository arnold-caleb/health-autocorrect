import csv
import random

# this is here because when we evaluate, we evaluate on the image paths, descriptions with errors 
# while having access to the error location in the descriptions/paragraphs. 
# otherwise, we would not have get_data here...
def get_eval_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append([row['image_path'], row['error_desc'], row['error_loc']])
    assert all(len(d) == 3 for d in data), "Each data entry should have exactly three elements."
    
    return data[1:]

def random_selection(num_sentences, selection_count):
    return random.sample(range(num_sentences), selection_count)

def identify_correct_sentences(text, errors, top_k_sentences, top_k_values, top_k_indices, threshold):
    assert len(top_k_sentences) == len(top_k_values) == len(top_k_indices), "Input lists should have the same length."
    
    correct_sentence_indices = set(range(len(top_k_sentences))) - set(errors)
    
    if not correct_sentence_indices: 
        return None

    # accuracy with model choosing correct sentences
    correct_sentence_indices_below_threshold = [index for value, index in zip(top_k_values, top_k_indices) if value >= threshold]

    # restrict potential errors to the number of known errors
    # correct_sentence_indices_below_threshold = correct_sentence_indices_below_threshold[:len(errors)]

    correctly_identified = correct_sentence_indices.intersection(correct_sentence_indices_below_threshold)
    model_accuracy = len(correctly_identified) / len(correct_sentence_indices)
    
    # accuracy if we randomly choose the correct sentences (lower bound for correct sentences)
    random_indices = random_selection(len(top_k_sentences), len(correct_sentence_indices))
    correctly_identified_random = correct_sentence_indices.intersection(random_indices)
    random_accuracy = len(correctly_identified_random) / len(correct_sentence_indices)

    return model_accuracy, random_accuracy

def identify_errors(text, errors, top_k_sentences, top_k_values, top_k_indices, threshold):
    assert len(top_k_sentences) == len(top_k_values) == len(top_k_indices), "Input lists should have the same length."
    
    if not errors:
        return None
        
    if len(errors) > len(top_k_sentences):
        return None

    # accuracy with model identifying the sentences with errors
    error_indices_below_threshold = [index for value, index in zip(top_k_values, top_k_indices) if value < threshold]

    # restrict potential errors to the number of known errors
    # error_indices_below_threshold = error_indices_below_threshold[:len(errors)]

    correctly_identified = set(errors).intersection(error_indices_below_threshold)
    model_accuracy = len(correctly_identified) / len(errors)
    
    # accuracy if we randomly choose the sentences with errors (lower bound for error identification)
    random_indices = random_selection(len(top_k_sentences), len(errors))
    correctly_identified_random = set(errors).intersection(random_indices)
    random_accuracy = len(correctly_identified_random) / len(errors)

    return model_accuracy, random_accuracy