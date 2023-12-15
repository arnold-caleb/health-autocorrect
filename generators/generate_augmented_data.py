import csv 
import random
import difflib
import re

from tqdm import tqdm 

import openai

def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)  # replaces one or more spaces with a single space
    text = re.sub(r'\n', ' ', text)  # replaces "returns" or newline characters with a single space
    return text.strip()  # removes leading and trailing spaces

def get_error_indices(original_sentences, error_sentences):
    matcher = difflib.SequenceMatcher()
    error_indices = []
    for i in range(min(len(original_sentences), len(error_sentences))):
        matcher.set_seqs(normalize_whitespace(original_sentences[i]), normalize_whitespace(error_sentences[i]))
        if matcher.ratio() < 0.92: # adjust this threshold
            error_indices.append(i)
    return error_indices

def suitable_prompt(text):
    prompts = {"error_generation": f"Given the medical findings: '{text}', rewrite the medical report including some errors that change the meaning of the report, they can fall in any of these categories: commission, perceptual, cognitive, technical, communication, recognition, overcall, and contextual errors. (Don't write the error name)",
               "data_augmentation": f"Given the medical findings: '{text}', rewrite it differently as an experienced radiologist would, ensuring the language is concise, precise and maintains all necessary information including the findings and the impression."}
    
    return prompts["data_augmentation"]

def alternative_desc(text, num_alternatives=5):
    alternatives = []
    openai.api_key = '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'  

    for _ in range(num_alternatives):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=suitable_prompt(text),
            temperature=0.8,
            max_tokens=100
        )

        error_text = response.choices[0].text.strip()
        error_text = normalize_whitespace(error_text)  # clean the error text

        # Normalize whitespaces before splitting into sentences
        original_sentences = normalize_whitespace(text).split('. ')
        error_sentences = error_text.split('. ')
        error_indices = get_error_indices(original_sentences, error_sentences)

        alternatives.append((error_text, error_indices))

    return alternatives

def make_finetune_csv():
    EXAMPLES = []

    with open('/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/data/train_csv_script.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: 
            EXAMPLES.append([row['row_number'], row['image'], row['text'],])

    selected_rows = random.sample(EXAMPLES[20000:], 6395 - 1438)

    with open('/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/data/finetune_set_3.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_number', 'image', 'text'])

        row_number = 25216

        for row in tqdm(selected_rows):
            _, image, description = row
            writer.writerow([row_number, image, description])
            alternative_results = alternative_desc(description) # an array of five alternative descriptions for description

            for alternative_result in alternative_results:
                row_number += 1
                alternative, _ = alternative_result 
                writer.writerow([row_number, image, alternative])
                
            # row_number += 1

make_finetune_csv()