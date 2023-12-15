import ast
import torch
import wandb
from tqdm import tqdm
import numpy as np

from ..predictions import get_top_k_sentences
from ..evaluation_utils import identify_errors, identify_correct_sentences

from sklearn.metrics import classification_report
import itertools

import torch.nn as nn

from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, GPT2Tokenizer
from models.text_correction import Correction
from models import ImageTextCorrection, ImageTextCorrection2

# Misc
import warnings
warnings.filterwarnings("ignore")
import re
import os
os.environ["TRANSFORMERS_CACHE"] = "/proj/vondrick/aa4870/freewilly"

# Metrics
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import Levenshtein

import openai

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()

def validate(model, dataloader, device, tokenizer, ntxent_loss, epoch, logger):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (_, xis, xls) in enumerate(tqdm(dataloader)):
            xis = xis.to(device)
            text_batch = xls
            text_encoded = tokenizer.batch_encode_plus(list(text_batch), padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            text_encoded = {key: value.to(device) for key, value in text_encoded.items()}

            zis, zjs = model(xis, text_encoded) 

            loss = ntxent_loss(zis, zjs) 
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)

    logger.info(f"Epoch {epoch}, Avg Loss: {avg_loss}")
    wandb.log({"val_loss": avg_loss})

    return avg_loss

def validate_negatives(cfg, model, dataloaders, epoch, device, tokenizer, classifier_ntxent_loss, logger):
    model.eval()
    losses = []
    pos = []
    neg = []
    correct_predictions = 0
    total_predictions = 0

    binary_criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for step, (images, reports, neg_reports) in enumerate(tqdm(dataloaders)):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Generate targets for positive and negative pairs
            pos_target = torch.ones(batch_size, dtype=torch.float32).to(device) 
            neg_target = torch.zeros(batch_size, dtype=torch.float32).to(device)

            encoded_reports = tokenizer.batch_encode_plus(reports, padding=True, truncation=True, max_length=512, return_tensors="pt")
            encoded_reports = {key: value.to(device) for key, value in encoded_reports.items()}
            encoded_neg_reports = tokenizer.batch_encode_plus(neg_reports, padding=True, truncation=True, max_length=512, return_tensors="pt")
            encoded_neg_reports = {key: value.to(device) for key, value in encoded_neg_reports.items()}

            # Positive pairs
            pos_output, pos_zis, pos_zjs = model(images, encoded_reports)
            pos_loss = classifier_ntxent_loss(pos_zis, pos_zjs) + binary_criterion(pos_output.squeeze(), pos_target)
            pos.append(pos_output.cpu())
            
            predictions = torch.sigmoid(pos_output) >= 0.5
            correct_predictions += (predictions == pos_target.float()).sum().item()
            total_predictions += predictions.numel()

            # Negative pairs
            neg_output, neg_zis, neg_zjs = model(images, encoded_neg_reports)
            neg_loss = binary_criterion(neg_output.squeeze(), neg_target)
            neg.append(neg_output.cpu())
            
            predictions = torch.sigmoid(neg_output.squeeze()) < 0.5
            correct_predictions += (predictions == neg_target.float()).sum().item()
            total_predictions += predictions.numel()

            loss = pos_loss + neg_loss
            losses.append(loss.item())

        accuracy = correct_predictions / total_predictions
        logger.info(f'====> Validation set loss: {np.mean(losses):.4f}')
        logger.info(f'====> Validation accuracy: {accuracy:.4f}')

    return np.mean(losses), torch.cat(pos).numpy(), torch.cat(neg).numpy(), accuracy

def compute_neg_weight(dataloader, n_batches=10):
    total_positives = 0
    total_negatives = 0

    for _, labels, _, _, _, _ in itertools.islice(dataloader, n_batches):
        total_negatives += (labels == 0).sum() 
        total_positives += (labels == 1).sum() 

    neg_weight = total_positives.float() / total_negatives.float()
    return neg_weight

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
error_detection_cfg = None

def validate_token_classifier(cfg, model, dataloaders, epoch, device, tokenizer):
    error_detection_cfg = cfg
    model.eval() 
    running_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []  # Store raw probabilities for ROC-AUC

    neg_weight = compute_neg_weight(dataloaders).to(device)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=neg_weight)
    criterion = FocalLoss()

    # threshold = 0.3
    # thresholds = [round(i * 0.1, 1) for i in range(5, 11)]
    thresholds = [round(i * 0.05, 2) for i in range(10, 21)]
    thresholds = [0.7]
    
    for threshold in thresholds: 
        print(threshold)
        with torch.no_grad():
            for batch_idx, (images, labels, neg_reports, reports, _, _) in enumerate(tqdm(dataloaders)):
                images = images.to(device)
                labels = labels.to(device)  
                batch_size = images.size(0)

                encoded_neg_reports = tokenizer.batch_encode_plus(neg_reports, padding="max_length", truncation=True, max_length=200, return_tensors="pt")
                encoded_neg_reports = {key: value.to(device) for key, value in encoded_neg_reports.items()}

                raw_output = model(images, encoded_neg_reports)
                output = torch.sigmoid(raw_output)

                # Check for any invalid outputs and labels
                if torch.any(output < 0) or torch.any(output > 1):
                    print("Invalid output detected!")
                    print(output)
                if torch.any((labels != 0) & (labels != 1)):
                    print("Invalid label detected!")
                    print(labels)

                loss = criterion(output.view(-1), labels.view(-1).float())

                running_loss += loss.item() * images.size(0)
            
                preds = (torch.sigmoid(output) > threshold).float()
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())
                all_probs.extend(torch.sigmoid(output).view(-1).cpu().numpy())  # Store probabilities

        epoch_loss = running_loss / len(dataloaders.dataset)

        print('Validation Loss: {:.4f}'.format(epoch_loss))
        print('Classification Report:')

        # error => 0, no error -=> 1
        report = classification_report(all_labels, all_preds, target_names=['error', 'no error'], output_dict=True)
        print(report)

        auc_score = roc_auc_score(all_labels, all_probs)
        print(f"ROC-AUC Score: {auc_score:.4f}")

        # Compute ROC values for visualization later
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

        conf_matrix = confusion_matrix(all_labels, all_preds)

        print('Confusion Matrix:')
        print(conf_matrix)

        from sklearn.metrics import precision_recall_fscore_support

        # Calculate per-class metrics
        precision_per_class, recall_per_class, fscore_per_class, support_per_class = precision_recall_fscore_support(all_labels, all_preds)
        # Calculate micro average metrics
        precision_micro, recall_micro, fscore_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
        # Calculate macro average metrics
        precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

        # Print per-class metrics
        print('Class "error" Precision: {:.4f}'.format(precision_per_class[0]))
        print('Class "error" Recall: {:.4f}'.format(recall_per_class[0]))
        print('Class "error" F1-Score: {:.4f}'.format(fscore_per_class[0]))
        print('Class "error" Support: {:.0f}'.format(support_per_class[0]))

        print('Class "no error" Precision: {:.4f}'.format(precision_per_class[1]))
        print('Class "no error" Recall: {:.4f}'.format(recall_per_class[1]))
        print('Class "no error" F1-Score: {:.4f}'.format(fscore_per_class[1]))
        print('Class "no error" Support: {:.0f}'.format(support_per_class[1]))

        # Print micro average metrics
        print('Micro Average Precision: {:.4f}'.format(precision_micro))
        print('Micro Average Recall: {:.4f}'.format(recall_micro))
        print('Micro Average F1-Score: {:.4f}'.format(fscore_micro))

        # Print macro average metrics
        print('Macro Average Precision: {:.4f}'.format(precision_macro))
        print('Macro Average Recall: {:.4f}'.format(recall_macro))
        print('Macro Average F1-Score: {:.4f}'.format(fscore_macro))


    return epoch_loss, report, (fpr, tpr, thresholds), conf_matrix








# Don't look at this code unless your intention is to do evaluation on the error correction module
# of the health-autocorrect model
#
#
#
#
#
#
#
#
#
#
#
#
import numpy as np
from scipy import stats

def compute_confidence_interval(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    ci = 1.96 * (std_dev / np.sqrt(len(data)))
    return mean, (mean-ci, mean+ci)

api_key = "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"
def refine_report_with_gpt4(original_report, text_format):
    openai.api_key = api_key 

    prompt = f"Refine the text to be more coherent and grammatically correct while retaining its original meaning: '{original_report}'"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=256
    )

    refined_text = response.choices[0].text.strip()

    second_prompt = f"Given this medical report '{refined_text}', write it following this template '{text_format}' replacing the words that are different."
    response2 = openai.Completion.create(
        engine="text-davinci-003",
        prompt=second_prompt,
        max_tokens=256
    )
    return response2.choices[0].text.strip()

def evaluate_error_correction(cfg, model, dataloaders, epoch, device, tokenizer, threshold=0.65):
    threshold = 1.0
    model.eval() 
    results = []

    all_bleu_scores, all_rouge_scores, all_edit_distances, all_meteor_scores = [], [], [], []
    rouge_1_f, rouge_2_f, rouge_l_f = [], [], []
    all_bleu1_scores, all_bleu2_scores, all_bleu3_scores, all_bleu4_scores = [], [], [], []

    with torch.no_grad():
        for batch_idx, (images, a, neg_reports, reports, _, v) in enumerate(tqdm(dataloaders)):
            images = images.to(device)
            batch_size = images.size(0)

            # Preprocessing reports
            encoded_neg_reports = tokenizer.batch_encode_plus(neg_reports, padding="max_length", truncation=True, max_length=200, return_tensors="pt")
            encoded_neg_reports = {key: value.to(device) for key, value in encoded_neg_reports.items()}

            # 1. Error identification part
            outputs = model(images, encoded_neg_reports)

            for i in range(batch_size):
                sigmoid_output = torch.sigmoid(outputs[i]) # this is equivalent to getting one example from the batch
                encoded_text = encoded_neg_reports['input_ids'][i]
                tokens = tokenizer.convert_ids_to_tokens(encoded_text.cpu().numpy())
                grouped_tokens, grouped_errors, grouped_sigmoid_vals = [], [], []
                temp_token, temp_error, temp_sigmoid = '', None, None
                
                # 1.1 Identify the erroneous and correct tokens
                for idx, (token, output) in enumerate(zip(tokens, sigmoid_output)):
                    if token.startswith("##"):
                        temp_token += token[2:]
                    else:
                        if temp_token:
                            grouped_tokens.append(temp_token)
                            grouped_errors.append(temp_error)
                            grouped_sigmoid_vals.append(temp_sigmoid)
                        temp_token = token
                        temp_error = 0 if output < float(threshold) else 1  
                        temp_sigmoid = output[0]
                
                grouped_tokens.append(temp_token)
                grouped_errors.append(temp_error)
                grouped_sigmoid_vals.append(temp_sigmoid)

                # 1.2 Put the tokens with errors in curly braces
                modified_text = ""  # Holds the text with errors wrapped in curly braces
                for token, error, sigmoid_val in zip(grouped_tokens, grouped_errors, grouped_sigmoid_vals):
                    if token == "[PAD]":
                        continue
                    if error == 0:  # If it's an error, wrap in curly braces
                        modified_text += f"{{{token}}} "
                    else:
                        modified_text += f"{token} "

                # print(modified_text)
                
                # 2. Error correction part
                correction_model = Correction.from_config(cfg, device)
                gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
                encoded_text = gpt_tokenizer(modified_text, padding="max_length", truncation=True, max_length=200, return_tensors="pt").to(device)

                # Using beam search for generating text
                # beam_outputs = correction_model.generate(
                #     encoded_text["input_ids"],
                #     max_length=200,            # Adjust as needed
                #     num_beams=5,               # Number of beams for beam search
                #     early_stopping=True,       # Stop when num_beams sentences are fully generated
                #     no_repeat_ngram_size=2     # To prevent the model from repeating the same n-grams
                # )

                # # Using nucleus sampling for generating text
                # nucleus_outputs = correction_model.generate(
                #     encoded_text["input_ids"],
                #     max_length=200,            # Adjust as needed
                #     do_sample=True,            # Enable sampling
                #     top_p=0.92,                # Adjust as needed for nucleus sampling
                #     top_k=0                    # Set to 0 for pure nucleus sampling
                # )

                # beam_predicted_text = gpt_tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
                # nucleus_predicted_text = gpt_tokenizer.decode(nucleus_outputs[0], skip_special_tokens=True)
                # beam_predicted_text = re.sub(r'[.,]+$', '', beam_predicted_text).strip()
                # nucleus_predicted_text = re.sub(r'[.,]+$', '', nucleus_predicted_text).strip()

                eval_predictions = correction_model(images[i].unsqueeze(0), encoded_text["input_ids"])
                eval_predicted_token_ids = torch.argmax(eval_predictions, dim=-1)
                predicted_text = gpt_tokenizer.decode(eval_predicted_token_ids[0], skip_special_tokens=True)
                predicted_text = re.sub(r'[.,]+$', '', predicted_text).strip()

                # print(predicted_text)
                
                # 3. Use the refine_report_with_gpt4() function to refine the output from the GPT model
                refined_text = refine_report_with_gpt4(predicted_text, neg_reports[i]) # only run this when you are very sure everything else is running because it could be very expensive

                print("---------------------")
                print("Autocorrected Text: ", refined_text)
                print("--")
                print("Groundtruth Text: ", reports[i])
                print("--")
                print("Original incorrect reports: ", neg_reports[i])
                print("---------------------")

                results.append({
                    "autocorrected_text": refined_text,
                    "groundtruth_text": reports[i]
                })

                if not predicted_text.strip():
                    print("Warning: Predicted text is empty. Skipping metrics computation for this instance.")
                    continue

                if not refined_text.strip():
                    print("Warning: Refined text is empty. Skipping metrics computation for this instance.")
                    continue
                
                # Compute metrics
                edit_distance, bleu1, bleu2, bleu3, bleu4, meteor, rouge_scores = compute_metrics(refined_text, reports[i])
                all_edit_distances.append(edit_distance)
                all_bleu1_scores.append(bleu1)
                all_bleu2_scores.append(bleu2)
                all_bleu3_scores.append(bleu3)
                all_bleu4_scores.append(bleu4)
                all_meteor_scores.append(meteor)
                all_rouge_scores.append(rouge_scores)
                rouge_1_f.append(rouge_scores['rouge-1']['f'])
                rouge_2_f.append(rouge_scores['rouge-2']['f'])
                rouge_l_f.append(rouge_scores['rouge-l']['f'])

                print(neg_reports[i], reports[i])

                if i == 20:
                    break

            break

    # Calculate average metrics
    avg_bleu1 = sum(all_bleu1_scores) / len(all_bleu1_scores)
    avg_bleu2 = sum(all_bleu2_scores) / len(all_bleu2_scores)
    avg_bleu3 = sum(all_bleu3_scores) / len(all_bleu3_scores)
    avg_bleu4 = sum(all_bleu4_scores) / len(all_bleu4_scores)
    avg_meteor = sum(all_meteor_scores) / len(all_meteor_scores)
    avg_edit_distance = sum(all_edit_distances) / len(all_edit_distances)
    avg_rouge_1_f = sum(rouge_1_f) / len(rouge_1_f)
    avg_rouge_2_f = sum(rouge_2_f) / len(rouge_2_f)
    avg_rouge_l_f = sum(rouge_l_f) / len(rouge_l_f)

    avg_bleu1, bleu1_ci = compute_confidence_interval(all_bleu1_scores)
    avg_bleu2, bleu2_ci = compute_confidence_interval(all_bleu2_scores)
    avg_bleu3, bleu3_ci = compute_confidence_interval(all_bleu3_scores)
    avg_bleu4, bleu4_ci = compute_confidence_interval(all_bleu4_scores)
    avg_meteor, meteor_ci = compute_confidence_interval(all_meteor_scores)
    avg_edit_distance, edit_distance_ci = compute_confidence_interval(all_edit_distances)
    avg_rouge_1_f, rouge_1_f_ci = compute_confidence_interval(rouge_1_f)
    avg_rouge_2_f, rouge_2_f_ci = compute_confidence_interval(rouge_2_f)
    avg_rouge_l_f, rouge_l_f_ci = compute_confidence_interval(rouge_l_f)

    print(f"BLEU-1: {avg_bleu1} CI: {bleu1_ci}")
    print(f"BLEU-2: {avg_bleu2} CI: {bleu2_ci}")
    print(f"BLEU-3: {avg_bleu3} CI: {bleu3_ci}")
    print(f"BLEU-4: {avg_bleu4} CI: {bleu4_ci}")
    print(f"METEOR: {avg_meteor} CI: {meteor_ci}")
    print(f"Edit Distance: {avg_edit_distance} CI: {edit_distance_ci}")
    print(f"ROUGE-1 F1: {avg_rouge_1_f} CI: {rouge_1_f_ci}")
    print(f"ROUGE-2 F1: {avg_rouge_2_f} CI: {rouge_2_f_ci}")
    print(f"ROUGE-L F1: {avg_rouge_l_f} CI: {rouge_l_f_ci}")

    # import pandas as pd
    # results_df = pd.DataFrame(results)

    # # Save the DataFrame to a CSV file
    # results_df.to_csv('/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/results/autocorrected_reports_exp_0.csv', index=False)


    # return avg_bleu, avg_rouge, avg_edit_distance

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk import word_tokenize

import nltk

nltk.download('wordnet')
nltk.download('punkt')

def compute_metrics(predicted_text, ground_truth):
    # Tokenize sentences for METEOR score computation
    tokenizer = nltk.word_tokenize
    ground_truth_tokens = tokenizer(ground_truth)
    predicted_text_tokens = tokenizer(predicted_text)
    
    # Compute edit distance
    edit_distance = np.sum([1 for i, j in zip(predicted_text, ground_truth) if i != j]) + abs(len(predicted_text) - len(ground_truth))
    
    # Compute BLEU score
    reference = [ground_truth.split()]
    candidate = predicted_text.split()
    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Compute METEOR score
    meteor = meteor_score([ground_truth_tokens], predicted_text_tokens)
    
    # Compute ROUGE scores
    rouge = Rouge()
    scores = rouge.get_scores(predicted_text, ground_truth)
    rouge_scores = scores[0]  # Get the scores for the first (and only) pair
    
    return edit_distance, bleu1, bleu2, bleu3, bleu4, meteor, rouge_scores





# Other experiments
#
#
#
#
#
#
#
#
#
# This section for the validation of the retrieval based report generation model

from models import ImageTextGroundingModelHierarchical
from PIL import Image
from torchvision.transforms import transforms

from utils.predictions.prediction_utils import encode_text_and_image

import faiss
import sqlite3

import warnings
warnings.filterwarnings("ignore")

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return preprocess(image).unsqueeze(0) # add extra dimension to simulate a batch with only one example

# get all the image paths and the embeddings of their corresponding textual descriptions
def get_data(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT * FROM report_data_1")
    data = c.fetchall()
    
    # Separate the returned data into individual lists
    ids, image_paths, texts, embeddings = zip(*data)
    
    # Convert bytes back to numpy arrays for the embeddings
    embeddings_np = [np.frombuffer(embedding, dtype=np.float32).reshape(-1, 1024) for embedding in embeddings]
    embeddings_np = [embedding / np.linalg.norm(embedding) for embedding in embeddings_np]
    return ids, image_paths, texts, embeddings_np

# Here is were get the image embedding and then do a similarity score on the whole dataset to retrieve 
# the most similar report
def retrieve_best_reports(model, tokenizer, image_path, faiss_index, db_path): 
    image_tensor = preprocess_image(image_path)
    # assuming your model has a method to get the image embedding
    _, _, image_embedding = encode_text_and_image("", image_tensor, model, tokenizer)
    image_embedding = image_embedding.cpu().numpy()
    image_embedding = image_embedding / np.linalg.norm(image_embedding)

    # Search the FAISS index for the top 1 most similar report embeddings
    D, I = faiss_index.search(image_embedding.reshape(1, -1), 1)  # Adjust the second parameter to retrieve more or fewer reports
   
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    top_reports = []

    for idx in I[0]:
        c.execute(f"SELECT text FROM report_data_1 WHERE id={idx}")
        fetch_result = c.fetchone()[0]
        if fetch_result is None:  # Check if fetch_result is None
            print(f"No row found for idx: {idx}")
        elif fetch_result not in top_reports:
            top_reports.append(fetch_result)

    conn.close()
    
    return top_reports


import sqlite3
import numpy as np

def retrieve_random_reports(db_path, num_reports=1):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM report_data_1")
    total_reports = c.fetchone()[0]

    random_indices = np.random.choice(range(total_reports), size=num_reports, replace=False)

    random_reports = []

    for idx in random_indices:
        c.execute(f"SELECT text FROM report_data_1 WHERE id={idx}")
        fetch_result = c.fetchone()
        if fetch_result is not None:
            random_reports.append(fetch_result[0])
        else:
            print(f"No row found for idx: {idx}")

    conn.close()
    
    return random_reports


def evaluate_retrieval_model(model, dataloaders, tokenizer, device, cfg):
    all_bleu_scores, all_rouge_scores, all_edit_distances, all_meteor_scores = [], [], [], []
    rouge_1_f, rouge_2_f, rouge_l_f = [], [], []
    all_bleu1_scores, all_bleu2_scores, all_bleu3_scores, all_bleu4_scores = [], [], [], []

    import faiss
    db_path = '/proj/vondrick/aa4870/embeddings.db'

    def create_faiss_index():
        _, _, _, embeddings_np = get_data(db_path)  
        embeddings_matrix = np.vstack(embeddings_np)  
        index = faiss.IndexFlatIP(embeddings_matrix.shape[1])  # Create FAISS index for inner product
        index.add(embeddings_matrix)  # Add normalized embeddings to the index
        return index

    faiss_index = create_faiss_index()

    # do the evaluation on the whole validation set
    with torch.no_grad():
        for batch_idx, (image_path, images, reports) in enumerate(tqdm(dataloaders)):
            images = images.to(device)
            images = images.to(device)
            batch_size = images.size(0)

            # for each data row
            for i in range(batch_size):
                top_report = retrieve_best_reports(model, tokenizer, image_path[i], faiss_index, db_path)
                print(top_report)
                # top_report = retrieve_random_reports(db_path)
                
                # print(top_report)

                # BOOM! apply autocorrect here!!!! 
                # TODO: This part will of course be very expensive and time consuming to do (need to add more money)
                # autocorrect_report = autocorrect_top_report(correction_model, image_path[i], top_report[0], cfg)
                # print(autocorrect_report)

                edit_distance, bleu1, bleu2, bleu3, bleu4, meteor, rouge_scores = compute_metrics(top_report[0], reports[i])

                # TODO: compute this to see if there is an improvement in the scores you are computing
                # edit_distance, bleu1, bleu2, bleu3, bleu4, meteor, rouge_scores = compute_metrics(autocorrect_report, reports[i])

                all_edit_distances.append(edit_distance)
                all_bleu1_scores.append(bleu1)
                all_bleu2_scores.append(bleu2)
                all_bleu3_scores.append(bleu3)
                all_bleu4_scores.append(bleu4)
                all_meteor_scores.append(meteor)
                all_rouge_scores.append(rouge_scores)
                rouge_1_f.append(rouge_scores['rouge-1']['f'])
                rouge_2_f.append(rouge_scores['rouge-2']['f'])
                rouge_l_f.append(rouge_scores['rouge-l']['f'])

                


    # Calculate average metrics
    avg_bleu1 = sum(all_bleu1_scores) / len(all_bleu1_scores)
    avg_bleu2 = sum(all_bleu2_scores) / len(all_bleu2_scores)
    avg_bleu3 = sum(all_bleu3_scores) / len(all_bleu3_scores)
    avg_bleu4 = sum(all_bleu4_scores) / len(all_bleu4_scores)
    avg_meteor = sum(all_meteor_scores) / len(all_meteor_scores)
    avg_edit_distance = sum(all_edit_distances) / len(all_edit_distances)
    avg_rouge_1_f = sum(rouge_1_f) / len(rouge_1_f)
    avg_rouge_2_f = sum(rouge_2_f) / len(rouge_2_f)
    avg_rouge_l_f = sum(rouge_l_f) / len(rouge_l_f)

    avg_bleu1, bleu1_ci = compute_confidence_interval(all_bleu1_scores)
    avg_bleu2, bleu2_ci = compute_confidence_interval(all_bleu2_scores)
    avg_bleu3, bleu3_ci = compute_confidence_interval(all_bleu3_scores)
    avg_bleu4, bleu4_ci = compute_confidence_interval(all_bleu4_scores)
    avg_meteor, meteor_ci = compute_confidence_interval(all_meteor_scores)
    avg_edit_distance, edit_distance_ci = compute_confidence_interval(all_edit_distances)
    avg_rouge_1_f, rouge_1_f_ci = compute_confidence_interval(rouge_1_f)
    avg_rouge_2_f, rouge_2_f_ci = compute_confidence_interval(rouge_2_f)
    avg_rouge_l_f, rouge_l_f_ci = compute_confidence_interval(rouge_l_f)

    print(f"BLEU-1: {avg_bleu1} CI: {bleu1_ci}")
    print(f"BLEU-2: {avg_bleu2} CI: {bleu2_ci}")
    print(f"BLEU-3: {avg_bleu3} CI: {bleu3_ci}")
    print(f"BLEU-4: {avg_bleu4} CI: {bleu4_ci}")
    print(f"METEOR: {avg_meteor} CI: {meteor_ci}")
    print(f"Edit Distance: {avg_edit_distance} CI: {edit_distance_ci}")
    print(f"ROUGE-1 F1: {avg_rouge_1_f} CI: {rouge_1_f_ci}")
    print(f"ROUGE-2 F1: {avg_rouge_2_f} CI: {rouge_2_f_ci}")
    print(f"ROUGE-L F1: {avg_rouge_l_f} CI: {rouge_l_f_ci}")


def autocorrect_top_report(model, image_path, top_report, error_detection_cfg, device="cuda"):
    threshold = 0.65

    # 1. Error Identification part
    tokenizer = AutoTokenizer.from_pretrained(error_detection_cfg.main.biomegatron)
    # model = ImageTextCorrection.from_config(error_detection_cfg, device)
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    encoded_text = tokenizer(top_report, padding="max_length", truncation=True, max_length=200, return_tensors="pt").to(device)
    
    output = model(image_tensor, encoded_text)
    m = torch.nn.Sigmoid()
    sigmoid_outputs = m(output[0])

    sigmoid_outputs = sigmoid_outputs.cpu().detach().numpy()
    input_ids = encoded_text['input_ids'].cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    grouped_tokens = []
    grouped_errors = []
    grouped_sigmoid_vals = [] 
    temp_token = ''

    print(tokens)
    print(sigmoid_outputs)

    # 1.1 Identify the erroneous and correct tokens
    for idx, (token, output) in enumerate(zip(tokens, sigmoid_outputs[0])):
        if token.startswith("##"):
            temp_token += token[2:]  
        else:
            if temp_token:
                grouped_tokens.append(temp_token)
                grouped_errors.append(temp_error)
                grouped_sigmoid_vals.append(temp_sigmoid)  

            temp_token = token
            temp_error = 0 if output < float(threshold) else 1  
            temp_sigmoid = output  

    grouped_tokens.append(temp_token)
    grouped_errors.append(temp_error)
    grouped_sigmoid_vals.append(temp_sigmoid)
    modified_text = "" # holds the text with errors wrapped in curly braces

    # 1.2 Color code here for the gradio interface and include curly braces for text formating
    for token, error, sigmoid_val in zip(grouped_tokens, grouped_errors, grouped_sigmoid_vals):
        if token == "[PAD]":
            continue
        if error == 0:  # if it's an error, let's make it red
            modified_text += f"{{{token}}} "
        else:
            modified_text += f"{token} "

    print("Modified text: ", modified_text)

    # 2. Error correction part
    correction_model = Correction.from_config(error_detection_cfg, device)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token 
    encoded_text = gpt_tokenizer(modified_text, padding="max_length", truncation=True, max_length=200, return_tensors="pt").to(device)
    eval_predictions = correction_model(image_tensor, encoded_text["input_ids"])
    eval_predicted_token_ids = torch.argmax(eval_predictions, dim=-1)
    eval_predicted_token_ids = eval_predicted_token_ids[0]
    predicted_text = gpt_tokenizer.decode(eval_predicted_token_ids, skip_special_tokens=True)
    predicted_text = re.sub(r'[.,]+$', '', predicted_text).strip()

    print("Predicted text: ", predicted_text)

    # 3. Use the refine_report() function to refine the output from the GPT model
    refined_text = refine_report_with_gpt4(predicted_text)

    return refined_text