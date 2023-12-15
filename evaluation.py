import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from utils import get_eval_data, identify_correct_sentences, identify_errors, get_top_k_sentences, preprocess_image

from models import ImageTextGroundingModelHierarchical

import warnings
warnings.filterwarnings("ignore")

import logging

def setup_eval_logger(name):
    logger = logging.getLogger(name)

    # Only add handlers if none are present
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
        
    return logger

def compute_accuracy(data, model, tokenizer, threshold):
    total_accuracy_errors = 0
    total_accuracy_correct_sentences = 0
    total_random_accuracy_errors = 0
    total_random_accuracy_correct_sentences = 0
    skipped_records_errors = 0
    skipped_records_correct_sentences = 0

    for i in tqdm(range(len(data)), desc="Computing Accuracy: "):
        image_tensor = preprocess_image(data[i][0])
        _, top_k_sentences, top_k_values, top_k_indices = get_top_k_sentences(image_tensor, data[i][1], model, tokenizer)
        errors = ast.literal_eval(data[i][2])

        result_errors = identify_errors(data[i][1], errors, top_k_sentences, top_k_values, top_k_indices, threshold)

        if result_errors is not None:
            accuracy_errors, random_accuracy_errors = result_errors
            total_accuracy_errors += accuracy_errors
            total_random_accuracy_errors += random_accuracy_errors
        else:
            skipped_records_errors += 1

        result_correct_sentences = identify_correct_sentences(data[i][1], errors, top_k_sentences, top_k_values, top_k_indices, threshold)
        
        if result_correct_sentences is not None:
            accuracy_correct_sentences, random_accuracy_correct_sentences = result_correct_sentences
            total_accuracy_correct_sentences += accuracy_correct_sentences
            total_random_accuracy_correct_sentences += random_accuracy_correct_sentences
        else:
            skipped_records_correct_sentences += 1

    avg_accuracy_errors = total_accuracy_errors / (len(data) - skipped_records_errors)
    avg_random_accuracy_errors = total_random_accuracy_errors / (len(data) - skipped_records_errors)

    avg_accuracy_correct_sentences = total_accuracy_correct_sentences / (len(data) - skipped_records_correct_sentences)
    avg_random_accuracy_correct_sentences = total_random_accuracy_correct_sentences / (len(data) - skipped_records_correct_sentences)

    combined_model_accuracy = 0.5 * avg_accuracy_errors + 0.5 * avg_accuracy_correct_sentences
    combined_random_accuracy = 0.5 * avg_random_accuracy_errors + 0.5 * avg_random_accuracy_correct_sentences

    return avg_accuracy_errors, avg_random_accuracy_errors, avg_accuracy_correct_sentences, avg_random_accuracy_correct_sentences, combined_model_accuracy, combined_random_accuracy


@hydra.main(config_path="./configs/models", config_name="grounding")
def main(cfg: DictConfig) -> None:
    logger = setup_eval_logger('Evaluation')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.main.biomegatron)

    # model = initialize_model(cfg, device)
    model = ImageTextGroundingModelHierarchical.from_config(cfg, device)

    data = get_eval_data(cfg.main.eval_data)
    assert len(data) > 0, "Data list should not be empty."

    threshold = 0.02  # set threshold here (cosine similarity ranges from -1 to 1)

    (
        avg_accuracy_errors,
        avg_random_accuracy_errors,
        avg_accuracy_correct_sentences,
        avg_random_accuracy_correct_sentences,
        combined_model_accuracy,
        combined_random_accuracy,
    ) = compute_accuracy(data, model, tokenizer, threshold)

    print(f"Average Error Accuracy:                    {avg_accuracy_errors * 100}%")
    print(f"Random Guessing Error Accuracy:            {avg_random_accuracy_errors * 100}%")
    print(f"Average Correct Sentence Accuracy:         {avg_accuracy_correct_sentences * 100}%")
    print(f"Random Guessing Correct Sentence Accuracy: {avg_random_accuracy_correct_sentences * 100}%")
    print(f"Combined Model Accuracy:                   {combined_model_accuracy * 100}%")
    print(f"Combined Random Guessing Accuracy:         {combined_random_accuracy * 100}%")

if __name__ == "__main__":
    main()