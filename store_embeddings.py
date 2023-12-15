import torch 
import numpy as np
from transformers import AutoTokenizer
import torch.nn.functional as F

import random
import pandas as pd

import gradio as gr
from gradio import components as gc

from models import ImageTextGroundingModelHierarchical
from utils import preprocess_image, get_top_k_tokens, get_top_k_sentences

from utils.predictions.prediction_utils import encode_text_and_image

import argparse

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sqlite3

import warnings
warnings.filterwarnings("ignore")

conn = sqlite3.connect('/proj/vondrick/aa4870/embeddings.db')
c = conn.cursor()

# Create a table if it doesn't exist
c.execute('''
          CREATE TABLE IF NOT EXISTS report_data_1
          (id INTEGER PRIMARY KEY,
          image_path TEXT,
          text TEXT,
          embedding BLOB)
          ''')
conn.commit()

def get_first_entry():
    # Query to select the first entry from the table
    c.execute("SELECT * FROM report_data_1 ORDER BY id ASC LIMIT 1")
    data = c.fetchone()  # fetchone() retrieves the first row of a SELECT statement
    
    if data is not None:
        id, image_path, text, embedding_blob = data
        # Convert bytes back to numpy array for the embedding
        embedding_np = np.frombuffer(embedding_blob, dtype=np.float32).reshape(-1, 1024)
        return id, image_path, text, embedding_np
    else:
        print("No data found")
        return None


def get_data():
    c.execute("SELECT * FROM report_data_1")
    data = c.fetchall()
    
    # Separate the returned data into individual lists
    ids, image_paths, texts, embeddings = zip(*data)
    
    # Convert bytes back to numpy arrays for the embeddings
    embeddings_np = [np.frombuffer(embedding, dtype=np.float32).reshape(-1, 1024) for embedding in embeddings]
    
    return ids, image_paths, texts, embeddings_np

def store_report_data(model, tokenizer, batch_size): 

    data = pd.read_csv('/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/data/train_csv_script.csv')
    report_descriptions = data['text'].tolist()
    image_paths = data['image'].tolist()

    num_batches = len(report_descriptions) // batch_size + (1 if len(report_descriptions) % batch_size != 0 else 0)

    def encode_all_reports(texts, model, tokenizer, device="cuda"):
        embeddings = []

        for text in texts:
            encoded_text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            text_embedding = model.module.text_encoder(encoded_text)
            embeddings.append(text_embedding.squeeze())

        return torch.stack(embeddings)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_texts = report_descriptions[start_idx:end_idx]
        batch_images = image_paths[start_idx:end_idx]

        # Process the batch
        batch_embeddings = encode_all_reports(batch_texts, model, tokenizer)

        # Convert tensor embeddings to numpy arrays and then to bytes
        for idx, (embedding, image_path, text) in enumerate(zip(batch_embeddings, batch_images, batch_texts)):
            embedding_np = embedding.detach().cpu().numpy()
            embedding_bytes = embedding_np.tobytes()
            db_idx = start_idx + idx  # Ensure unique ID across batches
            c.execute(
                "INSERT INTO report_data_1 (id, image_path, text, embedding) VALUES (?, ?, ?, ?)", 
                (db_idx, image_path, text, embedding_bytes)
            )

        conn.commit()  # Commit after each batch to save the changes
        print(f'Processed batch {i+1}/{num_batches}')


@hydra.main(config_path="./configs/models", config_name="grounding")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.main.biomegatron)
    model = ImageTextGroundingModelHierarchical.from_config(cfg, device)

    # store_report_data(model, tokenizer, batch_size=10)
    print(get_first_entry())



if __name__ == "__main__":
    main()



