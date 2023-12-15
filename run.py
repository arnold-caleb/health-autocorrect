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

import faiss
import sqlite3

import warnings
warnings.filterwarnings("ignore")

text1 = "FINDINGS: There is no focal consolidation, pleural effusion or pneumothorax. Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal. Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted. IMPRESSION: No acute cardiopulmonary process."
text2 = "FINDINGS: PA and lateral views of the chest provided. The lungs are adequately aerated. There is a focal consolidation at the left lung base adjacent to the lateral hemidiaphragm. There is mild vascular engorgement. There is bilateral apical pleural thickening. The cardiomediastinal silhouette is remarkable for aortic arch calcifications. The heart is top normal in size. IMPRESSION: Focal consolidation at the left lung base, possibly representing aspiration or pneumonia. Central vascular engorgement."
text3 = "FINDINGS: PA and lateral views of the chest provided. Lung volumes are somewhat low. There is no focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette is normal. Imaged osseous structures are intact. No free air below the right hemidiaphragm is seen. IMPRESSION: No acute intrathoracic process."
text4 = "FINDINGS: Widening of the superior mediastinum is consistent with the patient's known lymphadenopathy. Enlargement of the right hilum also consistent with the patient's known lymphadenopathy. Mild cardiomegaly noted. There is atelectasis of the bilateral lung bases. Small left pleural effusion. No pneumothorax seen. No free air under the diaphragm. "
text5 = "FINDINGS: In comparison with the study of , there is hazy opacification at both bases that appears to be increasing, consistent with layering pleural effusion and compressive atelectasis. The possibility of supervening pneumonia would be difficult to exclude in the appropriate clinical setting."
text6 = "FINDINGS:  Frontal and lateral radiographs of the chest demonstrate clear lungs.  The cardiac and mediastinal contours are normal.  No pleural abnormality is detected.  IMPRESSION:  No acute cardiopulmonary process."
    

EXAMPLES = [["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg", text1],
            ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000764/s57375967/096052b7-d256dc40-453a102b-fa7d01c6-1b22c6b4.jpg", text2],
            ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000898/s50771383/0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c.jpg", text3],
            ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11982468/s51202251/b63bc9ef-ebd07e7d-4512fd24-e6466103-4370f0f6.jpg", text4],
            ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p12/p12835259/s53868892/b0135f23-6428374c-6af30404-253276cd-52610318.jpg", text5],
            ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p16/p16599954/s50555368/51b522f3-2723496e-b6ed24dc-4e2b8d7d-3ccae46f.jpg", text6]]

DESCRIPTION = "A image-text grounding model to the task of identifying potential discrepancies in radiological reports. The system cross-references the written report (text) against the associated medical images. Using a Gradio interface, the sentences in the radiological report are ranked based on their relevance to the content of the corresponding images. These sentences are categorized into five levels of relevance: 'Not Relevant', 'Slightly Relevant', 'Neutral', 'Relevant', and 'Highly Relevant'. By highlighting sentences with low relevance to the images, the tool aids in detecting potential errors or oversights in the reports, serving as a valuable tool in quality assurance and medical report auditing."

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

def retrieve_best_reports(model, tokenizer, image_path, faiss_index, db_path): 
    image_tensor = preprocess_image(image_path)
    # assuming your model has a method to get the image embedding
    _, _, image_embedding = encode_text_and_image("", image_tensor, model, tokenizer)
    image_embedding = image_embedding.cpu().numpy()
    image_embedding = image_embedding / np.linalg.norm(image_embedding)

    # Search the FAISS index for the top 1 most similar report embeddings
    D, I = faiss_index.search(image_embedding.reshape(1, -1), 5)  # Adjust the second parameter to retrieve more or fewer reports
   
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

def create_gradio_interface(model, tokenizer, port_number, device="cuda"):

    def find_best_sentences(image, text, *args):
        # finding best sentences
        html_output = ""
        categories = ["Definitely Correct", "Likely Correct", "Neutral", "Likely Incorrect", "Definitely Incorrect"]
        colors = ["#70C1B3", "#B2DBBF", "#F3FFBD", "#dec7ad", "#f0972b"] 

        image_tensor = preprocess_image(image)
        _, top_k_sentences, top_k_sentence_values, _ = get_top_k_sentences(image_tensor, text, model, tokenizer, device=device)

        for sentence, value in zip(top_k_sentences, top_k_sentence_values):
            if value < 0.02:
                category_index = 4  # Definitely Incorrect
            elif 0.02 <= value < 0.04:
                category_index = 3  # Likely Incorrect
            elif 0.04 <= value < 0.06:
                category_index = 2  # Neutral
            elif 0.06 <= value < 0.08:
                category_index = 1  # Likely Correct
            else:
                category_index = 0  # Definitely Correct

            html_output += f'<span style="background-color: {colors[category_index]}; padding: 2px; margin: 20px 0;">{sentence}</span> '

        color_legend = "<h3>Color Legend:</h3>"

        for category, color in zip(categories, colors):
            color_legend += f'<span style="background-color: {color}; padding: 5px;">{category}</span><br /><br />'

        # retrieve the top reports
        import faiss
        db_path = '/proj/vondrick/aa4870/embeddings.db'

        def create_faiss_index():
            _, _, _, embeddings_np = get_data(db_path)  
            embeddings_matrix = np.vstack(embeddings_np)  
            index = faiss.IndexFlatIP(embeddings_matrix.shape[1])  # Create FAISS index for inner product
            index.add(embeddings_matrix)  # Add normalized embeddings to the index
            return index

        faiss_index = create_faiss_index()

        top_reports = retrieve_best_reports(model, tokenizer, image, faiss_index, db_path)
        
        reports_text = "\n\n".join(top_reports)

        return html_output, color_legend, reports_text

    image_input = gr.inputs.Image(type='filepath', label="Chest Radiograph (X-ray scan)")
    text_input = gr.inputs.Textbox(lines=5, label="Radiological report")

    gr.Interface(fn=find_best_sentences, inputs=[image_input, text_input], 
                 outputs=[gr.outputs.HTML(label="Sentences"), gr.outputs.HTML(label="Color Legend"), gr.outputs.Textbox(label="Best Retrieved Reports") ],
                 title="Summer 2023 -- Medical Error Detection", description=DESCRIPTION,
                 examples=EXAMPLES).launch(server_port=1766, share=True)

@hydra.main(config_path="./configs/models", config_name="grounding")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.main.biomegatron)
    model = ImageTextGroundingModelHierarchical.from_config(cfg, device)

    create_gradio_interface(model, tokenizer, cfg.main.port_number)
        
class ReportRetrievalSystem:
    def __init__(self, model, tokenizer, faiss_index, db_path, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.faiss_index = faiss_index
        self.db_path = db_path

    def retrieve_top_reports(self, query_image, top_k=10):
        _, _, image_embedding = encode_text_and_image("", query_image, self.model, self.tokenizer)
        image_embedding = image_embedding.squeeze().cpu().numpy()  # Ensure the image_embedding shape is [b, 1024]

        D, I = self.faiss_index.search(image_embedding.reshape(1, -1), top_k)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        top_reports = []
        for idx in I[0]:
            c.execute("SELECT text FROM report_data_1 WHERE id=?", (idx,))
            report = c.fetchone()[0]
            top_reports.append(report)
        
        conn.close()
        
        return top_reports

if __name__ == "__main__":
    main()
