import torch 
import numpy as np
from transformers import AutoTokenizer

import gradio as gr
from gradio import components as gc

from models import ImageTextCorrection
from models.text_correction import Correction
from utils import preprocess_image

from transformers import T5ForConditionalGeneration, T5Tokenizer

import gradio as gr
from gradio import components as gc

import argparse

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import warnings
warnings.filterwarnings("ignore")

import re

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

import os
os.environ["TRANSFORMERS_CACHE"] = "/proj/vondrick/aa4870/freewilly"

from nltk.translate.bleu_score import sentence_bleu
import Levenshtein

import openai
api_key = "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"

from rouge import Rouge
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, BertModel

def compute_metrics(refined_text, ground_truth):

    # Load pre-trained Word2Vec model
    # model = api.load('word2vec-google-news-300')
    # Load Universal Sentence Encoder
    # embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    # Load BERT Model and Tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Compute Edit Distance (Levenshtein Distance)
    edit_distance = Levenshtein.distance(refined_text, ground_truth)
    
    # Compute BLEU Score
    bleu_score = sentence_bleu([ground_truth.split()], refined_text.split())

    # ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(refined_text, ground_truth, avg=True)

    # Word Mover's Distance
    # tokens1 = refined_text.lower().split()
    # tokens2 = ground_truth.lower().split()
    # wmd_distance = model.wmdistance(tokens1, tokens2)

    # # Jaccard Similarity
    # set1 = set(refined_text.lower().split())
    # set2 = set(ground_truth.lower().split())
    # jaccard_sim = len(set1.intersection(set2)) / len(set1.union(set2))

    # # Cosine Similarity with TF-IDF
    # vectorizer = TfidfVectorizer()
    # tfidf_matrix = vectorizer.fit_transform([refined_text, ground_truth])
    # cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # # Universal Sentence Encoder
    # embeddings = embed([refined_text, ground_truth])
    # cosine_sim_use = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]

    # BERT embeddings
    # inputs1 = tokenizer(refined_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # outputs1 = bert_model(**inputs1)
    # embedding1 = outputs1['last_hidden_state'][:, 0, :].detach().numpy()

    # inputs2 = tokenizer(ground_truth, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # outputs2 = bert_model(**inputs2)
    # embedding2 = outputs2['last_hidden_state'][:, 0, :].detach().numpy()

    # cosine_sim_bert = cosine_similarity(embedding1, embedding2)[0][0]

    # Return your metrics along with the new ones
    return edit_distance, bleu_score, rouge_scores#, wmd_distance, jaccard_sim, cosine_sim, cosine_sim_use#, cosine_sim_bert

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

EXAMPLES = [[
    "/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p18/p18218780/s54542901/6fd800dc-21a78caa-c376bc29-424546a5-98b4d8f5.jpg", 
    "FINDINGS: Possible right pleural effusion is seen. There is more atelectasis at the right lower lung. Increased lung volumes on this examination, otherwise the cardiomediastinal silhouette is unchanged. No nodules and mass burden is unchanged since previous examination. IMPRESSION: More atelectasis at the right lower lung then previous radiograph.", 
    "FINDINGS: {Possible right pleural effusion} is seen. There is more atelectasis at the {right} lower lung. {Increased} lung volumes on this examination, otherwise the cardiomediastinal silhouette is unchanged. {No} nodules and mass burden is unchanged since previous examination. IMPRESSION: More atelectasis at the {right} lower lung then previous radiograph.",
    "FINDINGS:  Possible left pleural effusion is seen. There is more atelectasis at the left lower lung. Lower lung volumes on this examination, otherwise the cardiomediastinal silhouette is unchanged. Nodules and mass burden is unchanged since previous examination.  IMPRESSION:  More atelectasis at the left lower lung then previous radiograph."
    ], 
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11696506/s51293135/302aad08-c38beb4d-ce7fe2e5-a705b30e-9da23335.jpg",
    "FINDINGS: PA and axial views of the chest provided. Elevated right hemidiaphragm again noted with associated right basal infiltration. There is no focal lung mass concerning for pneumonia. No edema, small effusion or pneumothorax. The overall cardiomediastinal silhouette appears changed though the right heart borders partially obscured. Bony structures appear abnormal. Anchors are seen imbedded within the right scapula fossa. IMPRESSION: As above. Possible acute findings.",
    "FINDINGS: {PA and axial views of the chest provided}. Elevated right hemidiaphragm again noted with associated right basal {infiltration}. There is no focal {lung mass} concerning for pneumonia. No edema, {small effusion} or pneumothorax. The overall cardiomediastinal silhouette appears {changed} though the right heart borders partially obscured. Bony structures appear {abnormal}. Anchors are seen imbedded within the right {scapula} fossa. IMPRESSION: As above. {Possible acute findings}.",
    "FINDINGS:  PA and lateral views of the chest provided.   Elevated right hemidiaphragm again noted with associated right basal atelectasis.  There is no focal consolidation concerning for pneumonia.  No edema, large effusion or pneumothorax.  The overall cardiomediastinal silhouette appears unchanged though the right heart borders partially obscured.  Bony structures appear intact.  Anchors are seen imbedded within the right glenoid fossa.  IMPRESSION:  As above.  No acute findings."
    ],
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p16/p16599954/s50555368/51b522f3-2723496e-b6ed24dc-4e2b8d7d-3ccae46f.jpg",
    "FINDINGS: No abnormality is seen on the frontal and lateral radiographs of the chest. The cardiac and mediastinal contour is irregular. There is a pleural abnormality detected. IMPRESSION: Acute cardiopulmonary process is present.",
    "FINDINGS: {No abnormality} is seen on the frontal and lateral radiographs of the chest. The cardiac and mediastinal {contour is irregular}. There is a pleural {abnormality} detected. IMPRESSION: {Acute cardiopulmonary process} is present.",
    "FINDINGS:  Frontal and lateral radiographs of the chest demonstrate clear lungs.  The cardiac and mediastinal contours are normal.  No pleural abnormality is detected.  IMPRESSION:  No acute cardiopulmonary process."
    ], 
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11801365/s53180085/bdbe0050-34e81d42-3d04159a-36135c4b-e5ea42c8.jpg",
    "FINDINGS: The patient is rotated to the right. There are likely bilateral pleural effusions. Right base opacity is likely due to combination of pleural effusion and atelectasis, but underlying consolidation due to infection or aspiration is not excluded. A pneumothorax is seen. The cardiomediastinal silhouette is difficult to actually assessed given rib pedestal rotation, but the cardiac silhouette is likely small to mildly enlarged.",
    "FINDINGS: {The patient is rotated to the right}. There are likely bilateral pleural effusions. {Right base opacity} is likely due to combination of pleural effusion and atelectasis, but underlying consolidation due to infection or aspiration is not excluded. {A pneumothorax} is seen. The cardiomediastinal silhouette is difficult to actually assessed given rib {pedestal} rotation, but the cardiac silhouette is likely {small} to mildly enlarged.",
    "FINDINGS:  Patient is rotated to the left.  There are likely bilateral pleural effusions. Left base opacity is likely due to combination of pleural effusion and atelectasis, but underlying consolidation due to infection or aspiration is not excluded.  No pneumothorax is seen.  The cardiomediastinal silhouette is difficult to actually assessed given rib patient rotation, but the cardiac silhouette is likely top-normal to mildly enlarged.",
    ],
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11801365/s53180085/bdbe0050-34e81d42-3d04159a-36135c4b-e5ea42c8.jpg",
    "FINDINGS: Patient is rotated to the right. There are likely unilateral pleural effusions. Left base opacity is likely due to combination of pleural effusion and atelectasis, but underlying consolidation due to infection or asthma is not excluded. No pneumothorax is seen. The cardiomediastinal silhouette is difficult to actually assessed given rib patient rotation, but the cardiac silhouette is likely top-abnormal to strongly enlarged.",
    "FINDINGS: {Patient is rotated to the right}. {There are likely unilateral pleural effusions}. Left base opacity is likely due to combination of pleural effusion and atelectasis, but underlying consolidation due to infection or {asthma} is not excluded. No pneumothorax is seen. The cardiomediastinal silhouette is difficult to actually assessed given rib patient rotation, but the {cardiac silhouette is likely top-abnormal to strongly enlarged}.",
    "FINDINGS:  Patient is rotated to the left.  There are likely bilateral pleural effusions. Left base opacity is likely due to combination of pleural effusion and atelectasis, but underlying consolidation due to infection or aspiration is not excluded.  No pneumothorax is seen.  The cardiomediastinal silhouette is difficult to actually assessed given rib patient rotation, but the cardiac silhouette is likely top-normal to mildly enlarged."
    ], 
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10973582/s59909530/fe36673d-370f97c5-c767322c-5aa030c4-171dc947.jpg",
    "FINDINGS: Cardiac, mediastinal and hilar contoured are abnormal. Pulmonary vasculature is engorged. Focal consolidation, pleural effusion and pneumothorax is present. Degenerative changes are noted in the lumbar spine as well as within the thoracic spine. IMPRESSION: Acute cardiac lung abnormality is identified.",
    "FINDINGS: {Cardiac, mediastinal and hilar contoured} are abnormal. Pulmonary vasculature is {engorged}. Focal {consolidation, pleural effusion and pneumothorax} is present. Degenerative {changes are noted in the lumbar spine} as well as within the thoracic spine. IMPRESSION: Acute {cardiac lung abnormality} is identified.",
    "FINDINGS: Cardiac, mediastinal and hilar contours are normal. Pulmonary vasculature is not engorged. No focal consolidation, pleural effusion or pneumothorax is present. Degenerative changes are noted within both acromioclavicular joints as well as within the thoracic spine. IMPRESSION: No acute cardiopulmonary abnormality.",
    ],
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p18/p18012427/s54748444/1e34cc4e-95a3f46b-d4b9953c-74f728d3-525774ae.jpg",
    "FINDINGS: Cardiac, mediastinal and hilar contours are normal. Pulmonary vascularity is normal. Consolidative opacity in the left upper lobe is concerning for pneumonia. Right lung is clear. No pleural effusion or pneumothorax is seen. No acute osseous abnormality is detected. IMPRESSION: Left lower lobe pneumonia. Follow up radiographs after treatment are recommended to ensure resolution of this finding.",
    "FINDINGS: Cardiac, mediastinal and hilar contours are normal. Pulmonary vascularity is normal. {Consolidative opacity in the left upper lobe is concerning for pneumonia}. Right lung is clear. No pleural effusion or pneumothorax is seen. No acute osseous abnormality is detected. IMPRESSION: {Left lower lobe pneumonia}. Follow up radiographs after treatment are recommended to ensure resolution of this finding.",
    "FINDINGS: Cardiac, mediastinal and hilar contours are normal. Pulmonary vascularity is normal. Consolidative opacity in the left lower lobe is concerning for pneumonia. Right lung is clear. No pleural effusion or pneumothorax is seen. No acute osseous abnormality is detected. IMPRESSION: Left lower lobe pneumonia. Follow up radiographs after treatment are recommended to ensure resolution of this finding.",
    ],
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p13/p13485940/s50836836/3a45270f-da7f2455-91af41a6-56f29453-dd3d64d3.jpg",
    "FINDINGS PA and lateral views of the chest provided. There is significant focal consolidation, effusion, and pneumothorax. The cardiomediastinal silhouette is abnormal. Signs of congestion and edema are observed. Imaged osseous structures are fractured. Free air is present below the right hemidiaphragm. IMPRESSION: Acute intrathoracic process is present.",
    "FINDINGS PA and lateral views of the chest provided. There is {significant focal consolidation, effusion, and pneumothorax}. The cardiomediastinal silhouette is {abnormal}. {Signs of congestion and edema} are observed. Imaged osseous structures are {fractured}. {Free air is present below the right hemidiaphragm}. IMPRESSION: {Acute intrathoracic process is present}.",
    "FINDINGS: PA and lateral views of the chest provided. There is no focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette is normal. No signs of congestion or edema. Imaged osseous structures are intact. No free air below the right hemidiaphragm is seen. IMPRESSION: No acute intrathoracic process.",
    ],
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11982468/s51202251/b63bc9ef-ebd07e7d-4512fd24-e6466103-4370f0f6.jpg",
    "FINDINGS: Widening of the inferior mediastinum is consistent with the patient's known lymphadenopathy. Enlargement of the left hilum also consistent with the patient's known lymphadenopathy. Severe cardiomegaly noted. There is pneumothorax of the bilateral lung bases. Large left pleural effusion. Free air under the diaphragm is seen.",
    "FINDINGS: {Widening of the inferior mediastinum is consistent with the patient's known lymphadenopathy}. Enlargement of the {left} hilum also consistent with the patient's known lymphadenopathy. {Severe} cardiomegaly noted. There is {pneumothorax} of the bilateral lung bases. {Large} left pleural effusion. {Free air under the diaphragm is seen}. ",
    "FINDINGS: Widening of the superior mediastinum is consistent with the patient's known lymphadenopathy. Enlargement of the right hilum also consistent with the patient's known lymphadenopathy. Mild cardiomegaly noted. There is atelectasis of the bilateral lung bases. Small left pleural effusion. No pneumothorax seen. No free air under the diaphragm. ",
     ],
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p12/p12960546/s57503496/1eaeb60d-138b4256-819d59e4-6366772c-1c4d9342.jpg",
    "FINDINGS: As compared to the previous radiograph, there is decreasing size of the cardiac silhouette. Also, the pre-existing right pleural effusion has substantially decreased, it occupies now approximately half of the right hemithorax. A minimal left pleural effusion is present in changed manner. Subsequent areas of consolidation at the left and right lung base. No evidence of emphysema. Unchanged course of the pacemaker leads, unchanged position of the pacemaker in the right pectoral region.",
    "INDINGS:{ As compared to the previous radiograph, there is decreasing size of the cardiac silhouette. }Also, the pre-existing right pleural effusion has substantially decreased, it occupies now approximately half of the right hemithorax. A minimal left pleural effusion is present in changed manner. Subsequent areas of consolidation {at the left and right lung base}. No evidence of emphysema. {Unchanged course of the pacemaker leads, unchanged position of the pacemaker in the right pectoral region}.",
    "FINDINGS: As compared to the previous radiograph, there is increasing size of the cardiac silhouette. Also, the pre-existing right pleural effusion has substantially increased, it occupies now approximately half of the right hemithorax. A minimal left pleural effusion is present in unchanged manner. Subsequent areas of atelectasis at the left and right lung base. No evidence of pneumonia. Unchanged course of the pacemaker leads, unchanged position of the pacemaker in the left pectoral region.",
    ],
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p12/p12835259/s53868892/b0135f23-6428374c-6af30404-253276cd-52610318.jpg",
    "FINDINGS: In comparison with the study of yesterday, there is hazy opacification at only one base that appears to be decreasing, consistent with layering pleural effusion and compressive atelectasis. The possibility of supervening appendicitis would be difficult to exclude in the appropriate clinical setting.",
    "FINDINGS: In comparison with the study of {yesterday}, there is hazy opacification {at only one base} that appears to be {decreasing}, consistent with layering pleural effusion and compressive atelectasis. The possibility of supervening {appendicitis} would be difficult to exclude in the appropriate clinical setting.",
    "FINDINGS: In comparison with the study of , there is hazy opacification at both bases that appears to be increasing, consistent with layering pleural effusion and compressive atelectasis. The possibility of supervening pneumonia would be difficult to exclude in the appropriate clinical setting.",
    ],
    ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p17/p17137598/s50757588/e4c2606b-bbd5e904-7309afd1-919295a0-54002b88.jpg",
    "FINDINGS: Sternotomy. Right IJ central line tip low SVC. Elevated left hemidiaphragm, similar. Improved unilateral opacities. Trace fluid versus atelectasis in the right upper lung. Tortuous abdominal aorta. Small bilateral pleural effusions. Severe compression fracture in the upper thoracic spine. IMPRESSION: No change since prior exam.",
    "FINDINGS: Sternotomy. Right IJ central line tip low {SVC}. Elevated {left} hemidiaphragm, similar. Improved {unilateral} opacities. Trace fluid versus atelectasis {in the right upper lung}. Tortuous {abdominal aorta}. Small bilateral pleural effusions. {Severe} compression fracture {in the upper thoracic spine}. IMPRESSION: {No change} since prior exam.",
    "FINDINGS: Sternotomy. Right IJ central line tip low SVC. Elevated right hemidiaphragm, similar. Improved bibasilar opacities. Trace fluid versus atelectasis right lower lung. Tortuous thoracic aorta. Small bilateral pleural effusions. Mild compression fracture lower thoracic spine. IMPRESSION: Improvement since prior exam",
     ], 
     ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p16/p16355861/s52923104/6caa27ac-e170179d-b407ecc2-b2656b31-499e7dda.jpg", 
     "FINDINGS: There is a right basilar opacity likely due to a combination of moderate pleural effusion and underlying atelectasis. Blunting of the lateral and posterior costophrenic angles on the {left} suggests small effusion. These changes have decreased since prior exam. Pulmonary vascular congestion is not noted. The cardiomediastinal silhouette is stable. Mitral annular calcifications are again noted. {Right} chest wall dual lead pacing device is again noted. No acute osseous abnormalities. IMPRESSION: Moderate {right} pleural effusion with underlying atelectasis noting infection would also be possible. Pulmonary vascular congestion and probable {large} right pleural effusion as well.",
     "FINDINGS: There is a right basilar opacity likely due to a combination of moderate pleural effusion and underlying atelectasis. Blunting of the lateral and posterior costophrenic angles on the {left} suggests small effusion. These changes have decreased since prior exam. Pulmonary vascular congestion is not noted. The cardiomediastinal silhouette is stable. Mitral annular calcifications are again noted. {Right} chest wall dual lead pacing device is again noted. No acute osseous abnormalities. IMPRESSION: Moderate {right} pleural effusion with underlying atelectasis noting infection would also be possible. Pulmonary vascular congestion and probable {large} right pleural effusion as well.",
     "FINDINGS: There is a left basilar opacity likely due to a combination of moderate pleural effusion and underlying atelectasis. Blunting of the lateral and posterior costophrenic angles on the right suggests small effusion. These changes have increased since prior exam. Pulmonary vascular congestion is again noted. The cardiomediastinal silhouette is stable. Mitral annular calcifications are again noted. Left chest wall dual lead pacing device is again noted. No acute osseous abnormalities. IMPRESSION: Moderate left pleural effusion with underlying atelectasis noting infection would also be possible. Pulmonary vascular congestion and probable small right pleural effusion as well.",
     ],
     ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p15/p15024955/s55985529/47d21856-c549ab3b-62daeb3b-a564407b-a8d74713.jpg",
     "FINDINGS: {There is a significant change since the prior CXR}. {Unchanged appearance of neophagus}. Linear atelectasis is noted at the right lung base. Stable appearance of small left pleural effusion. Tiny right pleural effusion has largely resolved. The lungs are otherwise free of focal consolidations or pneumothorax. No pulmonary edema. Cardiomediastinal silhouette is within normal limits. Surgical clips are noted in the left upper quadrant. IMPRESSION: Stable small left pleural effusion. Right effusion has largely resolved. {No acute intrapulmonary process}.",
     "FINDINGS: {There is a significant change since the prior CXR}. {Unchanged appearance of neophagus}. Linear atelectasis is noted at the right lung base. Stable appearance of small left pleural effusion. Tiny right pleural effusion has largely resolved. The lungs are otherwise free of focal consolidations or pneumothorax. No pulmonary edema. Cardiomediastinal silhouette is within normal limits. Surgical clips are noted in the left upper quadrant. IMPRESSION: Stable small left pleural effusion. Right effusion has largely resolved. {No acute intrapulmonary process}.",
     "FINDINGS: There are no significant changes since the prior CXR performed . Unchanged appearance of neoesophagus. Linear atelectasis is noted at the right lung base. Stable appearance of small right pleural effusion. Tiny left pleural effusion has largely resolved. The lungs are otherwise free of focal consolidations or pneumothorax. No pulmonary edema. Cardiomediastinal silhouette is within normal limits. Surgical clips are noted in the left upper quadrant. IMPRESSION: Stable small right pleural effusion. Left effusion has largely resolved. No acute intrapulmonary process.",
     ],
     ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11896347/s52172826/f1f22f5f-42b39fcb-04c88f56-79b0889a-69085c5a.jpg",
     "FINDINGS: {A left-sided PICC is seen terminating in the the very proximal SVC without evidence of pneumothorax}. The lungs are clear with {focal consolidation}. {No pleural effusion or pneumothorax is seen}. The cardiac and mediastinal silhouettes are {unstable}. {No pulmonary} edema is seen. IMPRESSION: {Acute cardiopulmonary process}. {Pulmonary edema}.",
     "FINDINGS: {A left-sided PICC is seen terminating in the the very proximal SVC without evidence of pneumothorax}. The lungs are clear with {focal consolidation}. {No pleural effusion or pneumothorax is seen}. The cardiac and mediastinal silhouettes are {unstable}. {No pulmonary} edema is seen. IMPRESSION: {Acute cardiopulmonary process}. {Pulmonary edema}.",
     "FINDINGS: A right-sided PICC is seen terminating in the the very proximal SVC without evidence of pneumothorax. The lungs are clear without focal consolidation. No pleural effusion or pneumothorax is seen. The cardiac and mediastinal silhouettes are stable. No pulmonary edema is seen. IMPRESSION: No acute cardiopulmonary process. No pulmonary edema.,"
     ],
     ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11044665/s52563181/180c3df0-9a38d78a-54b92eec-09b5e49a-b3c70cb3.jpg",
     "FINDINGS: The {lung volumes are high} but clear. {Heart size is bottom-normal}, unchanged since . The mediastinal and hilar contours are normal. {There is a pleural effusion} or pneumothorax. IMPRESSION: {High lung volumes} but evidence of pneumonia. {Heart size bottom-normal}, unchanged.",
     "FINDINGS: The {lung volumes are high} but clear. {Heart size is bottom-normal}, unchanged since . The mediastinal and hilar contours are normal. {There is a pleural effusion} or pneumothorax. IMPRESSION: {High lung volumes} but evidence of pneumonia. {Heart size bottom-normal}, unchanged.",
     "FINDINGS: The lung volumes are low but clear. Heart size is top-normal, unchanged since . The mediastinal and hilar contours are normal. There is no pleural effusion or pneumothorax. IMPRESSION: Low lung volumes but no evidence of pneumonia. Heart size top normal, unchanged.",
     ],
     ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10070928/s56788284/f9ba0a37-3bfff3ae-caadb2d3-e96e4625-31cd47e0.jpg",
     "FINDINGS: AP single view of the chest has {been obtained with patient in prone} position. Comparison is made with the next preceding similar study of . Heart size, thoracic aorta and {medial} mediastinal structures unchanged. No advanced pulmonary vascular congestion is noted. The next preceding {portable chest examination of} identified pulmonary parenchymal densities overlying the {right upper} right lower lung field has cleared and the chest findings are now similar to what has been shown on the {oblique} frontal view of the PA and lateral chest examination of . A left-sided PICC line was already identified on that examination, well {not} detected on the lateral view and seen to overly the",
     "FINDINGS: AP single view of the chest has {been obtained with patient in prone} position. Comparison is made with the next preceding similar study of . Heart size, thoracic aorta and {medial} mediastinal structures unchanged. No advanced pulmonary vascular congestion is noted. The next preceding {portable chest examination of} identified pulmonary parenchymal densities overlying the {right upper} right lower lung field has cleared and the chest findings are now similar to what has been shown on the {oblique} frontal view of the PA and lateral chest examination of . A left-sided PICC line was already identified on that examination, well {not} detected on the lateral view and seen to overly the",
     "FINDINGS: AP single view of the chest has been obtained with patient in sitting semi-upright position. Comparison is made with the next preceding similar study of . Heart size, thoracic aorta and mediastinal structures unchanged. No advanced pulmonary vascular congestion is noted. The next preceding portable chest examination of , identified pulmonary parenchymal densities overlying the right lower lung field has cleared and the chest findings are now similar to what has been shown on the frontal view of the PA and lateral chest examination of . A left-sided PICC line was already identified on that examination, well detected on the lateral view and seen to overly the superior mediastinal structures on the frontal view, so to be located within the SVC at the level 2 cm above the carina. Although the quality of the portable chest examinations are not identical and the PICC line less contrast prominent as the wire apparently has been removed, the line appears to be in unchanged position. No pneumothorax is identified in the apical area. IMPRESSION: Apparently unchanged position of previously identified left-sided PICC line seen on PA and lateral of and AP single view examination of ."]]


# 32296,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p16/p16355861/s52923104/6caa27ac-e170179d-b407ecc2-b2656b31-499e7dda.jpg,FINDINGS: There is a left basilar opacity likely due to a combination of moderate pleural effusion and underlying atelectasis. Blunting of the lateral and posterior costophrenic angles on the right suggests small effusion. These changes have increased since prior exam. Pulmonary vascular congestion is again noted. The cardiomediastinal silhouette is stable. Mitral annular calcifications are again noted. Left chest wall dual lead pacing device is again noted. No acute osseous abnormalities. IMPRESSION: Moderate left pleural effusion with underlying atelectasis noting infection would also be possible. Pulmonary vascular congestion and probable small right pleural effusion as well.,FINDINGS: {There is a right basilar opacity} likely due to a combination of moderate pleural effusion and underlying atelectasis. {Blunting of the medial and posterior costophrenic angles} on the right suggests small effusion. These changes have decreased since prior exam. {Pulmonary vascular congestion is again not noted}. The cardiomediastinal silhouette is stable. {Mitral annular calcifications are again not noted}. Right chest wall dual lead pacing device is again noted. No acute osseous abnormalities. IMPRESSION: {Moderate right pleural effusion} with underlying atelectasis noting infection would also be possible. {Pulmonary vascular congestion and probable large right pleural eff,"[0, 1, 3, 5, 8, 9]"
# 32298,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p15/p15024955/s55985529/47d21856-c549ab3b-62daeb3b-a564407b-a8d74713.jpg,FINDINGS: There are no significant changes since the prior CXR performed . Unchanged appearance of neoesophagus. Linear atelectasis is noted at the right lung base. Stable appearance of small right pleural effusion. Tiny left pleural effusion has largely resolved. The lungs are otherwise free of focal consolidations or pneumothorax. No pulmonary edema. Cardiomediastinal silhouette is within normal limits. Surgical clips are noted in the left upper quadrant. IMPRESSION: Stable small right pleural effusion. Left effusion has largely resolved. No acute intrapulmonary process.,,"[0, 1, 10, 11]"
# 32300,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11896347/s52172826/f1f22f5f-42b39fcb-04c88f56-79b0889a-69085c5a.jpg,FINDINGS: A right-sided PICC is seen terminating in the the very proximal SVC without evidence of pneumothorax. The lungs are clear without focal consolidation. No pleural effusion or pneumothorax is seen. The cardiac and mediastinal silhouettes are stable. No pulmonary edema is seen. IMPRESSION: No acute cardiopulmonary process. No pulmonary edema.,FINDINGS: {A left-sided PICC is seen terminating in the the very proximal SVC without evidence of pneumothorax}. The lungs are {opaque with focal consolidation}. {A pleural effusion is seen}. The cardiac and mediastinal silhouettes are {unstable}. {Moderate pulmonary edema is seen}. IMPRESSION: {No acute cardiopulmonary process}. {No pulmonary edema}.,"[0, 1, 2, 3, 4, 5, 6]"
# 32301,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p18/p18282058/s54031132/d43b7a00-44859e47-0194a80d-a3061679-2ba7a282.jpg,FINDINGS: The cardiomediastinal silhouette and pulmonary vasculature are normal. The lungs are clear. There is no pleural effusion or pneumothorax. IMPRESSION: No acute intrathoracic abnormality.,FINDINGS: The cardiomediastinal silhouette and {pulmonary vasculature} are abnormal. The {lungs} are not clear. There is a {pleural effusion} and {pneumothorax}. IMPRESSION: {Acute} intrathoracic abnormality.,"[0, 1, 2, 3]"
# 32302,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p18/p18282058/s54031132/d43b7a00-44859e47-0194a80d-a3061679-2ba7a282.jpg,FINDINGS: The cardiomediastinal silhouette and pulmonary vasculature are normal. The lungs are clear. There is no pleural effusion or pneumothorax. IMPRESSION: No acute intrathoracic abnormality.,FINDINGS: {The cardiomediastinal silhouette and pulmonary vasculature are abnormal}. The lungs are clear. There is no pleural effusion or pneumothorax. {IMPRESSION: Acute intrathoracic abnormality}. IMPRESSION: {No abnormalities} are present.,"[0, 3]"
# 32303,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p15/p15491742/s54925028/fa664a6f-a9ee76a5-80ed95df-afb41fa3-88e9fd9e.jpg,"FINDINGS: As compared to the previous radiograph, there is unchanged evidence of thoracic scoliosis, leading to asymmetry of the rib cage and differences in radiolucency of the lung parenchyma. However, there is no evidence of focal parenchymal opacities on the frontal and the lateral radiograph. No pleural effusions. Normal size of the cardiac silhouette. Normal hilar and mediastinal structures. No pleural effusions. No pulmonary edema.","FINDINGS: As compared to the previous radiograph, there is {unchanged evidence of thoracic kyphosis}, leading to asymmetry of the rib cage and differences in radiolucency of the lung parenchyma. However, there is {no evidence of focal parenchymal opacities on the frontal and the lateral radiograph.} {No pleural emphysema.} Normal size of the cardiac silhouette. Normal hilar and mediastinal structures. No pleural effusions. {No pulmonary edema.}","[0, 1, 2, 3, 4]"
# 32304,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p15/p15491742/s54925028/fa664a6f-a9ee76a5-80ed95df-afb41fa3-88e9fd9e.jpg,"FINDINGS: As compared to the previous radiograph, there is unchanged evidence of thoracic scoliosis, leading to asymmetry of the rib cage and differences in radiolucency of the lung parenchyma. However, there is no evidence of focal parenchymal opacities on the frontal and the lateral radiograph. No pleural effusions. Normal size of the cardiac silhouette. Normal hilar and mediastinal structures. No pleural effusions. No pulmonary edema.","FINDINGS: As compared to the previous radiograph, there is unchanged evidence of thoracic scoliosis, leading to asymmetry of the rib cage and differences in radiolucency of the lung parenchyma. {However, there is evidence of focal parenchymal opacities on the frontal and the lateral radiograph. }No pleural {effusionss. }Normal size of the cardiac silhouette. {Abnormal }hilar and mediastinal structures. No pleural {effusionss. }No pulmonary edema.","[2, 3, 4, 5, 6]"
# 32305,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p14/p14504848/s59937671/87784bea-11aeb6c6-e92be372-635f959d-4ffed18b.jpg,FINDINGS: The lung volumes are normal. Moderate cardiomegaly with signs of minimal fluid overload but no overt pulmonary edema. No pleural effusions. No focal parenchymal opacity suggesting pneumonia. No pneumothorax.,FINDINGS: {The lung volumes are abnormal. Moderate cardiomegaly with signs of moderate fluid overload and overt pulmonary edema}. No pleural effusions. {Focal parenchymal opacity suggesting pneumothorax.} No pneumothorax.,"[0, 1, 3]"
# 32306,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p14/p14504848/s59937671/87784bea-11aeb6c6-e92be372-635f959d-4ffed18b.jpg,FINDINGS: The lung volumes are normal. Moderate cardiomegaly with signs of minimal fluid overload but no overt pulmonary edema. No pleural effusions. No focal parenchymal opacity suggesting pneumonia. No pneumothorax.,FINDINGS: {The lung volumes are reduced}. {Moderate cardiomegaly with signs of significant fluid overload and overt pulmonary edema.} {Large bilateral pleural effusions.} {Focal parenchymal opacity suggesting pneumonia.} {A small pneumothorax.},"[0, 1]"
# 32308,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11044665/s52563181/180c3df0-9a38d78a-54b92eec-09b5e49a-b3c70cb3.jpg,"FINDINGS: The lung volumes are low but clear. Heart size is top-normal, unchanged since . The mediastinal and hilar contours are normal. There is no pleural effusion or pneumothorax. IMPRESSION: Low lung volumes but no evidence of pneumonia. Heart size top normal, unchanged.","FINDINGS: The {liver} volumes are {high} but clear. Heart size is {small}, {increased} since . The mediastinal and hilar contours are {unusual}. There is a {moderate} pleural effusion or {bronchitis}. IMPRESSION: {Large} lung volumes but {evidence} of pneumonia. Heart size {decreased}, {decreased} since .","[0, 1, 2, 3, 4, 5]"
# 32309,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10070928/s56788284/f9ba0a37-3bfff3ae-caadb2d3-e96e4625-31cd47e0.jpg,"FINDINGS: AP single view of the chest has been obtained with patient in sitting semi-upright position. Comparison is made with the next preceding similar study of . Heart size, thoracic aorta and mediastinal structures unchanged. No advanced pulmonary vascular congestion is noted. The next preceding portable chest examination of , identified pulmonary parenchymal densities overlying the right lower lung field has cleared and the chest findings are now similar to what has been shown on the frontal view of the PA and lateral chest examination of . A left-sided PICC line was already identified on that examination, well detected on the lateral view and seen to overly the superior mediastinal structures on the frontal view, so to be located within the SVC at the level 2 cm above the carina. Although the quality of the portable chest examinations are not identical and the PICC line less contrast prominent as the wire apparently has been removed, the line appears to be in unchanged position. No pneumothorax is identified in the apical area. IMPRESSION: Apparently unchanged position of previously identified left-sided PICC line seen on PA and lateral of and AP single view examination of .","FINDINGS: AP single view of the chest has been obtained with patient in sitting semi-upright position. Comparison is made with the next preceding {similar study} of . Heart size, thoracic aorta and mediastinal structures {unchanged}. No advanced pulmonary vascular congestion is noted. The next preceding portable chest examination of {, identified pulmonary parenchymal densities} overlying the right lower lung field has cleared and the chest findings are now {similar to what has been shown on the frontal view of the PA} and lateral chest examination of . A left-sided PICC line was already identified on that examination, {well detected} on the lateral view and seen to overly the superior mediastinal structures on the frontal view","[1, 2, 4, 5]"
# 32310,/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10070928/s56788284/f9ba0a37-3bfff3ae-caadb2d3-e96e4625-31cd47e0.jpg,"FINDINGS: AP single view of the chest has been obtained with patient in sitting semi-upright position. Comparison is made with the next preceding similar study of . Heart size, thoracic aorta and mediastinal structures unchanged. No advanced pulmonary vascular congestion is noted. The next preceding portable chest examination of , identified pulmonary parenchymal densities overlying the right lower lung field has cleared and the chest findings are now similar to what has been shown on the frontal view of the PA and lateral chest examination of . A left-sided PICC line was already identified on that examination, well detected on the lateral view and seen to overly the superior mediastinal structures on the frontal view, so to be located within the SVC at the level 2 cm above the carina. Although the quality of the portable chest examinations are not identical and the PICC line less contrast prominent as the wire apparently has been removed, the line appears to be in unchanged position. No pneumothorax is identified in the apical area. IMPRESSION: Apparently unchanged position of previously identified left-sided PICC line seen on PA and lateral of and AP single view examination of .","FINDINGS: AP single view of the chest has {been obtained with patient in prone} position. Comparison is made with the next preceding similar study of . Heart size, thoracic aorta and {medial} mediastinal structures unchanged. No advanced pulmonary vascular congestion is noted. The next preceding {portable chest examination of} identified pulmonary parenchymal densities overlying the {right upper} right lower lung field has cleared and the chest findings are now similar to what has been shown on the {oblique} frontal view of the PA and lateral chest examination of . A left-sided PICC line was already identified on that examination, well {not} detected on the lateral view and seen to overly the","[0, 2, 4, 5]"


@hydra.main(config_path="./configs/models", config_name="error_classification")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    def format_text(text):
        text = re.sub(r' \.', '.', text)
        sentences = text.split('. ')
        capitalized_sentences = [s.capitalize() for s in sentences if s]
        
        return '. '.join(capitalized_sentences)

    def classify_tokens(image_path, input_text, error_locations, ground_truth, threshold):
        # 1. Error Identification part
        tokenizer = AutoTokenizer.from_pretrained(cfg.main.biomegatron)
        model = ImageTextCorrection.from_config(cfg, device)
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        encoded_text = tokenizer(input_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
        output = model(image_tensor, encoded_text)
        m = torch.nn.Sigmoid()
        sigmoid_outputs = m(output)

        print(sigmoid_outputs[0])

        sigmoid_outputs = sigmoid_outputs.cpu().detach().numpy()
        input_ids = encoded_text['input_ids'].cpu().numpy()[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        grouped_tokens = []
        grouped_errors = []
        grouped_sigmoid_vals = [] 
        temp_token = ''

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
                temp_error = 0 if output < float(1 - threshold) else 1  
                temp_sigmoid = output[0]  

        grouped_tokens.append(temp_token)
        grouped_errors.append(temp_error)
        grouped_sigmoid_vals.append(temp_sigmoid)
        html_output = ""
        modified_text = "" # holds the text with errors wrapped in curly braces

        # 1.2 Color code here for the gradio interface and include curly braces for text formating
        for token, error, sigmoid_val in zip(grouped_tokens, grouped_errors, grouped_sigmoid_vals):
            if token == "[PAD]":
                continue
            if error == 0:  # if it's an error, let's make it red
                html_output += f'<span style="color:red; border-bottom: 2px dotted black; font-size: 20px;">{token} </span>'
                modified_text += f"{{{token}}} "
            else:
                html_output += f'<span style="font-size: 16px;">{token} </span>'
                modified_text += f"{token} "

        html_output = format_text(html_output)
        print("Modified text: ", modified_text)

        # 2. Error correction part
        correction_model = Correction.from_config(cfg, device)
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
        refined_text = refine_report_with_gpt4(predicted_text, error_locations)
        corrected_html_output = f'Corrected text: <p><span style="color:green; border-bottom: 2px dotted black; font-size: 16px;">{refined_text}</span></p>'

        edit_distance, bleu, rouge_scores = compute_metrics(refined_text, ground_truth)

        metrics_html_output = f'''
        <p>Edit Distance: <span style="color:blue;">{edit_distance}</span></p>
        <p>BLEU Score: <span style="color:blue;">{bleu}</span></p>
        <p>ROUGE: <span style="color:blue;">{rouge_scores}</span></p>
        <p>Word Mover's Distance: <span style="color:blue;"> </span></p>
        <p>Jaccard Similarity: <span style="color:blue;"> </span></p>
        <p>Cosine simlarity (TF-IDF): <span style="color:blue;"> </span></p>
        <p>Cosine similarity (USE): <span style="color:blue;"> </span></p>
        <p>Cosine similarity (BERT): <span style="color:blue;"> </span></p>
        '''

        return f"Errors identified: <p>{html_output}</p>", corrected_html_output


    iface = gr.Interface(
        fn=classify_tokens, 
        inputs=[
            gr.inputs.Image(type="filepath", label="Image Input"), 
            gr.inputs.Textbox(label="Text Input"), 
            gr.inputs.Textbox(label="Error locations in curly braces"),
            gr.inputs.Textbox(label="Ground truth description"),
            gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.65, label="Error Sensitivity Threshold"),
        ], 
        outputs=[
            gr.outputs.HTML(label="Classified Tokens"), 
            gr.outputs.HTML(label="Corrected Text"),
            # gr.outputs.HTML(label="Metrics (compare autocorrected text with the groundtruth)")  # This is the added line for the metrics
        ],
        examples=EXAMPLES,
    )
    iface.launch(share=True)

    
if __name__ == "__main__":
    main()

# use this when it fails to run pip install gradio_client==0.2.7

# This will be useful when computing metrics of the model performace

