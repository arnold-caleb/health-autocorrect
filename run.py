import torch 
import numpy as np
from transformers import AutoTokenizer

import gradio as gr
from gradio import components as gc

from model import ImageTextGroundingModelHierarchical
from utils import get_top_k_tokens
from helpers import extract_findings, preprocess_image

text1 = "FINDINGS: There is no focal consolidation, pleural effusion or pneumothorax. Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal. Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted. IMPRESSION: No acute cardiopulmonary process."
text2 = "FINDINGS: PA and lateral views of the chest provided. The lungs are adequately aerated. There is a focal consolidation at the left lung base adjacent to the lateral hemidiaphragm. There is mild vascular engorgement. There is bilateral apical pleural thickening. The cardiomediastinal silhouette is remarkable for aortic arch calcifications. The heart is top normal in size. IMPRESSION: Focal consolidation at the left lung base, possibly representing aspiration or pneumonia. Central vascular engorgement."
text3 = "FINDINGS: PA and lateral views of the chest provided. Lung volumes are somewhat low. There is no focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette is normal. Imaged osseous structures are intact. No free air below the right hemidiaphragm is seen. IMPRESSION: No acute intrathoracic process."

EXAMPLES = [["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg", extract_findings(text1)],
            ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000764/s57375967/096052b7-d256dc40-453a102b-fa7d01c6-1b22c6b4.jpg", extract_findings(text2)],
            ["/proj/vondrick/aa4870/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000898/s50771383/0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c.jpg", extract_findings(text3)],]

DESCRIPTION = "This demo showcases an Image-Text Grounding model that finds the top tokens from the input text that best describe the input image. The model is based on the hierarchical organization theory of cross-modal attention and uses a combination of state-of-the-art image and text encoders."

def create_gradio_interface(model, tokenizer):

    def find_best_token(image, text, k):
        image_tensor = preprocess_image(image)
        text, top_k_tokens, top_k_values = get_top_k_tokens(image_tensor, text, model, tokenizer, k=k)

        # Compute quartile thresholds
        Q1, Q2, Q3 = np.percentile(top_k_values, [25, 50, 75])

        categorized_tokens = {
            "Strongly Disagree": [],
            "Mildly Disagree": [],
            "Neutral": [],
            "Mildly Agree": [],
            "Strongly Agree": []
        }

        print(top_k_values)

        for token, value in zip(top_k_tokens, top_k_values):
            if value < Q1:
                categorized_tokens["Strongly Disagree"].append(token)
            elif Q1 <= value < Q2:
                categorized_tokens["Mildly Disagree"].append(token)
            elif Q2 <= value < Q3:
                categorized_tokens["Neutral"].append(token)
            elif Q3 <= value < 1.0: 
                categorized_tokens["Mildly Agree"].append(token)
            else:  # this will only execute if value == 1.0
                categorized_tokens["Strongly Agree"].append(token)

        return text, categorized_tokens["Strongly Disagree"], categorized_tokens["Mildly Disagree"], categorized_tokens["Neutral"], categorized_tokens["Mildly Agree"], categorized_tokens["Strongly Agree"]

    image_input = gr.inputs.Image(type='filepath')
    text_input = gr.inputs.Textbox(lines=5)
    k_input = gr.inputs.Slider(minimum=1, maximum=20, default=10, label="Top-k tokens")
    output_processed_text = gr.outputs.Textbox(label="Processed Text")
    output_sd = gr.outputs.Textbox(label="Strongly Disagree Word")
    output_md = gr.outputs.Textbox(label="Mildly Disagree Word")
    output_neu = gr.outputs.Textbox(label="Neutral Word")
    output_ma = gr.outputs.Textbox(label="Mildly Agree Word")
    output_sa = gr.outputs.Textbox(label="Strongly Agree Word")

    gr.Interface(fn=find_best_token, 
                 inputs=[image_input, text_input, k_input], 
                 outputs=[output_processed_text, output_sd, output_md, output_neu, output_ma, output_sa],
                 title="Image-Text Grounding Demo", 
                 description=DESCRIPTION,
                 examples=EXAMPLES,
                 # article=article,
                ).launch(server_port=7864, share=True)


def initialize_model(device):
    model = ImageTextGroundingModelHierarchical(256).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    model.load_state_dict(torch.load("best_model_weights_grounding_cv12.pth"))

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = initialize_model(device)

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    create_gradio_interface(model, tokenizer)
    # print(model)

    # image_tensor = preprocess_image(EXAMPLES[1][0])
    # text, top_k_tokens, top_k_values = get_top_k_tokens(image_tensor, EXAMPLES[1][1], model, tokenizer, k=10)

if __name__ == "__main__":
    main()
