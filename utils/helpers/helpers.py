import torch
from PIL import Image
from torchvision.transforms import transforms

def remove_extra_whitespace(text):
    import re

    stripped_text = text.strip()
    no_extra_whitespace_text = re.sub('\s+', ' ', stripped_text)
    return no_extra_whitespace_text

def extract_findings(text):
    lines = text.replace('_', '').strip().split("\n")
    findings = False
    findings_lines = []

    for line in lines:
        if "FINDINGS:" in line:
            findings = True
        if findings:
            findings_lines.append(line.strip())

    return " ".join(findings_lines)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return preprocess(image).unsqueeze(0) # add extra dimension to simulate a batch with only one example

def fix_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == 'n_averaged':
            continue  # Skip the unwanted key
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

# this is not needed for now.
def tsne_visualization():
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # visualize tsne
    import numpy as np
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # Load your npy files
    image_embeddings = np.load('image_embeddings.npy')
    text_embeddings = np.load('text_embeddings.npy')

    # Concatenate both embeddings
    all_embeddings = np.concatenate([image_embeddings, text_embeddings], axis=0)

    # Apply t-SNE transformation
    tsne = TSNE(n_components=3, random_state=0)  # Changed to 3 components for 3D
    embeddings_3d = tsne.fit_transform(all_embeddings)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Image embeddings in red
    ax.scatter(embeddings_3d[:len(image_embeddings), 0], embeddings_3d[:len(image_embeddings), 1], 
            embeddings_3d[:len(image_embeddings), 2], color='r', label='Image embeddings')

    # Text embeddings in blue
    ax.scatter(embeddings_3d[len(image_embeddings):, 0], embeddings_3d[len(image_embeddings):, 1], 
            embeddings_3d[len(image_embeddings):, 2], color='b', label='Text embeddings')

    ax.legend()

    # Save the figure
    plt.savefig('tsne_embeddings_3d.png', dpi=300)

    plt.show()
