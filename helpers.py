import torch
from PIL import Image
from torchvision.transforms import transforms

def extract_findings(text):
        lines = text.strip(' ').replace("\n", "").replace('_', '').strip(' ').split(" ")
        findings = False
        findings_lines = []
        better_finding_lines = []

        for line in lines:
            if "FINDINGS:" in line:
                findings = True
            if findings:
                findings_lines.append(line)

        for finding in findings_lines:
            better_finding_lines.append(finding.strip(' '))

        return " ".join(better_finding_lines)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return preprocess(image).unsqueeze(0)
