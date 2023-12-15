import torch 
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor

class DataTransformImageTextDataset(object):
    def __init__(self, transform):
        self.transform_image = transform

    def __call__(self, sample):
        
        image = self.transform_image(sample["image"])
        labels = sample["labels"]
        error_report = sample["incorrect"]
        correct_report = sample["correct"]
        errored_report = sample["errored_report"]
        masked_insertions = sample["masked_error"]

        return image, labels, error_report, correct_report, errored_report, masked_insertions

class TextImageDataset(Dataset):
    def __init__(self, tokenizer, transforms=None):
        self.dataframe = pd.read_csv("/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/data/curriculum.csv", on_bad_lines='skip')
        self.dataframe = self.dataframe[self.dataframe["incorrect"].notna()]
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try: 
            row = self.dataframe.iloc[idx]
            image_path = row['image_path']
            correct_report = row['correct']
            errored_report = row['incorrect']

            image = Image.open(image_path).convert('RGB')

            encoded_report = self.tokenizer.encode_plus(errored_report, padding="do_not_pad", return_tensors="pt")
            tokens = self.tokenizer.convert_ids_to_tokens(encoded_report['input_ids'].squeeze())
            
            token_labels = self.bio_tagging(tokens)
            mask_insertions = self.insert_mask(token_labels)
            tokens, token_labels = self.remove_braces(tokens, token_labels)

            label_dict = {'O': 1, 'B-ERR': 0, 'I-ERR': 0}
            label_ids = [label_dict[label] for label in token_labels]

            cleaned_text = errored_report.replace("{", "").replace("}", "")

            # adjust this for training experiments, choose from the following
            training_tokens = [50, 100, 200]
            num_tokens = 200

            if len(tokens) < num_tokens:
                for i in range(num_tokens - len(tokens)):
                    tokens.append("[PAD]")
                    label_ids.append(1)
            else:
                tokens = tokens[:(num_tokens - 1)]  
                tokens.append("[SEP]")  
                label_ids = label_ids[:num_tokens]

            label_ids = torch.tensor(label_ids)

            return self.transforms({
                "image": image,
                "labels": label_ids,
                "incorrect": cleaned_text,
                "correct": correct_report,
                "errored_report": errored_report,
                "masked_error": mask_insertions
            })
            
        except Exception as e:
            print(f'Error with idx: {idx}, errored_report: {errored_report}')
            raise e

    def bio_tagging(self, tokens):
        in_error = False
        beginning_of_error = False
        token_labels = []

        for token in tokens:
            if token == '{':
                in_error = True
                beginning_of_error = True
                tag = 'O'
                token_labels.append(tag)
                continue
            elif token == '}':
                in_error = False
                tag = 'O'
                token_labels.append(tag)
                continue
            else:
                if in_error:
                    if beginning_of_error:
                        tag = 'B-ERR'
                        beginning_of_error = False
                    else:
                        tag = 'I-ERR'
                else:
                    tag = 'O'
                
                token_labels.append(tag)
        return token_labels

    def remove_braces(self, tokens, labels):
        indices = [i for i, token in enumerate(tokens) if token not in ['{', '}']]
        tokens = [tokens[i] for i in indices]
        labels = [labels[i] for i in indices]
        return tokens, labels

    def insert_mask(self, text):
        modified_text = []
        is_error = False

        for token in text:
            if token == "{" and not is_error:
                is_error = True
                continue
            elif token == "}" and is_error:
                is_error = False
                continue

            if is_error:
                modified_text.append("[ERROR]")
            elif not is_error:
                modified_text.append(token)

            return "".join(modified_text)
