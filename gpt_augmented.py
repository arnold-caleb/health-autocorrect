import csv
import openai
from tqdm import tqdm

from helpers import extract_findings

# function to generate alternative descriptions from the current finds from radiological reports
def generate_alternative_descriptions(text, num_alternatives=1):
    alternatives = []
    openai.api_key = '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'  

    for _ in range(num_alternatives):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Given the medical findings: '{text}', create different versions of errors within the text don't just negate everything",
            temperature=0.8,
            max_tokens=100
        )

        alternatives.append(response.choices[0].text.replace('\n', ' ').strip())

    return alternatives

def main():
    # read existing, old CSV file -- contains original medical reports from mimic-cxr dataset
    with open('locations.csv', 'r') as f:
        reader = csv.reader(f)
        old_data = list(reader)

    # prepare new data
    new_data = []
    row_number = 1

    # iterate through old data, skipping the first row (header)
    for row in tqdm(old_data[1:500], desc="Processing data"):
        _, image_path, text_file_path = row
        text_file_path = text_file_path.strip()  

        with open(text_file_path, 'r') as f:
            original_description = f.read()

        findings = extract_findings(original_description)
        new_descriptions = generate_alternative_descriptions(findings, num_alternatives=10)
        
        new_data.append([row_number, image_path, findings]) # make the findings part of the new dataset
        row_number += 1  

        for new_description in new_descriptions:
            new_data.append([row_number, image_path, new_description])
            row_number += 1  

        # break

    # for data in new_data:
    #     _, _, text = data 
    #     print(text + '\n')

    # write new data into new CSV file
    with open('/proj/vondrick/aa4870/physionet.org/gpt_augmented_dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_number', 'image', 'text'])  # write the header
        writer.writerows(new_data)


if __name__ == "__main__":
    main()
