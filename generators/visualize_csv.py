import csv
from prettytable import PrettyTable

# Specify the CSV file path
csv_file_path = "/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/data/cleaned_error_correction.csv"

def display_csv(csv_file_path, num_rows=5):
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Use PrettyTable to create a nice tabular display
        table = PrettyTable(field_names=["correct", "incorrect"])
        
        # Iterate over the rows in the CSV file and add them to the table
        for i, row in enumerate(reader):
            if i >= num_rows:
                break
            table.add_row([row["correct"], row["incorrect"]])
        
        # Print the table
        print(table)

display_csv(csv_file_path, num_rows=5)  # Change num_rows as needed
