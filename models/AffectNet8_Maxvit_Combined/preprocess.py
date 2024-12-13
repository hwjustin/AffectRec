import os
import csv

# Define the input folders and output file

input_folder = "dataset_new/annotations/VA_Estimation_Challenge/Validation_Set"
expr_train_folder = "dataset_new/annotations/EXPR_Classification_Challenge/Train_Set"
expr_validation_folder = "dataset_new/annotations/EXPR_Classification_Challenge/Validation_Set"
output_file = "dataset_new/csv/annotations_validation.csv"

# Prepare the output data
output_data = []

def find_expression_value(filename, index):
    # Search in both EXPR folders
    for folder in [expr_train_folder, expr_validation_folder]:
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as file:
                lines = file.readlines()
                if index < len(lines):
                    return lines[index].strip()  # Return the expression value if found
    return None

# Iterate through all files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
            # Skip the header and enumerate the rows
            for index, line in enumerate(lines[1:], start=1):
                valence, arousal = line.strip().split(",")
                expr_value = find_expression_value(filename, index)
                if expr_value is not None and expr_value != "-1":
                    output_data.append([filename.replace(".txt", ""), f"{index:05}", valence, arousal, expr_value])

# Write the output data to a CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["filename", "index", "valence", "arousal", "expression"])
    # Write the rows
    writer.writerows(output_data)

print(f"CSV file '{output_file}' has been generated.")


