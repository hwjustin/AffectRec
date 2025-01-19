import pandas as pd

# Define the input and output file paths
input_file = "dataset_new/annotations/MTL_Challenge/validation_set.txt"  # Replace with the path to your text file
output_file = "dataset_new/csv_new/annotation_validation.csv"

# Open and read the input file
with open(input_file, "r") as file:
    lines = file.readlines()

# Split lines into headers and data
header = lines[0].strip().split(",")[:4]  # Extract the header row
data = [line.strip().split(",")[:4] for line in lines[1:]]  # Extract the data rows

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=header)

# Select only the desired columns: image, valence, arousal, expression
df_filtered = df[["image", "valence", "arousal", "expression"]]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv(output_file, index=False)

print(f"CSV file '{output_file}' has been created with selected headers.")
