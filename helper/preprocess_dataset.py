import pandas as pd


input_file = "dataset_new/annotations/MTL_Challenge/validation_set.txt"  
output_file = "dataset_new/csv_new/annotation_validation.csv"


with open(input_file, "r") as file:
    lines = file.readlines()

header = lines[0].strip().split(",")[:4] 
data = [line.strip().split(",")[:4] for line in lines[1:]]

df = pd.DataFrame(data, columns=header)

df_filtered = df[["image", "valence", "arousal", "expression"]]

df_filtered.to_csv(output_file, index=False)

print(f"CSV file '{output_file}' has been created with selected headers.")
