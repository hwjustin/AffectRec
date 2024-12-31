import pandas as pd
import torch

# Path to your CSV file
csv_path = "dataset_new/csv_new/annotation_train.csv"  # Replace with the actual path

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path, dtype={"filename": str, "index": str})

# Check if 'expression' column exists
if "expression" not in df.columns:
    raise ValueError("The 'expression' column is missing in the CSV file!")

# Filter out rows where 'expression' is -1
df_filtered = df[df['expression'] != -1]

# Filter out rows where 'valence' or 'arousal' are out of range
df_filtered = df_filtered[
    (df_filtered['valence'] >= -1) & (df_filtered['valence'] <= 1) &
    (df_filtered['arousal'] >= -1) & (df_filtered['arousal'] <= 1)
]

# Calculate class frequencies
class_counts = df_filtered['expression'].value_counts().sort_index()
total_samples = len(df_filtered)
num_classes = len(class_counts)

# Compute weights inversely proportional to class frequencies
weights = total_samples / (class_counts * num_classes)
weights = torch.tensor(weights.values, dtype=torch.float32)

# Optional: Normalize the weights
weights /= weights.sum()

# Print the calculated weights
print("Class counts per label:\n", class_counts)
print("Calculated class weights:\n", weights)
