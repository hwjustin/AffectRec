import pandas as pd
import torch

csv_path = "dataset_new/csv_new/annotation_train.csv"  # Replace with the actual path


df = pd.read_csv(csv_path, dtype={"filename": str, "index": str})


if "expression" not in df.columns:
    raise ValueError("The 'expression' column is missing in the CSV file!")


df_filtered = df[df['expression'] != -1]

df_filtered = df_filtered[
    (df_filtered['valence'] >= -1) & (df_filtered['valence'] <= 1) &
    (df_filtered['arousal'] >= -1) & (df_filtered['arousal'] <= 1)
]

class_counts = df_filtered['expression'].value_counts().sort_index()
total_samples = len(df_filtered)
num_classes = len(class_counts)


weights = total_samples / (class_counts * num_classes)
weights = torch.tensor(weights.values, dtype=torch.float32)


weights /= weights.sum()

print("Class counts per label:\n", class_counts)
print("Calculated class weights:\n", weights)
