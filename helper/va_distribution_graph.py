import pandas as pd
import matplotlib.pyplot as plt

# File paths
train_csv_path = "dataset_new/csv_new/annotation_train.csv"
validation_csv_path = "dataset_new/csv_new/annotation_validation.csv"

# Load the train and validation datasets
train_data = pd.read_csv(train_csv_path)
validation_data = pd.read_csv(validation_csv_path)

# Filter out rows where 'expression' is -1
train_data = train_data[train_data['expression'] != -1]
validation_data = validation_data[validation_data['expression'] != -1]

# Filter out rows where 'valence' or 'arousal' are out of range
train_data = train_data[
    (train_data['valence'] >= -1) & (train_data['valence'] <= 1) &
    (train_data['arousal'] >= -1) & (train_data['arousal'] <= 1)
]
validation_data = validation_data[
    (validation_data['valence'] >= -1) & (validation_data['valence'] <= 1) &
    (validation_data['arousal'] >= -1) & (validation_data['arousal'] <= 1)
]

# Extract valence and arousal values
train_valence = train_data['valence']
train_arousal = train_data['arousal']

validation_valence = validation_data['valence']
validation_arousal = validation_data['arousal']

# Filter data within the range [-1, 1]
train_filtered = train_data[(train_valence.between(-1, 1)) & (train_arousal.between(-1, 1))]
validation_filtered = validation_data[(validation_valence.between(-1, 1)) & (validation_arousal.between(-1, 1))]

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(train_filtered['valence'], train_filtered['arousal'], s=10, label='Train', alpha=0.7)
plt.scatter(validation_filtered['valence'], validation_filtered['arousal'], s=10, label='Validation', alpha=0.7)
# plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
# plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.title("Valence-Arousal Scatter Plot")
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.legend()
plt.tight_layout()

# Save the image to a file
output_path = "valence_arousal_scatter_plot_no_grid.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Plot saved to {output_path}")
