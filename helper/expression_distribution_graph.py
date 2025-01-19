import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
train_file = "dataset_new/csv_new/annotation_train.csv"
validation_file = "dataset_new/csv_new/annotation_validation.csv"

# Emotion mapping
emotion_mapping = {
    0: "Neutral",
    1: "Anger",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Sadness",
    6: "Surprise",
    7: "Other",
}

# Read data
train_data = pd.read_csv(train_file)
validation_data = pd.read_csv(validation_file)



train_data = train_data[train_data['expression'] != -1]
validation_data = validation_data[validation_data['expression'] != -1]


train_data = train_data[
    (train_data['valence'] >= -1) & (train_data['valence'] <= 1) &
    (train_data['arousal'] >= -1) & (train_data['arousal'] <= 1)
]
validation_data = validation_data[
    (validation_data['valence'] >= -1) & (validation_data['valence'] <= 1) &
    (validation_data['arousal'] >= -1) & (validation_data['arousal'] <= 1)
]


train_data["emotion"] = train_data["expression"].map(emotion_mapping)
validation_data["emotion"] = validation_data["expression"].map(emotion_mapping)


train_data = train_data.dropna(subset=["emotion"])
validation_data = validation_data.dropna(subset=["emotion"])


train_freq = train_data["emotion"].value_counts(normalize=True) * 100
validation_freq = validation_data["emotion"].value_counts(normalize=True) * 100

all_emotions = list(emotion_mapping.values())

freq_df = pd.DataFrame(index=all_emotions)
freq_df["Train"] = train_freq.reindex(all_emotions, fill_value=0)
freq_df["Validation"] = validation_freq.reindex(all_emotions, fill_value=0)

bar_width = 0.35
index = np.arange(len(freq_df.index))

plt.figure(figsize=(10, 6))
plt.bar(index, freq_df["Train"], bar_width, label="Train", alpha=0.8)
plt.bar(index + bar_width, freq_df["Validation"], bar_width, label="Validation", alpha=0.8)

plt.ylabel("Relative Frequency [%]")
plt.title("Frequency of Expression")
plt.xticks(index + bar_width / 2, freq_df.index, rotation=45)
plt.yticks(np.arange(0, 45, 5)) 
plt.ylim(0, 40) 
plt.legend(title="Dataset")
plt.tight_layout()


output_path = "expression_frequency.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved as {output_path}")
