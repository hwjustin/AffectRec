import pandas as pd
import os

def shuffle_csv():
    """
    Shuffle rows of a CSV file and save to a new file.
    """
    input_file = "dataset_new/csv/annotations_train.csv"
    output_file = "dataset_new/csv/annotations_train_shuffled.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return
    
    # Read the CSV file
    df = pd.read_csv(input_file, dtype={"filename": str, "index": str})
    
    # Shuffle the DataFrame
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    
    # Save the shuffled DataFrame to a new file
    shuffled_df.to_csv(output_file, index=False)
    print(f"Shuffled CSV saved to {output_file}")

if __name__ == "__main__":
    shuffle_csv()
