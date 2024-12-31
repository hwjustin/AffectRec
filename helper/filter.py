import os
import pandas as pd
from PIL import Image

# Function to filter the dataframe based on image existence
def filter_existing_images(csv_file_path, root_dir, output_csv_file):
    # Load the CSV file into a dataframe
    dataframe = pd.read_csv(csv_file_path, dtype={"filename": str, "index": str})
    print(f"Number of original entries: {len(dataframe)}")

    # List to store valid rows
    valid_entries = []

    # Iterate over each row in the dataframe
    for idx in range(len(dataframe)):
        image_path = os.path.join(
            root_dir,
            f"{dataframe['filename'].iloc[idx]}",
            f"{dataframe['index'].iloc[idx]}.jpg"
        )

        # Check if the image path exists
        if os.path.exists(image_path):
            try:
                # Try opening the image to verify it's valid
                with Image.open(image_path) as img:
                    valid_entries.append(dataframe.iloc[idx])
            except Exception as e:
                print(f"Invalid image file at {image_path}: {e}")

    # Create a new dataframe with valid entries
    filtered_dataframe = pd.DataFrame(valid_entries)

    # Save the filtered dataframe to a new CSV file
    filtered_dataframe.to_csv(output_csv_file, index=False)
    print(f"Filtered CSV saved to {output_csv_file}")
    print(f"Number of valid entries: {len(filtered_dataframe)}")

# Example usage
csv_file_path = "dataset_new/csv/annotations_train_shuffled.csv"  # Path to the input CSV file
root_dir = "dataset_new/cropped_aligned"  # Root directory for images
output_csv_file = "dataset_new/csv/annotations_train_shuffled_filtered.csv"  # Path for the filtered output CSV

filter_existing_images(csv_file_path, root_dir, output_csv_file)