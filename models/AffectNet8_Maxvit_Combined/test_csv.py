# Just a quick file for finding csv value
import pandas as pd

def find_value_in_column(file_path, column_name, value_to_find):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Check if the specified column exists
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in the dataset.")
            return None

        # Find rows where the column matches the specified value
        result = df[df[column_name] == value_to_find]

        if result.empty:
            print(f"No rows found with {column_name} = '{value_to_find}'.")
        else:
            print(f"Rows with {column_name} = '{value_to_find}':")
            print(result)

        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
# Specify the path to your CSV file
csv_file_path = 'dataset_new/csv/annotations.csv'
column_to_search = 'filename'
value_to_search = '4-30-1920x1080'

# Call the function
found_rows = find_value_in_column(csv_file_path, column_to_search, value_to_search)