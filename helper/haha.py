import pandas as pd

# Load the CSV file
file_path = 'dataset_new/csv_new/annotation_train.csv'
data = pd.read_csv(file_path)

# Condition 1: Rows where valence and arousal are outside the range [-1, 1]
out_of_range_condition = (data['valence'] > 1) | (data['valence'] < -1) | (data['arousal'] > 1) | (data['arousal'] < -1)
out_of_range_count = out_of_range_condition.sum()

# Filter the rows that satisfy condition 1
out_of_range_rows = data[out_of_range_condition]

# Print the first 30 rows that are out of range
print(out_of_range_rows.head(500))

# Condition 2: Rows where valence/arousal are outside the range [-1, 1] and expression is -1
expression_condition = (data['expression'] == -1)
out_of_range_and_expression_condition = out_of_range_condition & expression_condition
out_of_range_and_expression_count = out_of_range_and_expression_condition.sum()

# Condition 3: Rows where valence or arousal equals -5.0
valence_arousal_equals_minus_five = (data['valence'] == -5.0) | (data['arousal'] == -5.0)
valence_arousal_equals_minus_five_count = valence_arousal_equals_minus_five.sum()

# Print the results
print("Total number of rows where valence/arousal is out of range [-1, 1]:", out_of_range_count)
print("Total number of rows where valence/arousal is out of range [-1, 1] and expression is -1:", out_of_range_and_expression_count)
print("Total number of rows where valence or arousal equals -5.0:", valence_arousal_equals_minus_five_count)

