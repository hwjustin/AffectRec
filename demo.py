import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

# Specify the path to your CSV file
file_path = 'dataset/filelists/training.csv'

# Read the first line
with open(file_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    first_line = next(reader)  # Get the first line
    print(first_line)
    row_data = next(reader)  # Get the first line
    print(row_data)


# Load the image
image_path = row_data[0]
image = mpimg.imread(f"dataset/images/Manually_Annotated_Images/{image_path}")

# Define the bounding box coordinates
face_x = int(row_data[1])       # Example x-coordinate
face_y = int(row_data[2])         # Example y-coordinate
face_width = int(row_data[3])     # Example width
face_height = int(row_data[4])      # Example height

# Your input string with coordinates
data_str = row_data[5]

# Remove the single quotes and split the string into a list of numbers
data_str = data_str.strip("'")
coordinates = list(map(float, data_str.split(';')))

# Separate the x and y coordinates
x_coords = coordinates[0::2]
y_coords = coordinates[1::2]

# Create a figure and axis
plt.figure(figsize=(10, 6))
plt.imshow(image)  # Show the image
plt.scatter(x_coords, y_coords, c='red', marker='o')  # Overlay the points
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.grid(False)  # Disable grid for image plot
plt.axis('off')  # Turn off the axis

# Save the modified image to a new file
output_path = 'output_image_with_facial_landmarks.jpg'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


