import os
import json
import numpy as np
import matplotlib.pyplot as plt

def traverse_and_get_data(folder_path):
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                try:
                    json_data = json.load(file)
                    if isinstance(json_data, list) and len(json_data) > 0:
                        # Assuming the structure is [x, 25], and we are interested in [1, 25]
                        for j in range(len(json_data)):
                            data = json_data[j]  # Adjust this if needed
                            all_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {filename}: {e}")

    return all_data

def calculate_average(data):
    if data:
        # Convert the list of lists to a numpy array for easy calculation
        data_array = np.array(data)
        # Calculate the average along axis 0
        average_data = np.mean(data_array, axis=0)
        average_data = average_data * 0.1 + 0.5
        return average_data.tolist()
    else:
        return []

# Replace 'your_folder_path' with the path to your folder containing JSON files
folder_path = './results/temp'

# folder_path = './long'
# folder_path = './results/150'
all_data = traverse_and_get_data(folder_path)
average_data = calculate_average(all_data)
with open('./average_data.json', 'w') as f:
    json.dump(average_data, f)
    print("saved")
plt.plot(average_data)

plt.show()
print(f"Total Average Data: {average_data}")
