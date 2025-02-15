import numpy as np
import json

# Helper function to count missing values
def count_missing_values(data, properties):
    missing_counts = {prop: 0 for prop in properties}
    for entry in data:
        for prop in properties:
            value = entry.get(prop)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                missing_counts[prop] += 1
    return missing_counts

# Helper function to replace missing values
def replace_missing_with_mean(data, properties):
    means = {}
    for prop in properties:
        # Calculates the mean of non-missing values for the property
        prop_values = [entry[prop] for entry in data if entry.get(prop) is not None]
        means[prop] = np.mean(prop_values)

        # Replaces the missing values with the mean
        for entry in data:
            if entry.get(prop) is None or (isinstance(entry[prop], float) and np.isnan(entry[prop])):
                entry[prop] = means[prop]

        print(f"[INFO] Mean for '{prop}': {means[prop]} (used for replacing missing values)")

    return data, means

# Load the datasets
input_files = ["train.json", "valid.json", "test.json"]
datasets = {}

for file_name in input_files:
    with open(file_name, "r") as f:
        datasets[file_name] = json.load(f)

# Properties to process
target_properties = [
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
]

# Replaces missing values in the datasets
for file_name in input_files:
    print(f"\nProcessing {file_name}...")
    data = datasets[file_name]
    print(f"[INFO] Counting missing values...")
    missing_counts = count_missing_values(data, target_properties)
    print(f"[INFO] Missing values before preprocessing: {missing_counts}")
    data, _ = replace_missing_with_mean(data, target_properties)
    datasets[file_name] = data
    print(f"[INFO] Missing values after preprocessing ({file_name}): {count_missing_values(data, target_properties)}")

# Saves the processed datasets
output_files = ["train.json", "valid.json", "test.json"]
for file_name, data in zip(output_files, datasets.values()):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=2)
