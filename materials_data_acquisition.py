from mp_api.client import MPRester
import json
from tqdm import tqdm
import math

# Initialize MPRester with the API key
api_key = "8L0ocLa8xw0ee62xteavyHyhO1dRCNmZ"
mpr = MPRester(api_key)

# Read material IDs from file
with open('material_ids.txt', 'r') as f:
    material_ids = [line.strip() for line in f]

# Total materials and split sizes
total_materials = len(material_ids)
test_size = valid_size = math.ceil(total_materials * 0.1)  # 10% each for test and valid
train_size = total_materials - (test_size + valid_size)

# Fields to fetch
fields = [
    "material_id",
    "structure",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
]

# Container for data
all_materials = []

# Process in chunks of 10000
chunk_size = 10000

try:
    for i in tqdm(range(0, len(material_ids), chunk_size)):
        chunk = material_ids[i:i+chunk_size]

        # Fetch data for each material ID in the chunk
        materials = mpr.materials.summary.search(
            material_ids=chunk,
            fields=fields
        )

        for material in materials:
            try:
                material_entry = {
                    "material_id": material.material_id,
                    "band_gap": material.band_gap,
                    "formation_energy_per_atom": material.formation_energy_per_atom,
                    "energy_above_hull": material.energy_above_hull,
                    "structure": {
                        "lattice": material.structure.lattice.as_dict(),
                        "sites": [site.as_dict() for site in material.structure.sites]
                    }
                }

                all_materials.append(material_entry)
            except Exception as e:
                print(f"Error processing material {material.material_id}: {e}")

except Exception as e:
    print(f"Error fetching materials: {e}")

# Split data into train, validation, and test sets
test_data = all_materials[-test_size:]
valid_data = all_materials[-test_size-valid_size:-test_size]
train_data = all_materials[:train_size]

# Save the splits to JSON files
output_files = {
    "train.json": train_data,
    "valid.json": valid_data,
    "test.json": test_data
}

for file_name, data in output_files.items():
    with open(file_name, "w") as f:
        json.dump(data, f, indent=2)
