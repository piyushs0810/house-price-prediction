import json
from pathlib import Path

import pandas as pd


# Find the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the training dataset from your local computer
DATA_PATH = BASE_DIR / "data" / "train.csv"

# Output file used by the deployed application
OUTPUT_PATH = BASE_DIR / "src" / "reference_values.json"


# Load the dataset
training_data = pd.read_csv(DATA_PATH)

# Remove the target and ID columns
X_reference = training_data.drop(
    columns=["SalePrice", "Id"]
)

# Calculate default values
default_values = {}

for column in X_reference.columns:
    if pd.api.types.is_numeric_dtype(X_reference[column]):
        default_values[column] = float(
            X_reference[column].median()
        )
    else:
        mode_values = X_reference[column].mode(dropna=True)

        if not mode_values.empty:
            default_values[column] = str(mode_values.iloc[0])
        else:
            default_values[column] = None


# Extract options for categorical dropdowns
dropdown_options = {}

for column in X_reference.columns:
    if not pd.api.types.is_numeric_dtype(X_reference[column]):
        dropdown_options[column] = sorted(
            X_reference[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )


# Save the defaults and dropdown options
reference_data = {
    "default_values": default_values,
    "dropdown_options": dropdown_options,
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(reference_data, file, indent=2, ensure_ascii=False)


print(f"Reference file created successfully: {OUTPUT_PATH}")
print(f"Number of default values: {len(default_values)}")
print(f"Number of categorical columns: {len(dropdown_options)}")