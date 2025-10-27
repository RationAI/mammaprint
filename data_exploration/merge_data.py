import pandas as pd
from pathlib import Path

# Read all data*.xlsx files
data_files = [
    "data_exploration/data_2023-09-25.xlsx",
    "data_exploration/data_test_set_2024-02-16.xlsx",
    "data_exploration/data-dodatek-2023-10-19.xlsx",
    "data_exploration/Mammaprint-surova data.xlsx"
]

print("Reading data files...")

# Combine all data files (check all sheets in each)
all_dfs = []

for file in data_files:
    print(f"Reading {file}...")
    excel_file = pd.ExcelFile(file)

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file, sheet_name=sheet_name)
        print(f"  - Sheet '{sheet_name}': {len(df)} rows")
        all_dfs.append(df)

# Concatenate all data
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal rows combined: {len(combined_df)}")
print(f"Columns: {list(combined_df.columns)}")

# Define column mapping
column_mapping = {
    'Číslo\nzáznamu': 'record_num',
    'mammaprint index': 'mammaprint_index',
    'Typ': 'type'
}

# Find the actual column names (case-insensitive matching)
mapped_cols = {}
for col in combined_df.columns:
    col_lower = col.lower().strip()
    for source_name, target_name in column_mapping.items():
        if col_lower == source_name.lower():
            mapped_cols[col] = target_name
            break

print(f"\nMapped columns: {mapped_cols}")

# Data quality filtering
print("\n--- Data Quality Filtering ---")
original_count = len(combined_df)

# Find mammaprint index and type columns
mammaprint_col = None
type_col = None
for col in combined_df.columns:
    if 'mammaprint' in col.lower() and 'index' in col.lower():
        mammaprint_col = col
    if col.lower().strip() == 'typ':
        type_col = col

# Filter invalid mammaprint index values
if mammaprint_col:
    print(f"Filtering mammaprint index column: {mammaprint_col}")
    mammaprint_values = pd.to_numeric(combined_df[mammaprint_col], errors='coerce')
    valid_mammaprint = (mammaprint_values >= -1) & (mammaprint_values <= 1) & mammaprint_values.notna()
    before = len(combined_df)
    combined_df = combined_df[valid_mammaprint]
    dropped = before - len(combined_df)
    print(f"  Dropped {dropped} rows with invalid mammaprint index (not between -1 and 1)")

# Filter invalid type values
if type_col:
    print(f"Filtering type column: {type_col}")
    valid_types = ['Luminal A', 'Luminal B', 'A Luminal', 'B Luminal']
    valid_type = combined_df[type_col].astype(str).str.strip().isin(valid_types)
    before = len(combined_df)
    combined_df = combined_df[valid_type]
    dropped = before - len(combined_df)
    print(f"  Dropped {dropped} rows with invalid type (not Luminal A/B)")

print(f"\nTotal rows after filtering: {len(combined_df)} (dropped {original_count - len(combined_df)} rows)")

# Create new dataframe with mapped columns
result_df = pd.DataFrame()

# Map the main columns
for source_col, target_col in mapped_cols.items():
    result_df[target_col] = combined_df[source_col]

# Get all other columns (not mapped)
other_cols = [col for col in combined_df.columns if col not in mapped_cols.keys()]
print(f"\nColumns to merge into 'note': {other_cols}")

# Merge other columns into 'note' column
def merge_columns_to_note(row):
    notes = []
    for col in other_cols:
        value = row[col]
        # Skip NaN/empty values
        if pd.notna(value) and str(value).strip() != '':
            notes.append(f"{col}: {value}")
    return "; ".join(notes) if notes else ""

print("\nMerging other columns into 'note' field...")
result_df['note'] = combined_df.apply(merge_columns_to_note, axis=1)

# Remove duplicates
print(f"\nRows before removing duplicates: {len(result_df)}")
result_df_unique = result_df.drop_duplicates()
print(f"Rows after removing duplicates: {len(result_df_unique)}")

# Save to CSV
output_file = 'data_exploration/merged_data.csv'
result_df_unique.to_csv(output_file, index=False, encoding='utf-8')

print(f"\n✅ Successfully created: {output_file}")
print(f"Final dataset: {len(result_df_unique)} rows, {len(result_df_unique.columns)} columns")
print(f"Columns: {list(result_df_unique.columns)}")
