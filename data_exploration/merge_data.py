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

# Data quality validation (mark as OK or BAD, don't drop)
print("\n--- Data Quality Validation ---")

# Find mammaprint index and type columns
mammaprint_col = None
type_col = None
for col in combined_df.columns:
    if 'mammaprint' in col.lower() and 'index' in col.lower():
        mammaprint_col = col
    if col.lower().strip() == 'typ':
        type_col = col

# Initialize category column - all start as OK
combined_df['data_quality_category'] = 'OK'

# Get the source columns for the main 3 fields
record_num_col = None
for source_col, target_col in mapped_cols.items():
    if target_col == 'record_num':
        record_num_col = source_col

# Check if any of the main 3 columns are None/NaN
missing_main_cols_count = 0
print("Validating main columns for missing values...")

if record_num_col:
    missing_record_num = combined_df[record_num_col].isna()
    combined_df.loc[missing_record_num, 'data_quality_category'] = 'BAD'
    missing_main_cols_count += missing_record_num.sum()
    if missing_record_num.sum() > 0:
        print(f"  Found {missing_record_num.sum()} rows with missing record_num")

if mammaprint_col:
    missing_mammaprint = combined_df[mammaprint_col].isna()
    combined_df.loc[missing_mammaprint, 'data_quality_category'] = 'BAD'
    missing_main_cols_count += missing_mammaprint.sum()
    if missing_mammaprint.sum() > 0:
        print(f"  Found {missing_mammaprint.sum()} rows with missing mammaprint_index")

if type_col:
    missing_type = combined_df[type_col].isna()
    combined_df.loc[missing_type, 'data_quality_category'] = 'BAD'
    missing_main_cols_count += missing_type.sum()
    if missing_type.sum() > 0:
        print(f"  Found {missing_type.sum()} rows with missing type")

print(f"  Total rows with missing main columns: {missing_main_cols_count}")

# Check mammaprint index validity
invalid_mammaprint_count = 0
if mammaprint_col:
    print(f"Validating mammaprint index column: {mammaprint_col}")
    mammaprint_values = pd.to_numeric(combined_df[mammaprint_col], errors='coerce')
    invalid_mammaprint = (mammaprint_values < -1) | (mammaprint_values > 1) | mammaprint_values.isna()
    combined_df.loc[invalid_mammaprint, 'data_quality_category'] = 'BAD'
    invalid_mammaprint_count = invalid_mammaprint.sum()
    print(f"  Found {invalid_mammaprint_count} rows with invalid mammaprint index (not between -1 and 1 or NaN)")

# Check type validity
invalid_type_count = 0
if type_col:
    print(f"Validating type column: {type_col}")
    valid_types = ['Luminal A', 'Luminal B', 'A Luminal', 'B Luminal']
    invalid_type = ~combined_df[type_col].astype(str).str.strip().isin(valid_types) | combined_df[type_col].isna()
    combined_df.loc[invalid_type, 'data_quality_category'] = 'BAD'
    invalid_type_count = invalid_type.sum()
    print(f"  Found {invalid_type_count} rows with invalid type (not Luminal A/B or NaN)")

ok_count = (combined_df['data_quality_category'] == 'OK').sum()
bad_count = (combined_df['data_quality_category'] == 'BAD').sum()
print(f"\nData quality summary:")
print(f"  OK:  {ok_count} rows")
print(f"  BAD: {bad_count} rows")
print(f"  Total: {len(combined_df)} rows")

# Create new dataframe with mapped columns
result_df = pd.DataFrame()

# Map the main columns
for source_col, target_col in mapped_cols.items():
    result_df[target_col] = combined_df[source_col]

# Add category column
result_df['category'] = combined_df['data_quality_category']

# Get all other columns (not mapped and not the category column)
other_cols = [col for col in combined_df.columns if col not in mapped_cols.keys() and col != 'data_quality_category']
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

# Check for conflicting records
print("\n--- Checking for conflicting records ---")
if 'record_num' in result_df.columns:
    # Group by record_num and check if there are different values in main columns
    grouped = result_df.groupby('record_num')

    conflicting_records = []
    for record_num, group in grouped:
        if len(group) > 1:
            # Check if mammaprint_index or type differ
            unique_mammaprint = group['mammaprint_index'].nunique() if 'mammaprint_index' in group.columns else 0
            unique_type = group['type'].nunique() if 'type' in group.columns else 0

            if unique_mammaprint > 1 or unique_type > 1:
                # Found conflicting data for this record_num
                conflicting_records.append(record_num)
                # Mark all rows with this record_num as BAD
                result_df.loc[result_df['record_num'] == record_num, 'category'] = 'BAD'

    if len(conflicting_records) > 0:
        print(f"  Found {len(conflicting_records)} record_nums with conflicting data (different mammaprint_index or type)")
        print(f"  Marked all rows for these records as BAD")
    else:
        print("  No conflicting records found")

# Remove exact duplicates (where all 3 main columns are identical)
print("\n--- Removing exact duplicates ---")
print(f"Rows before removing duplicates: {len(result_df)}")

# Define columns to check for duplicates
main_cols = ['record_num', 'mammaprint_index', 'type']
available_main_cols = [col for col in main_cols if col in result_df.columns]

if available_main_cols:
    result_df_unique = result_df.drop_duplicates(subset=available_main_cols, keep='first')
    removed = len(result_df) - len(result_df_unique)
    print(f"Rows after removing exact duplicates: {len(result_df_unique)}")
    print(f"Removed {removed} exact duplicate rows (same {', '.join(available_main_cols)})")
else:
    result_df_unique = result_df.drop_duplicates()
    print(f"Rows after removing duplicates: {len(result_df_unique)}")

# Save to CSV
output_file = 'data_exploration/merged_data.csv'
result_df_unique.to_csv(output_file, index=False, encoding='utf-8')

# Final statistics
final_ok = (result_df_unique['category'] == 'OK').sum()
final_bad = (result_df_unique['category'] == 'BAD').sum()

print(f"\n✅ Successfully created: {output_file}")
print(f"Final dataset: {len(result_df_unique)} rows, {len(result_df_unique.columns)} columns")
print(f"Columns: {list(result_df_unique.columns)}")
print(f"\nFinal category breakdown:")
print(f"  OK:  {final_ok} rows ({final_ok/len(result_df_unique)*100:.1f}%)")
print(f"  BAD: {final_bad} rows ({final_bad/len(result_df_unique)*100:.1f}%)")
