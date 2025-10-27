import pandas as pd
from pathlib import Path
from datetime import datetime

# Initialize markdown report
md_report = []

def add_heading(text, level=1):
    md_report.append(f"{'#' * level} {text}\n")

def add_text(text):
    md_report.append(f"{text}\n")

def add_list_item(text, indent=0):
    md_report.append(f"{'  ' * indent}- {text}\n")

def add_code_block(text, lang=""):
    md_report.append(f"```{lang}\n{text}\n```\n")

# Start report
add_heading("Data Comparison Report", 1)
add_text(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Read the source file (check all sheets)
source_file = "data_exploration/Mammaprint-surova data.xlsx"
print(f"Reading {source_file}...")

# Read all sheets from source file
source_excel = pd.ExcelFile(source_file)
print(f"  - Found {len(source_excel.sheet_names)} sheet(s)")

add_heading("Source File", 2)
add_text(f"**File:** `{source_file}`\n")
add_text(f"**Number of sheets:** {len(source_excel.sheet_names)}\n")

source_dfs = []
source_sheet_info = []
for sheet_name in source_excel.sheet_names:
    df = pd.read_excel(source_file, sheet_name=sheet_name)
    print(f"  - Sheet '{sheet_name}': {len(df)} rows")
    source_dfs.append(df)
    source_sheet_info.append((sheet_name, len(df), len(df.columns)))

# Document source sheets
add_heading("Sheets in Source File", 3)
for sheet_name, rows, cols in source_sheet_info:
    add_list_item(f"**{sheet_name}**: {rows} rows, {cols} columns")
add_text("")

# Combine all sheets from source file
source_df = pd.concat(source_dfs, ignore_index=True)
add_text(f"**Total rows (all sheets combined):** {len(source_df)}")
add_text(f"**Total columns:** {len(source_df.columns)}\n")

# Read all data*.xlsx files
data_files = [
    "data_exploration/data_2023-09-25.xlsx",
    "data_exploration/data_test_set_2024-02-16.xlsx",
    "data_exploration/data-dodatek-2023-10-19.xlsx"
]

add_heading("Data Files (data*.xlsx)", 2)

# Combine all data files (check all sheets in each)
combined_dfs = []
data_files_info = []

for file in data_files:
    print(f"Reading {file}...")
    excel_file = pd.ExcelFile(file)
    file_sheets = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file, sheet_name=sheet_name)
        print(f"  - Sheet '{sheet_name}': {len(df)} rows")
        combined_dfs.append(df)
        file_sheets.append((sheet_name, len(df), len(df.columns)))

    data_files_info.append((file, file_sheets))

# Document data files
for file, sheets in data_files_info:
    add_heading(f"`{file}`", 3)
    for sheet_name, rows, cols in sheets:
        add_list_item(f"**{sheet_name}**: {rows} rows, {cols} columns")
    add_text("")

# Concatenate all data files
combined_df = pd.concat(combined_dfs, ignore_index=True)
add_text(f"**Total rows in data*.xlsx files (with duplicates):** {len(combined_df)}\n")

# Remove duplicates from combined data
combined_df_unique = combined_df.drop_duplicates()
add_text(f"**Total unique rows in data*.xlsx files:** {len(combined_df_unique)}\n")

# Find rows in source that are not in combined data
# First, let's check if the columns match
source_cols = set(source_df.columns)
combined_cols = set(combined_df.columns)

# Find common columns to use for comparison
common_cols = list(source_cols & combined_cols)

if common_cols:
    # Compare data using common columns
    print(f"Comparing data using {len(common_cols)} common columns...")

    # Create a key for comparison by combining all columns as strings
    source_keys = source_df[common_cols].astype(str).apply(lambda row: '|||'.join(row.values), axis=1)
    combined_keys = combined_df[common_cols].astype(str).apply(lambda row: '|||'.join(row.values), axis=1)

    # Find rows in source that are not in combined
    missing_from_data = source_df[~source_keys.isin(combined_keys)]
    print(f"Found {len(missing_from_data)} rows in source that are NOT in data files")

    # Find rows in data files that are not in source
    missing_from_source = combined_df[~combined_keys.isin(source_keys)]
    print(f"Found {len(missing_from_source)} rows in data files that are NOT in source")

    # Also check for duplicates within source
    source_duplicates = source_df[source_df.duplicated(subset=common_cols, keep=False)]
    print(f"Found {len(source_duplicates)} duplicate rows in source")

    # Data quality checks
    add_heading("Data Quality Issues", 2)

    # Find mammaprint index column (case-insensitive)
    mammaprint_col = None
    type_col = None
    for col in combined_df.columns:
        if 'mammaprint' in col.lower() and 'index' in col.lower():
            mammaprint_col = col
        if col.lower().strip() == 'typ':
            type_col = col

    invalid_rows = []

    # Check mammaprint index values
    if mammaprint_col:
        print(f"Checking mammaprint index column: {mammaprint_col}")
        # Convert to numeric, coercing errors to NaN
        mammaprint_values = pd.to_numeric(combined_df[mammaprint_col], errors='coerce')
        invalid_mammaprint = combined_df[
            (mammaprint_values < -1) | (mammaprint_values > 1) | mammaprint_values.isna()
        ]
        invalid_rows.append(('mammaprint_index', invalid_mammaprint))
        print(f"  Invalid mammaprint index values: {len(invalid_mammaprint)}")

        add_heading("Invalid Mammaprint Index Values", 3)
        add_text(f"**Expected:** Values between -1 and 1")
        add_text(f"**Found:** {len(invalid_mammaprint)} invalid rows\n")

        if len(invalid_mammaprint) > 0:
            if len(invalid_mammaprint) <= 10:
                add_text("All invalid rows:\n")
                add_code_block(invalid_mammaprint[[mammaprint_col]].to_string(index=False))
            else:
                add_text("Preview (first 10 rows):\n")
                add_code_block(invalid_mammaprint[[mammaprint_col]].head(10).to_string(index=False))
        add_text("")

    # Check type values
    if type_col:
        print(f"Checking type column: {type_col}")
        valid_types = ['Luminal A', 'Luminal B', 'A Luminal', 'B Luminal']
        invalid_type = combined_df[
            ~combined_df[type_col].astype(str).str.strip().isin(valid_types) &
            combined_df[type_col].notna()
        ]
        invalid_rows.append(('type', invalid_type))
        print(f"  Invalid type values: {len(invalid_type)}")

        add_heading("Invalid Type Values", 3)
        add_text(f"**Expected:** 'Luminal A', 'Luminal B', 'A Luminal', or 'B Luminal'")
        add_text(f"**Found:** {len(invalid_type)} invalid rows\n")

        if len(invalid_type) > 0:
            unique_invalid_types = invalid_type[type_col].unique()
            add_text(f"Invalid type values found: {list(unique_invalid_types)}\n")

            if len(invalid_type) <= 10:
                add_text("All invalid rows:\n")
                add_code_block(invalid_type[[type_col]].to_string(index=False))
            else:
                add_text("Preview (first 10 rows):\n")
                add_code_block(invalid_type[[type_col]].head(10).to_string(index=False))
        add_text("")

    # Add comparison results
    add_heading("Comparison Results", 2)

    # Rows in source but not in data files
    add_heading("Rows in Source NOT in data*.xlsx", 3)
    if len(missing_from_data) == 0:
        add_text("✅ **All rows from the source file are contained in data*.xlsx files!**\n")
    else:
        add_text(f"⚠️ **Found {len(missing_from_data)} rows**\n")
        add_text("These missing rows have been exported to `source_not_in_data.csv` for detailed review.\n")

        # Show preview of missing rows
        if len(missing_from_data) <= 10:
            add_text("All missing rows:\n")
            add_code_block(missing_from_data.to_string(index=False))
        else:
            add_text("Preview (first 10 rows):\n")
            add_code_block(missing_from_data.head(10).to_string(index=False))

    # Rows in data files but not in source
    add_heading("Rows in data*.xlsx NOT in Source", 3)
    if len(missing_from_source) == 0:
        add_text("✅ **All rows from data*.xlsx files are contained in the source file!**\n")
    else:
        add_text(f"⚠️ **Found {len(missing_from_source)} rows**\n")
        add_text("These extra rows have been exported to `data_not_in_source.csv` for detailed review.\n")

        # Show preview of extra rows
        if len(missing_from_source) <= 10:
            add_text("All extra rows:\n")
            add_code_block(missing_from_source.to_string(index=False))
        else:
            add_text("Preview (first 10 rows):\n")
            add_code_block(missing_from_source.head(10).to_string(index=False))

    # Duplicates
    if len(source_duplicates) > 0:
        add_heading("Duplicate Rows in Source", 2)
        add_text(f"Found **{len(source_duplicates)} duplicate rows** in the source file.\n")

        if len(source_duplicates) <= 20:
            add_heading("All Duplicate Rows", 3)
            add_code_block(source_duplicates.to_string(index=False))
        else:
            add_heading("Duplicate Rows Preview (first 20)", 3)
            add_code_block(source_duplicates.head(20).to_string(index=False))
            add_text(f"\n_Showing 20 of {len(source_duplicates)} duplicate rows_\n")

    # Write markdown report
    with open('data_exploration/data_diff_report.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(md_report))

    # Export missing/extra rows to CSV if any exist
    if len(missing_from_data) > 0:
        missing_from_data.to_csv('data_exploration/source_not_in_data.csv', index=False)
        print("Exported: source_not_in_data.csv")

    if len(missing_from_source) > 0:
        missing_from_source.to_csv('data_exploration/data_not_in_source.csv', index=False)
        print("Exported: data_not_in_source.csv")

    print("Report written to: data_diff_report.md")

else:
    add_text("❌ **ERROR: No common columns found between source and data files!**\n")

    # Write markdown report
    with open('data_exploration/data_diff_report.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(md_report))

    print("ERROR: No common columns found!")
    print("Report written to: data_diff_report.md")
