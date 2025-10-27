import pandas as pd
from pathlib import Path

DATA_PATH = "/mnt/data/Projects/Data/MOU/breast/mammaprint"

# Read the merged data
merged_data_file = "data_exploration/merged_data.csv"
print(f"Reading merged data from: {merged_data_file}")

merged_df = pd.read_csv(merged_data_file)

print(f"Loaded {len(merged_df)} records")
print(f"Columns: {list(merged_df.columns)}")

data_path = Path(DATA_PATH)
print(f"\nChecking files in: {data_path}")

if not data_path.exists():
    print(f"WARNING: Path does not exist: {data_path}")
    print("Please update the DATA_PATH variable in the script")
    exit(1)

# Get all folders in the data directory
print("\nScanning directory for folders and files...")
all_folders = {folder.name for folder in data_path.iterdir() if folder.is_dir()}
print(f"Found {len(all_folders)} folders in directory")

# Get all .mrxs files in the data directory
all_files = {file.stem for file in data_path.iterdir() if file.is_file() and file.suffix == '.mrxs'}
print(f"Found {len(all_files)} .mrxs files in directory")

# Get all record_nums from CSV
csv_records = set(merged_df['record_num'].astype(str).str.strip())
print(f"Found {len(csv_records)} records in CSV")

# Function to generate possible folder name variations
def get_folder_variations(record_num):
    """Generate possible folder name variations for a record number."""
    variations = [record_num]  # Start with exact match

    # Handle formats like "2023/835" -> "P2023_00835", "P2023_835", "2023_835", etc.
    if '/' in record_num:
        parts = record_num.split('/')
        if len(parts) == 2:
            year, num = parts
            # Check if num has suffix like "-1"
            num_base = num
            suffix = ""
            if '-' in num:
                num_parts = num.split('-', 1)
                num_base = num_parts[0]
                suffix = '-' + num_parts[1]

            # Try with P prefix and underscore
            variations.append(f"P{year}_{num_base}{suffix}")
            variations.append(f"P{year}_{num_base.zfill(5)}{suffix}")  # Zero-padded to 5 digits
            variations.append(f"P{year}_{num_base.zfill(4)}{suffix}")  # Zero-padded to 4 digits
    
    return variations


# Function to generate possible file name variations
def get_file_variations(record_num):
    """Generate possible .mrxs file name variations."""
    return [f"{x}.mrxs" for x in get_folder_variations(record_num)]

# Function to find matching folder
def find_matching_folder(record_num, all_folders):
    """Try to find a matching folder using various naming conventions."""
    variations = get_folder_variations(record_num)

    for var in variations:
        if var in all_folders:
            return var, True

    return None, False

# Function to find matching files (can be multiple with suffixes like -1, -2)
def find_matching_files(record_num, all_files):
    """Try to find all matching .mrxs files using various naming conventions.
    Returns list of matched file names and whether any were found."""
    variations = get_folder_variations(record_num)  # Use same variations as folders
    matched = []

    # First check for exact matches
    for var in variations:
        if var in all_files:
            matched.append(var)

    # Also check for files with suffixes (e.g., record-1, record-2)
    for var in variations:
        for file in all_files:
            # Check if file starts with the variation and has a suffix
            if file.startswith(var + '-') and file not in matched:
                matched.append(file)

    return matched, len(matched) > 0

# Check CSV records against filesystem
print("\n--- Checking CSV records against filesystem ---")
results = []
matched_folders = set()
matched_files = set()

for idx, row in merged_df.iterrows():
    record_num = str(row['record_num']).strip()

    # Try to find matching folder with variations
    matched_folder_name, folder_exists = find_matching_folder(record_num, all_folders)
    if folder_exists:
        matched_folders.add(matched_folder_name)

    # Try to find matching .mrxs files with variations (can be multiple)
    matched_file_list, file_exists = find_matching_files(record_num, all_files)
    if file_exists:
        # Add all matched files to the set
        for f in matched_file_list:
            matched_files.add(f)

    # Determine issue
    if not folder_exists and not file_exists:
        issue = 'Both missing'
    elif not folder_exists:
        issue = 'Folder missing'
    elif not file_exists:
        issue = 'File missing'
    else:
        issue = 'OK'

    results.append({
        'record_num': record_num,
        'matched_folder': matched_folder_name if folder_exists else None,
        'matched_files': matched_file_list if file_exists else [],
        'file_count': len(matched_file_list),
        'folder_exists': folder_exists,
        'file_exists': file_exists,
        'issue': issue
    })

print(f"Completed checking {len(results)} records")

results_df = pd.DataFrame(results)

# Check for folders in directory not in CSV (excluding already matched folders)
print("\n--- Checking for folders in directory not in CSV ---")
unmatched_folders = all_folders - matched_folders
print(f"Found {len(unmatched_folders)} unmatched folders")

# Try reverse matching: for each unmatched folder, see if it could match a CSV record
reverse_matched = []
truly_unmatched_folders = []

for folder in unmatched_folders:
    found_match = False
    # Generate reverse variations (e.g., "P2023_00835" -> "2023/835")
    if folder.startswith('P') and '_' in folder:
        # P2023_00835 -> 2023/835 or P2023_00835-1 -> 2023/835-1
        parts = folder[1:].split('_', 1)  # Remove P and split on first underscore
        if len(parts) == 2:
            year = parts[0]
            num_part = parts[1]
            # Handle dashes
            if '-' in num_part:
                num_parts = num_part.split('-', 1)
                num = num_parts[0].lstrip('0') or '0'
                suffix = '-' + num_parts[1]
                possible_csv = f"{year}/{num}{suffix}"
            else:
                num = num_part.lstrip('0') or '0'
                possible_csv = f"{year}/{num}"

            if possible_csv in csv_records:
                reverse_matched.append((folder, possible_csv))
                found_match = True

    if not found_match and '_' in folder and not folder.startswith('P'):
        # 2023_00835 -> 2023/835 or 2024_02851-1 -> 2024/2851-1
        parts = folder.split('_', 1)
        if len(parts) == 2:
            year = parts[0]
            num_part = parts[1]
            # Handle dashes
            if '-' in num_part:
                num_parts = num_part.split('-', 1)
                num = num_parts[0].lstrip('0') or '0'
                suffix = '-' + num_parts[1]
                possible_csv = f"{year}/{num}{suffix}"
            else:
                num = num_part.lstrip('0') or '0'
                possible_csv = f"{year}/{num}"

            if possible_csv in csv_records:
                reverse_matched.append((folder, possible_csv))
                found_match = True

    if not found_match:
        truly_unmatched_folders.append(folder)

if len(reverse_matched) > 0:
    print(f"Found {len(reverse_matched)} folders that match CSV records (but were using different naming)")
    print("These folders should have been matched in the forward pass - investigating...")

folders_not_in_csv = truly_unmatched_folders
print(f"Truly unmatched folders: {len(folders_not_in_csv)}")

# Check for files in directory not in CSV (excluding already matched files)
print("\n--- Checking for files in directory not in CSV ---")
unmatched_files = all_files - matched_files
print(f"Found {len(unmatched_files)} unmatched files")

# Try reverse matching: for each unmatched file, see if it could match a CSV record
reverse_matched_files = []
truly_unmatched_files = []

for file in unmatched_files:
    found_match = False
    # Generate reverse variations (e.g., "P2023_00835" -> "2023/835")
    if file.startswith('P') and '_' in file:
        # P2023_00835 -> 2023/835 or P2023_00835-1 -> 2023/835-1
        parts = file[1:].split('_', 1)  # Remove P and split on first underscore
        if len(parts) == 2:
            year = parts[0]
            num_part = parts[1]
            # Handle dashes
            if '-' in num_part:
                num_parts = num_part.split('-', 1)
                num = num_parts[0].lstrip('0') or '0'
                suffix = '-' + num_parts[1]
                possible_csv = f"{year}/{num}{suffix}"
            else:
                num = num_part.lstrip('0') or '0'
                possible_csv = f"{year}/{num}"

            if possible_csv in csv_records:
                reverse_matched_files.append((file, possible_csv))
                found_match = True

    if not found_match and '_' in file and not file.startswith('P'):
        # 2023_00835 -> 2023/835 or 2024_02851-1 -> 2024/2851-1
        parts = file.split('_', 1)
        if len(parts) == 2:
            year = parts[0]
            num_part = parts[1]
            # Handle dashes
            if '-' in num_part:
                num_parts = num_part.split('-', 1)
                num = num_parts[0].lstrip('0') or '0'
                suffix = '-' + num_parts[1]
                possible_csv = f"{year}/{num}{suffix}"
            else:
                num = num_part.lstrip('0') or '0'
                possible_csv = f"{year}/{num}"

            if possible_csv in csv_records:
                reverse_matched_files.append((file, possible_csv))
                found_match = True

    if not found_match:
        truly_unmatched_files.append(file)

if len(reverse_matched_files) > 0:
    print(f"Found {len(reverse_matched_files)} files that match CSV records (but were using different naming)")
    print("These files should have been matched in the forward pass - investigating...")

files_not_in_csv = truly_unmatched_files
print(f"Truly unmatched files: {len(files_not_in_csv)}")

# Summary statistics
total = len(results_df)
ok_count = len(results_df[results_df['issue'] == 'OK'])
both_missing = len(results_df[results_df['issue'] == 'Both missing'])
folder_missing = len(results_df[results_df['issue'] == 'Folder missing'])
file_missing = len(results_df[results_df['issue'] == 'File missing'])
multiple_files_count = len(results_df[results_df['file_count'] > 1])
total_file_count = results_df['file_count'].sum()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total records in CSV: {total}")
print(f"✅ OK (folder + file exist): {ok_count}")
print(f"⚠️  Both folder and file missing: {both_missing}")
print(f"⚠️  Only folder missing: {folder_missing}")
print(f"⚠️  Only file missing (folder exists): {file_missing}")
print(f"ℹ️  Records with multiple files: {multiple_files_count}")
print(f"ℹ️  Total .mrxs files found: {total_file_count}")
print(f"⚠️  Folders in directory not in CSV: {len(folders_not_in_csv)}")
print(f"⚠️  Files in directory not in CSV: {len(files_not_in_csv)}")

# Print first 10 of each problem set
print("\n" + "="*60)
print("PROBLEM SETS (first 10 of each)")
print("="*60)

# Both missing
both_missing_df = results_df[results_df['issue'] == 'Both missing']
if len(both_missing_df) > 0:
    print(f"\n1. Both folder and file missing ({len(both_missing_df)} total):")
    print(both_missing_df[['record_num']].head(10).to_string(index=False))
else:
    print("\n1. Both folder and file missing: None")

# Folder missing
folder_missing_df = results_df[results_df['issue'] == 'Folder missing']
if len(folder_missing_df) > 0:
    print(f"\n2. Folder missing ({len(folder_missing_df)} total):")
    print(folder_missing_df[['record_num']].head(10).to_string(index=False))
else:
    print("\n2. Folder missing: None")

# File missing
file_missing_df = results_df[results_df['issue'] == 'File missing']
if len(file_missing_df) > 0:
    print(f"\n3. File missing ({len(file_missing_df)} total):")
    print(file_missing_df[['record_num', 'matched_folder']].head(10).to_string(index=False))
else:
    print("\n3. File missing: None")

# Records with multiple files
multiple_files_df = results_df[results_df['file_count'] > 1]
if len(multiple_files_df) > 0:
    print(f"\n4. Records with multiple files ({len(multiple_files_df)} total) - sample:")
    for idx, row in multiple_files_df.head(10).iterrows():
        files_str = ', '.join(row['matched_files'])
        print(f"  {row['record_num']}: {row['file_count']} files ({files_str})")
else:
    print("\n4. Records with multiple files: None")

# Successfully matched with name variations
matched_folder_variations = results_df[(results_df['issue'] == 'OK') & (results_df['matched_folder'].notna()) & (results_df['record_num'] != results_df['matched_folder'])]

if len(matched_folder_variations) > 0:
    print(f"\n5. Successfully matched folders with name variations ({len(matched_folder_variations)} total) - sample:")
    print(matched_folder_variations[['record_num', 'matched_folder']].head(10).to_string(index=False))

# Folders not in CSV
if len(folders_not_in_csv) > 0:
    print(f"\n6. Folders in directory not in CSV ({len(folders_not_in_csv)} total):")
    folders_list = sorted(list(folders_not_in_csv))[:10]
    for folder in folders_list:
        print(f"  - {folder}")
else:
    print("\n6. Folders in directory not in CSV: None")

# Files not in CSV
if len(files_not_in_csv) > 0:
    print(f"\n7. Files in directory not in CSV ({len(files_not_in_csv)} total):")
    files_list = sorted(list(files_not_in_csv))[:10]
    for file in files_list:
        print(f"  - {file}.mrxs")
else:
    print("\n7. Files in directory not in CSV: None")

print("\n" + "="*60)
