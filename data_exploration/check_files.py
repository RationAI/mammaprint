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
print("\nScanning directory for folders...")
all_folders = {folder.name for folder in data_path.iterdir() if folder.is_dir()}
print(f"Found {len(all_folders)} folders in directory")

# Get all record_nums from CSV
csv_records = set(merged_df['record_num'].astype(str).str.strip())
print(f"Found {len(csv_records)} records in CSV")

# Check CSV records against filesystem
print("\n--- Checking CSV records against filesystem ---")
results = []

for idx, row in merged_df.iterrows():
    record_num = str(row['record_num']).strip()

    # Check if folder exists
    folder_path = data_path / record_num
    folder_exists = folder_path.exists() and folder_path.is_dir()

    # Check if .mrxs file exists
    file_path = folder_path / f"{record_num}.mrxs"
    file_exists = file_path.exists() and file_path.is_file()

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
        'folder_exists': folder_exists,
        'file_exists': file_exists,
        'issue': issue
    })

print(f"Completed checking {len(results)} records")

results_df = pd.DataFrame(results)

# Check for folders in directory not in CSV
print("\n--- Checking for folders in directory not in CSV ---")
folders_not_in_csv = all_folders - csv_records
print(f"Found {len(folders_not_in_csv)} folders not in CSV")

# Summary statistics
total = len(results_df)
ok_count = len(results_df[results_df['issue'] == 'OK'])
both_missing = len(results_df[results_df['issue'] == 'Both missing'])
folder_missing = len(results_df[results_df['issue'] == 'Folder missing'])
file_missing = len(results_df[results_df['issue'] == 'File missing'])

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total records in CSV: {total}")
print(f"✅ OK (folder + file exist): {ok_count}")
print(f"⚠️  Both folder and file missing: {both_missing}")
print(f"⚠️  Only folder missing: {folder_missing}")
print(f"⚠️  Only file missing (folder exists): {file_missing}")
print(f"⚠️  Folders in directory not in CSV: {len(folders_not_in_csv)}")

# Print first 10 of each problem set
print("\n" + "="*60)
print("PROBLEM SETS (first 10 of each)")
print("="*60)

# Both missing
both_missing_df = results_df[results_df['issue'] == 'Both missing']
if len(both_missing_df) > 0:
    print(f"\n1. Both folder and file missing ({len(both_missing_df)} total):")
    print(both_missing_df[['record_num', 'issue']].head(10).to_string(index=False))
else:
    print("\n1. Both folder and file missing: None")

# Folder missing
folder_missing_df = results_df[results_df['issue'] == 'Folder missing']
if len(folder_missing_df) > 0:
    print(f"\n2. Folder missing ({len(folder_missing_df)} total):")
    print(folder_missing_df[['record_num', 'issue']].head(10).to_string(index=False))
else:
    print("\n2. Folder missing: None")

# File missing
file_missing_df = results_df[results_df['issue'] == 'File missing']
if len(file_missing_df) > 0:
    print(f"\n3. File missing ({len(file_missing_df)} total):")
    print(file_missing_df[['record_num', 'issue']].head(10).to_string(index=False))
else:
    print("\n3. File missing: None")

# Folders not in CSV
if len(folders_not_in_csv) > 0:
    print(f"\n4. Folders in directory not in CSV ({len(folders_not_in_csv)} total):")
    folders_list = sorted(list(folders_not_in_csv))[:10]
    for folder in folders_list:
        print(f"  - {folder}")
else:
    print("\n4. Folders in directory not in CSV: None")

print("\n" + "="*60)
