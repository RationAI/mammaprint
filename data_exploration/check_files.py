import pandas as pd
from pathlib import Path
import os

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

print("first 10 files")
files = os.listdir(data_path)
for f in files[:10]:
    print(f)

results = []

for idx, row in merged_df.iterrows():
    record_num = str(row['record_num']).strip()

    # Check if folder exists
    folder_path = data_path / record_num
    folder_exists = folder_path.exists() and folder_path.is_dir()

    # Check if .mrxs file exists
    file_path = data_path / f"{record_num}.mrxs"
    file_exists = file_path.exists() and file_path.is_file()

    # Determine issue
    issue = ""
    if not folder_exists and not file_exists:
        issue = 'Both missing'
    if not folder_exists:
        issue = 'Folder missing'
    elif not file_exists:
        issue = 'File missing'
    else:
        issue = 'OK'

    results.append({
        'record_num': record_num,
        'folder_exists': folder_exists,
        'file_exists': file_exists,
        'folder_path': str(folder_path) if folder_exists else None,
        'file_path': str(file_path) if file_exists else None,
        'issue': issue
    })

    if idx % 10 == 0:
        print(f"Checked {idx + 1}/{len(merged_df)} records...")

print(f"\nCompleted checking {len(results)} records")

results_df = pd.DataFrame(results)

# Summary statistics
total = len(results_df)
ok_count = len(results_df[results_df['issue'] == 'OK'])
both_count = len(results_df[results_df['issue'] == 'Both missing'])
folder_missing = len(results_df[results_df['issue'] == 'Folder missing'])
file_missing = len(results_df[results_df['issue'] == 'File missing'])
empty_records = len(results_df[results_df['issue'] == 'Empty or NaN record_num'])

print("\n--- Summary ---")
print(f"Total records: {total}")
print(f"OK (folder + file exist): {ok_count}")
print(f"Both missing: {both_count}")
print(f"only Folder missing: {folder_missing}")
print(f"only File missing: {file_missing}")
print(f"Empty/NaN record_num: {empty_records}")

output_file = 'data_exploration/file_check_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

issues_df = results_df[results_df['issue'] != 'OK']
if len(issues_df) > 0:
    issues_file = 'data_exploration/file_check_issues.csv'
    issues_df.to_csv(issues_file, index=False)
    print(f"⚠️  Issues only saved to: {issues_file}")
    print(f"\nFound {len(issues_df)} records with issues")
else:
    print("\n✅ All records have corresponding folders and files!")
