from pathlib import Path

project_root = Path(__file__).parent.parent  # only two .parent levels

csv_file = project_root / "data" / "paralympics_raw.csv"
excel_file = project_root / "data" / "paralympics_all_raw.xlsx"

print(csv_file.exists(), excel_file.exists())  # should both print True
print(csv_file)  # optional: print the full path so you can confirm visually