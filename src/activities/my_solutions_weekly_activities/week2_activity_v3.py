from pathlib import Path

# This script is located in the project root, so find the path to the current filexs and then go to the parent of that file
project_root = Path(__file__).parent.parent

# Find the .csv file relative to the project root and join to that path the data folder and then the example.csv file
csv_file = project_root.joinpath('data', 'paralympics_raw.csv')
# csv_file = project_root / 'data' / 'example.csv' # this notation would also work, even though you think the '/' is only unix/macOS

# Check if the file exists, this will print 'true' if it exists
print(csv_file.exists())