"""
COMP0035 Week 2: Pandas Workshop Activities
--------------------------------------------
This script follows the same structure as the official workshop tutorial.

Learning objectives:
1. Read data from CSV and Excel files using pandas
2. Inspect the data (structure, types, missing values)
3. Clean and transform data (e.g. fix text, handle missing values)
4. Filter and sort data
5. Summarise data using grouping
6. Create simple charts
7. Export cleaned data

Make sure you’ve installed pandas, openpyxl, and matplotlib in your virtual
pip install pandas openpyxl matplotlib
"""

# ---------------------------------------------------------------------
# Activity 1: Import libraries
# ---------------------------------------------------------------------
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Activity 2: Define data file paths
# ---------------------------------------------------------------------
# Path(__file__) gives you the current Python file location.
# .parent goes up one directory level. We go up to find 'data/'.
project_root = Path(__file__).parent.parent  # points to src/activities
data_folder = project_root / "data"

csv_path = data_folder / "paralympics_raw.csv"
excel_path = data_folder / "paralympics_all_raw.xlsx"

print("✅ Checking data file paths:")
print("CSV file:", csv_path)
print("Excel file:", excel_path)
print("Files exist?", csv_path.exists(), excel_path.exists())
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 3: Load CSV and Excel data into DataFrames
# ---------------------------------------------------------------------
# The CSV contains event-level data (year, participants, etc.)
# The Excel has additional sheets for more details.
events_df = pd.read_csv(csv_path)
all_data_df = pd.read_excel(excel_path, sheet_name=0)
medals_df = pd.read_excel(excel_path, sheet_name=1)

print("✅ Data loaded successfully!")
print("CSV shape:", events_df.shape)
print("Excel sheets shapes:", all_data_df.shape, medals_df.shape)
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 4: Inspect the first few rows of the CSV data
# ---------------------------------------------------------------------
print("🔹 First 5 rows of events_df:")
print(events_df.head())
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 5: View column names and data types
# ---------------------------------------------------------------------
print("🔹 Columns in the DataFrame:")
print(events_df.columns.tolist())

print("\n🔹 Data types of each column:")
print(events_df.dtypes)
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 6: Basic information and summary statistics
# ---------------------------------------------------------------------
print("🔹 DataFrame info summary:")
print(events_df.info())

print("\n🔹 Summary statistics for all columns:")
print(events_df.describe(include='all'))
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 7: Find missing values
# ---------------------------------------------------------------------
print("🔹 Missing values per column:")
print(events_df.isna().sum())

# Identify rows that contain any missing value
missing_rows = events_df[events_df.isna().any(axis=1)]
print("\nRows containing missing values:")
print(missing_rows)
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 8: Fill or remove missing values
# ---------------------------------------------------------------------
# Option 1: fill numeric NaNs with 0
events_df = events_df.fillna(0)
print("✅ Missing numeric values replaced with 0.")
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 9: Explore and clean categorical data
# ---------------------------------------------------------------------
# Example column: 'type' (Summer/Winter)
print("🔹 Unique values before cleaning:")
print(events_df['type'].unique())

# Make text lowercase and strip extra spaces
events_df['type'] = events_df['type'].astype(str).str.strip().str.lower()

print("🔹 Unique values after cleaning:")
print(events_df['type'].unique())
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 10: Filter data (e.g., only Summer Games)
# ---------------------------------------------------------------------
summer_df = events_df[events_df['type'] == 'summer']
print(f"✅ Filtered summer events ({len(summer_df)} rows):")
print(summer_df.head(3))
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 11: Sort data by a column
# ---------------------------------------------------------------------
# Sort descending by number of male participants
top_events = events_df.sort_values(by='participants_m', ascending=False)
print("🔹 Top 5 events by male participants:")
print(top_events[['events', 'participants_m']].head(5))
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 12: Group and aggregate (Excel medals data)
# ---------------------------------------------------------------------
# Check expected columns exist before grouping
if {'country', 'gold', 'silver', 'bronze'}.issubset(medals_df.columns):
    medals_by_country = medals_df.groupby('country')[[
        'gold', 'silver', 'bronze']].sum()
    medals_by_country = medals_by_country.sort_values(by='gold', 
                                                      ascending=False)
    print("🏅 Top 10 countries by gold medals:")
    print(medals_by_country.head(10))
else:
    print("⚠️ Medals sheet does not have expected columns.")
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 13: Merge two DataFrames (optional)
# ---------------------------------------------------------------------
# The CSV and Excel might not have matching column names, so we check.
if 'events' in events_df.columns and 'event' in medals_df.columns:
    merged_df = pd.merge(events_df, medals_df, left_on='events', 
                         right_on='event', how='left')
    print("✅ Merged DataFrame created:", merged_df.shape)
else:
    merged_df = events_df.copy()
    print("⚠️ Merge skipped (no matching column names).")
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 14: Plot histograms for numeric data
# ---------------------------------------------------------------------
merged_df.hist(figsize=(10, 6))
plt.suptitle("Histogram of numeric columns")
plt.tight_layout()
plt.show()
print("✅ Displayed histograms.")
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 15: Boxplot
# ---------------------------------------------------------------------
numeric_cols = merged_df.select_dtypes('number').columns
merged_df[numeric_cols].boxplot(figsize=(10, 6))
plt.title("Boxplot of numeric columns")
plt.tight_layout()
plt.show()
print("✅ Displayed boxplot.")
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 16: Line chart (e.g. participants over time)
# ---------------------------------------------------------------------
if {'year', 'participants'}.issubset(merged_df.columns):
    merged_df.plot(x='year', y='participants', kind='line', 
                   title='Participants over Time')
    plt.tight_layout()
    plt.show()
    print("✅ Displayed line chart.")
else:
    print("⚠️ Could not plot line chart (columns missing).")
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 17: Export cleaned data
# ---------------------------------------------------------------------
output_path = data_folder / "paralympics_cleaned.csv"
merged_df.to_csv(output_path, index=False)
print(f"✅ Cleaned data exported to: {output_path}")
print("-" * 70)

# ---------------------------------------------------------------------
# Activity 18: Wrap up
# ---------------------------------------------------------------------
print("🎉 Week 2 workshop activities completed successfully!")
print("You have practised: loading, inspecting, cleaning, filtering, grouping," \
"plotting, and exporting data.")
