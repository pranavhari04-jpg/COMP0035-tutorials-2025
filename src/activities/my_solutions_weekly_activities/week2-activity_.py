"""
COMP0035 Week 2 — Pandas Tutorial
---------------------------------
This script walks through ALL 18 activities from the Week 2 Pandas tutorial.

By the end, you will know how to:
✅ Load CSV and Excel files into pandas DataFrames
✅ Explore data structure, column types, and summary stats
✅ Detect and handle missing values
✅ Explore and clean categorical (text) data
✅ Filter and sort data
✅ Perform group-by and aggregation
✅ Merge dataframes
✅ Visualise data with Matplotlib
✅ Export cleaned data

Each section below matches a learning activity.
"""

# ---------------------------------------------------------------------------
# Activity 1: Import libraries and prepare file paths
# ---------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to your project folder dynamically
# __file__ = this Python file → .parent goes up one folder
project_root = Path(__file__).parent.parent  # Points to src/activities
data_dir = project_root / "data"

# Build full file paths to the dataset
csv_file = data_dir / "paralympics_raw.csv"
excel_file = data_dir / "paralympics_all_raw.xlsx"

# Check that the files exist before continuing
print("Checking data file paths...")
print("CSV path:  ", csv_file)
print("Excel path:", excel_file)
print("Files exist?:", csv_file.exists(), excel_file.exists())
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 2: Read CSV and Excel files into pandas DataFrames
# ---------------------------------------------------------------------------

# We use try/except to handle any file errors gracefully
try:
    events_df = pd.read_csv(csv_file)
    all_excel_df = pd.read_excel(excel_file, sheet_name=0)
    medals_df = pd.read_excel(excel_file, sheet_name=1)
except Exception as e:
    print("❌ ERROR reading files:", e)
    raise SystemExit

print("✅ Loaded data successfully!")
print(f"CSV shape: {events_df.shape}")
print(f"Excel Sheet 1 shape: {all_excel_df.shape}")
print(f"Excel Sheet 2 shape: {medals_df.shape}")
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 3: Display the first few rows (head) and check data structure
# ---------------------------------------------------------------------------

print("🔹 Preview of Events Data (first 5 rows):")
print(events_df.head())  # Shows the first 5 rows
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 4: Explore structure and data types
# ---------------------------------------------------------------------------

print("🔹 Columns in the dataset:")
print(events_df.columns.tolist())  # Prints all column names

print("\n🔹 Data types for each column:")
print(events_df.dtypes)  # Displays column data types

print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 5: Display shape and use .info()
# ---------------------------------------------------------------------------

print("🔹 DataFrame info() summary:")
print(events_df.info())  # Shows non-null counts and memory usage
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 6: Summary statistics
# ---------------------------------------------------------------------------

# describe() gives numeric stats (mean, std, min, etc.) and categorical stats (count, unique)
print("🔹 Summary statistics for all columns:")
print(events_df.describe(include="all"))
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 7: Identify missing values
# ---------------------------------------------------------------------------

print("🔹 Missing values per column:")
print(events_df.isna().sum())  # Count missing per column

# Find rows that have *any* missing values
missing_rows = events_df[events_df.isna().any(axis=1)]
print(f"\nNumber of rows with missing values: {len(missing_rows)}")
print(missing_rows)
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 8: Handle missing data (fill or drop)
# ---------------------------------------------------------------------------

# Example strategy: fill NaN with 0 for numeric columns
events_df_filled = events_df.fillna(0)
print("✅ Replaced NaN with 0 for numeric analysis safety.")
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 9: Explore categorical columns (like 'type')
# ---------------------------------------------------------------------------

if "type" in events_df.columns:
    print("🔹 Unique values before cleaning:")
    print(events_df["type"].unique())

    # Clean by stripping spaces and making lowercase
    events_df["type"] = events_df["type"].astype(str).str.strip().str.lower()

    print("🔹 Unique values after cleaning:")
    print(events_df["type"].unique())
else:
    print("⚠️ No 'type' column found, skipping categorical cleanup.")
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 10: Filter rows based on conditions
# ---------------------------------------------------------------------------

# Filter only "summer" type rows
if "type" in events_df.columns:
    summer_events = events_df[events_df["type"] == "summer"]
    print(f"✅ Filtered summer events ({len(summer_events)} rows):")
    print(summer_events.head(3))
else:
    print("⚠️ Could not filter 'summer' events – column missing.")
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 11: Sort data by a numeric column
# ---------------------------------------------------------------------------

if "participants_m" in events_df.columns:
    top_participants = events_df.sort_values(
        by="participants_m", ascending=False
    ).head(5)
    print("🔹 Top 5 events by male participants:")
    # Show only the relevant columns for clarity
    print(top_participants[["events", "participants_m"]])
else:
    print("⚠️ No column 'participants_m' found for sorting.")
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 12: Group and aggregate data (Excel medals sheet)
# ---------------------------------------------------------------------------

if {"country", "gold", "silver", "bronze"}.issubset(medals_df.columns):
    medal_summary = (
        medals_df.groupby("country")[["gold", "silver", "bronze"]]
        .sum()
        .sort_values(by="gold", ascending=False)
    )
    print("🏅 Top 10 countries by total gold medals:")
    print(medal_summary.head(10))
else:
    print("⚠️ Expected medal columns not found in Excel sheet.")
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 13: Merge DataFrames (optional join)
# ---------------------------------------------------------------------------

# Attempt to merge events and medals if both have a common column
if "event" in medals_df.columns and "events" in events_df.columns:
    merged_df = pd.merge(events_df, medals_df, left_on="events", right_on="event", how="left")
    print(f"✅ Merged DataFrame shape: {merged_df.shape}")
else:
    merged_df = events_df.copy()  # fallback
    print("⚠️ Skipping merge – matching 'event' columns not found.")
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 14: Create a histogram of numeric columns
# ---------------------------------------------------------------------------

# hist() automatically plots all numeric columns
try:
    merged_df.hist(figsize=(10, 6))
    plt.suptitle("Histogram of numeric columns", fontsize=14)
    plt.tight_layout()
    plt.show()
    print("✅ Displayed histograms.")
except Exception as e:
    print("⚠️ Histogram plotting failed:", e)
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 15: Boxplot for numeric columns
# ---------------------------------------------------------------------------

try:
    numeric_cols = merged_df.select_dtypes("number").columns
    merged_df[numeric_cols].boxplot(figsize=(10, 6))
    plt.title("Boxplot of numeric columns")
    plt.tight_layout()
    plt.show()
    print("✅ Displayed boxplots.")
except Exception as e:
    print("⚠️ Boxplot plotting failed:", e)
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 16: Line chart example
# ---------------------------------------------------------------------------

# Plot a line of total participants by year if both columns exist
if {"year", "participants"}.issubset(merged_df.columns):
    merged_df.plot(x="year", y="participants", kind="line", title="Participants over time")
    plt.tight_layout()
    plt.show()
    print("✅ Displayed line chart for participants over time.")
else:
    print("⚠️ 'year' or 'participants' column not found – skipping line plot.")
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 17: Export cleaned data
# ---------------------------------------------------------------------------

output_path = data_dir / "paralympics_cleaned.csv"
try:
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data exported to: {output_path}")
except Exception as e:
    print("⚠️ Export failed:", e)
print("-" * 70)

# ---------------------------------------------------------------------------
# Activity 18: Wrap up
# ---------------------------------------------------------------------------

print("🎉 All Week 2 Pandas activities completed successfully!")
print("You have now practiced reading, exploring, cleaning, grouping, merging, plotting, and exporting data.")
