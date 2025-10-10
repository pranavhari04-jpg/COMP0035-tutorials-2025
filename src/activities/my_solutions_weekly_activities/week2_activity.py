"""
COMP0035 Week 2 Activities – Pandas Practice
--------------------------------------------
This script covers all tasks from the Week 2 tutorial:
1. Load CSV and Excel data
2. Explore structure and summary statistics
3. Detect and handle missing values
4. Explore categorical/text data
5. Filter, sort, and group data
6. Merge dataframes (optional)
7. Visualise data using Matplotlib
8. Export cleaned data
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Locate data files
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent  # → src/activities
data_dir = project_root / "data"

csv_file = data_dir / "paralympics_raw.csv"
excel_file = data_dir / "paralympics_all_raw.xlsx"

print("✅ Checking data file paths:")
print("CSV path:  ", csv_file)
print("Excel path:", excel_file)
print("Files exist?:", csv_file.exists(), excel_file.exists())
print("-" * 60)

# ---------------------------------------------------------------------------
# Load CSV and Excel sheets
# ---------------------------------------------------------------------------
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
print("-" * 60)


# ---------------------------------------------------------------------------
# Helper: Describe a DataFrame
# ---------------------------------------------------------------------------
def describe_dataframe(df, name="DataFrame"):
    """Print structure, types, and summary statistics."""
    print(f"\n🔹 {name} overview")
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nData types:\n", df.dtypes)
    print("\nSummary statistics:\n", df.describe(include="all"))


describe_dataframe(events_df, "Events CSV")

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
print("\n🔹 Missing values per column:")
print(events_df.isna().sum())

missing_rows = events_df[events_df.isna().any(axis=1)]
print(f"\nNumber of rows with missing values: {len(missing_rows)}")

# Option: fill missing numeric columns with 0 for debugging
events_df_filled = events_df.fillna(0)
print("✅ Filled NaNs with 0 temporarily for further analysis.")
print("-" * 60)


# ---------------------------------------------------------------------------
# Categorical / text data
# ---------------------------------------------------------------------------
if "type" in events_df.columns:
    print("\n🔹 Unique event types before cleaning:")
    print(events_df["type"].unique())

    # Clean text fields
    events_df["type"] = events_df["type"].astype(str).str.strip().str.lower()

    print("\nUnique event types after cleaning:")
    print(events_df["type"].unique())
else:
    print("⚠️ No 'type' column found in CSV – skipping text cleaning.")

print("-" * 60)


# ---------------------------------------------------------------------------
# Filtering and sorting examples
# ---------------------------------------------------------------------------
if "type" in events_df.columns:
    summer_events = events_df[events_df["type"] == "summer"]
    print(f"Summer events count: {len(summer_events)}")

if "participants_m" in events_df.columns:
    top_participants = events_df.sort_values(
        by="participants_m", ascending=False
    ).head(5)
    print("\nTop 5 events by male participants:")
    print(top_participants[["events", "participants_m"]])

print("-" * 60)


# ---------------------------------------------------------------------------
# Grouping and aggregation (medals by country)
# ---------------------------------------------------------------------------
if {"country", "gold", "silver", "bronze"}.issubset(medals_df.columns):
    medal_summary = (
        medals_df.groupby("country")[["gold", "silver", "bronze"]]
        .sum()
        .sort_values(by="gold", ascending=False)
    )
    print("\n🏅 Top 10 countries by gold medals:")
    print(medal_summary.head(10))
else:
    print("⚠️ Expected medal columns not found in Excel sheet.")

print("-" * 60)


# ---------------------------------------------------------------------------
# Merge example (optional)
# ---------------------------------------------------------------------------
if "event" in medals_df.columns and "event" in events_df.columns:
    merged_df = pd.merge(events_df, medals_df, on="event", how="left")
    print(f"Merged DataFrame shape: {merged_df.shape}")
else:
    merged_df = events_df.copy()  # fallback
    print("⚠️ Skipping merge – 'event' column missing in one dataset.")

print("-" * 60)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
try:
    # Histogram
    merged_df.hist(figsize=(10, 6))
    plt.suptitle("Histogram of numeric columns")
    plt.tight_layout()
    plt.show()

    # Boxplot example
    numeric_cols = merged_df.select_dtypes("number").columns
    if len(numeric_cols) > 0:
        merged_df[numeric_cols].boxplot(figsize=(10, 6))
        plt.title("Boxplot of numeric columns")
        plt.tight_layout()
        plt.show()
except Exception as e:
    print("⚠️ Plotting issue:", e)

# ---------------------------------------------------------------------------
# Export cleaned data
# ---------------------------------------------------------------------------
output_dir = project_root / "data"
output_dir.mkdir(exist_ok=True)

cleaned_csv = output_dir / "paralympics_cleaned.csv"
try:
    merged_df.to_csv(cleaned_csv, index=False)
    print(f"✅ Cleaned dataset exported to: {cleaned_csv}")
except Exception as e:
    print("⚠️ Export failed:", e)

print("\n🎉 Week 2 activities completed successfully!")
