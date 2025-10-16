"""
COMP0035 Week 2 — Pandas Workshop (Detailed / Tutorial-matching)
Save as:
    src/activities/my_solutions_weekly_activities/week2_activity.py

Run:
    source .venv/bin/activate    # (Mac/Linux) or .venv\Scripts\activate on Windows
    python src/activities/my_solutions_weekly_activities/week2_activity.py

This script follows the tutorial activities exactly and provides:
 - a `describe_dataframe()` function that prints shape, first/last 5 rows, columns,
   dtypes, .info(), and .describe() (Activity 3 requirement).
 - non-truncated printing (no "..." hiding columns).
 - step-by-step functions for missing values, plotting, grouping, merging and export.
"""

# ---------------------------------------------------------------------
# Activity 1: Imports & pandas display settings (no truncation for columns)
# ---------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Ensure terminal shows ALL columns; we avoid printing thousands of rows,
# but head()/tail() will be printed fully (no '...') using .to_string()
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", None)

# ---------------------------------------------------------------------
# Utility: safe print of a DataFrame portion without truncation
# ---------------------------------------------------------------------
def print_head_tail(df, n=5):
    """
    Print first n and last n rows of df without column truncation.
    Uses .to_string() so terminal prints full columns.
    """
    if df.empty:
        print("(empty DataFrame)")
        return
    print(f"-- first {n} rows --")
    print(df.head(n).to_string(index=False))
    print(f"\n-- last {n} rows --")
    print(df.tail(n).to_string(index=False))

# ---------------------------------------------------------------------
# Activity 3: describe_dataframe() - exactly matching the tutorial requests
# ---------------------------------------------------------------------
def describe_dataframe(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Print descriptive information about a DataFrame.

    This implements the tutorial checklist:
      - print shape (rows, columns)
      - print first 5 rows and last 5 rows (untruncated)
      - print column labels
      - print column datatypes
      - print df.info()
      - print df.describe()
    """
    print(f"\n=== Describe: {name} ===")
    # 1. Shape
    print("Shape (rows, columns):", df.shape)

    # 2. First 5 and last 5 rows (full columns)
    print_head_tail(df, n=5)

    # 3. Column labels
    print("\nColumn labels:")
    print(list(df.columns))

    # 4. Column datatypes
    print("\nColumn data types:")
    # to_string() is used so the dtype list does not get truncated
    print(df.dtypes.to_string())

    # 5. info()
    print("\nDataFrame info():")
    # df.info() prints directly; it returns None, so we call it and continue
    df.info()

    # 6. describe() results
    print("\nSummary statistics (describe):")
    # include='all' to show both numeric and non-numeric; .to_string() to avoid truncation
    try:
        print(df.describe(include="all").to_string())
    except Exception:
        # older pandas versions might raise for include='all' in some cases;
        # fall back to numeric describe
        print("Could not describe all columns; showing numeric describe instead.")
        print(df.describe().to_string())
    print(f"=== End describe: {name} ===\n")

# ---------------------------------------------------------------------
# Activity 2: Define file paths and check files exist
# ---------------------------------------------------------------------
project_root = Path(__file__).parent.parent  # src/activities
data_dir = project_root / "data"
csv_file = data_dir / "paralympics_raw.csv"
excel_file = data_dir / "paralympics_all_raw.xlsx"

print("✅ Checking data file paths:")
print("CSV path:  ", csv_file)
print("Excel path:", excel_file)
print("Files exist?:", csv_file.exists(), excel_file.exists())
print("-" * 80)

# ---------------------------------------------------------------------
# Activity 2 (continued): Load CSV and Excel - with defensive error handling
# ---------------------------------------------------------------------
try:
    events_df = pd.read_csv(csv_file)
    excel_sheet1_df = pd.read_excel(excel_file, sheet_name=0)  # first sheet
    excel_sheet2_df = pd.read_excel(excel_file, sheet_name=1)  # second sheet (medals)
except FileNotFoundError as e:
    print("ERROR: Data file not found:", e)
    raise SystemExit
except Exception as e:
    print("ERROR reading data files:", e)
    raise SystemExit

print("✅ Loaded data into DataFrames:")
print(f" events_df (CSV) shape: {events_df.shape}")
print(f" excel_sheet1_df shape: {excel_sheet1_df.shape}")
print(f" excel_sheet2_df shape: {excel_sheet2_df.shape}")
print("-" * 80)

# ---------------------------------------------------------------------
# Activity 3: Use the describe_dataframe function for each DataFrame
# ---------------------------------------------------------------------
# Per tutorial: call the describe function for each of the three dataframes.
describe_dataframe(events_df, "Events CSV (paralympics_raw.csv)")
describe_dataframe(excel_sheet1_df, "Excel Sheet 1 (paralympics_all_raw.xlsx - sheet 0)")
describe_dataframe(excel_sheet2_df, "Excel Sheet 2 (medals) (paralympics_all_raw.xlsx - sheet 1)")

# ---------------------------------------------------------------------
# Activity 4: Identify missing values (and print examples)
# ---------------------------------------------------------------------
def identify_missing(df: pd.DataFrame, name: str):
    """Print counts of missing values per column and example rows with missing values."""
    print(f"\n--- Missing value analysis: {name} ---")
    missing_counts = df.isna().sum()
    print("Missing values per column:")
    print(missing_counts.to_string())
    rows_with_missing = df[df.isna().any(axis=1)]
    print(f"\nRows with any missing values: {len(rows_with_missing)}")
    if not rows_with_missing.empty:
        print(rows_with_missing.to_string(index=False))
    print(f"--- End missing analysis: {name} ---\n")

identify_missing(events_df, "Events CSV")

# ---------------------------------------------------------------------
# Activity 5: Demonstrate ways to handle missing values (drop, fill, median)
# ---------------------------------------------------------------------
def fill_missing_example(df: pd.DataFrame, numeric_strategy: str = "zero"):
    """
    Show examples of filling missing values.
      numeric_strategy: 'zero' -> fill numeric NaN with 0
                        'median' -> fill numeric NaN with column median
                        'drop' -> drop rows with any NaN
    Returns a new DataFrame (copy).
    """
    df_copy = df.copy()
    if numeric_strategy == "zero":
        df_copy = df_copy.fillna(0)
        print("Filled NaN with 0 for all columns (temporary example).")
    elif numeric_strategy == "median":
        num_cols = df_copy.select_dtypes("number").columns
        for c in num_cols:
            median = df_copy[c].median()
            df_copy[c] = df_copy[c].fillna(median)
        # non-numeric left untouched
        print("Filled numeric NaN with median for numeric columns.")
    elif numeric_strategy == "drop":
        before = len(df_copy)
        df_copy = df_copy.dropna()
        after = len(df_copy)
        print(f"Dropped rows with NaN: before={before}, after={after}.")
    else:
        raise ValueError("Unknown numeric_strategy")
    return df_copy

# show 2 options (zero and median) but keep original events_df untouched unless you choose to assign
events_zero_filled = fill_missing_example(events_df, numeric_strategy="zero")
events_median_filled = fill_missing_example(events_df, numeric_strategy="median")

# ---------------------------------------------------------------------
# Activity 6: Explore categorical columns (unique, value_counts, cleaning)
# ---------------------------------------------------------------------
def explore_and_clean_categorical(df: pd.DataFrame, column: str):
    """Show unique values and value counts, then clean whitespace and case."""
    print(f"\n--- Categorical exploration for column '{column}' ---")
    if column not in df.columns:
        print(f"Column '{column}' not found.")
        return df
    print("Unique (raw):", df[column].unique())
    print("Value counts (raw):")
    print(df[column].value_counts(dropna=False).to_string())
    # Clean: strip whitespace and lower case (common exercise)
    df_clean = df.copy()
    df_clean[column] = df_clean[column].astype(str).str.strip().str.lower()
    print("\nUnique (cleaned):", df_clean[column].unique())
    print("Value counts (cleaned):")
    print(df_clean[column].value_counts(dropna=False).to_string())
    print(f"--- End categorical exploration for '{column}' ---\n")
    return df_clean

# Example: clean the 'type' column and overwrite events_df variable for further processing
events_df = explore_and_clean_categorical(events_df, "type")

# ---------------------------------------------------------------------
# Activity 7: Referencing specific columns/rows: examples of loc/iloc/at/iat/query
# ---------------------------------------------------------------------
def locating_examples(df: pd.DataFrame):
    """Demonstrate a few common locating patterns."""
    print("\n--- Locating examples ---")
    # bracket notation
    if 'country' in df.columns:
        print("Example - df['country'] (first 5):")
        print(df['country'].head(5).to_string(index=False))
    # loc using boolean mask
    if 'type' in df.columns:
        summer_mask = df['type'] == 'summer'
        print(f"Number of summer events: {summer_mask.sum()}")
    # iloc example: third row
    if len(df) >= 3:
        print("Third row by position (iloc[2]):")
        print(df.iloc[2].to_string())
    print("--- End locating examples ---\n")

locating_examples(events_df)

# ---------------------------------------------------------------------
# Activity 8: Remove columns example (drop URL/highlights/disabilities_included)
# ---------------------------------------------------------------------
def remove_unneeded_columns(df: pd.DataFrame, to_remove):
    """Return a copy of df with specified columns removed if they exist."""
    df_copy = df.copy()
    existing = [c for c in to_remove if c in df_copy.columns]
    df_copy = df_copy.drop(columns=existing)
    print(f"Removed columns: {existing} (if present).")
    return df_copy

events_for_prep = remove_unneeded_columns(events_df, ["URL", "disabilities_included", "highlights"])

# ---------------------------------------------------------------------
# Activity 9: Resolve missing/incorrect values example (string fixes + numeric coercion)
# ---------------------------------------------------------------------
def resolve_missing_and_incorrect(df: pd.DataFrame):
    """
    Fix a couple of common issues:
     - strip whitespace & lowercase for all object columns
     - coerce numeric-looking columns to numeric (errors -> NaN)
     - show before/after sample
    """
    df_copy = df.copy()
    # Make object columns tidy
    obj_cols = df_copy.select_dtypes(include="object").columns
    for c in obj_cols:
        df_copy[c] = df_copy[c].astype(str).str.strip()
    # Coerce 'countries', 'events' if they should be numeric
    for col in ['countries', 'events', 'participants_m', 'participants_f', 'participants']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    print("Applied basic string cleaning and numeric coercion (where applicable).")
    return df_copy

events_prepped = resolve_missing_and_incorrect(events_for_prep)

# ---------------------------------------------------------------------
# Activity 10: Change datatypes example (convert start/end to datetime)
# ---------------------------------------------------------------------
def convert_dtypes(df: pd.DataFrame):
    df_copy = df.copy()
    for col in ['start', 'end']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    print("Converted 'start' and 'end' to datetime where possible.")
    return df_copy

events_prepped = convert_dtypes(events_prepped)

# ---------------------------------------------------------------------
# Activity 11: Add new columns (example: compute participants_total if missing)
# ---------------------------------------------------------------------
def add_new_columns(df: pd.DataFrame):
    df_copy = df.copy()
    # If participants exists use it; else compute from male+female
    if 'participants' not in df_copy.columns or df_copy['participants'].isna().any():
        if {'participants_m', 'participants_f'}.issubset(df_copy.columns):
            df_copy['participants'] = df_copy['participants_m'].fillna(0) + df_copy['participants_f'].fillna(0)
            print("Added/filled 'participants' as participants_m + participants_f.")
    # Example computed column: duration (days) between start and end if datetime
    if {'start', 'end'}.issubset(df_copy.columns) and pd.api.types.is_datetime64_any_dtype(df_copy['start']):
        df_copy['duration_days'] = (df_copy['end'] - df_copy['start']).dt.days
        print("Added 'duration_days' column where start/end datetimes exist.")
    return df_copy

events_prepped = add_new_columns(events_prepped)

# ---------------------------------------------------------------------
# Activity 12: Joining dataframes (example using a conservative approach)
# ---------------------------------------------------------------------
def join_events_medals(events_df, medals_df):
    """
    Try to join medals_df onto events_df.
    The tutorial expects a join — we check for sensible key names and attempt matches.
    """
    # Common case: events_df has 'events' and medals_df has 'event' OR both may have 'event' or 'events'
    left_key = None
    right_key = None
    if 'events' in events_df.columns and 'event' in medals_df.columns:
        left_key, right_key = 'events', 'event'
    elif 'event' in events_df.columns and 'event' in medals_df.columns:
        left_key = right_key = 'event'
    elif 'events' in events_df.columns and 'events' in medals_df.columns:
        left_key = right_key = 'events'

    if left_key:
        merged = pd.merge(events_df, medals_df, left_on=left_key, right_on=right_key, how='left')
        print(f"Merged on keys: left='{left_key}' right='{right_key}' -> merged shape: {merged.shape}")
        return merged
    else:
        print("No obvious join key found between events and medals — skipping merge.")
        return events_df.copy()

merged_df = join_events_medals(events_prepped, excel_sheet2_df)

# ---------------------------------------------------------------------
# Activity 13: Save prepared dataset (also show sample)
# ---------------------------------------------------------------------
prepared_output = data_dir / "paralympics_prepared.csv"
merged_df.to_csv(prepared_output, index=False)
print(f"Saved prepared dataset to: {prepared_output}")
print("Preview of saved prepared dataset (first 5 rows):")
print(merged_df.head().to_string(index=False))

# ---------------------------------------------------------------------
# Activity 14–16: Plotting examples (histogram, boxplot, timeseries)
#    — plots are also saved as PNG to the data folder for submission
# ---------------------------------------------------------------------
def safe_save_plot(fig, filename):
    """Save current matplotlib figure to the data folder."""
    out = data_dir / filename
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved plot to: {out}")

# Histogram of participants_m/participants_f (if present)
def plot_histogram(df):
    cols = [c for c in ['participants_m', 'participants_f', 'participants'] if c in df.columns]
    if not cols:
        print("No numeric participant columns found for histogram.")
        return
    ax = df[cols].hist(figsize=(8, 4))
    plt.suptitle("Histogram of participants columns")
    plt.tight_layout()
    # Save figure
    fig = plt.gcf()
    safe_save_plot(fig, "hist_participants.png")
    plt.show()

plot_histogram(merged_df)

# Boxplot for numeric columns
def plot_boxplot(df):
    numeric_cols = df.select_dtypes("number").columns
    if len(numeric_cols) == 0:
        print("No numeric columns for boxplot.")
        return
    ax = df[numeric_cols].boxplot(figsize=(10, 4))
    plt.title("Boxplot of numeric columns")
    fig = plt.gcf()
    safe_save_plot(fig, "boxplot_numeric.png")
    plt.show()

plot_boxplot(merged_df)

# Timeseries: participants over year (if year numeric and participants exist)
def plot_timeseries(df):
    if {'year', 'participants'}.issubset(df.columns):
        # Ensure year is numeric
        df_ts = df.dropna(subset=['year', 'participants']).copy()
        df_ts['year'] = pd.to_numeric(df_ts['year'], errors='coerce')
        df_ts = df_ts.dropna(subset=['year'])
        df_group = df_ts.groupby('year')['participants'].sum().reset_index()
        ax = df_group.plot(x='year', y='participants', kind='line', marker='o', title='Participants over time')
        plt.tight_layout()
        fig = plt.gcf()
        safe_save_plot(fig, "timeseries_participants.png")
        plt.show()
    else:
        print("Cannot plot timeseries: missing 'year' or 'participants' columns.")

plot_timeseries(merged_df)

# ---------------------------------------------------------------------
# Activity 17: Example usage of prepared data (simple query)
# ---------------------------------------------------------------------
print("\nExample: participants by type (summer/winter) aggregated:")
if {'type', 'participants'}.issubset(merged_df.columns):
    agg = merged_df.groupby('type')['participants'].sum()
    print(agg.to_string())
else:
    print("type/participants columns not both present; skipping aggregation example.")

# ---------------------------------------------------------------------
# Activity 18: Final checks & wrap-up (export already done above)
# ---------------------------------------------------------------------
final_output = data_dir / "paralympics_cleaned.csv"
merged_df.to_csv(final_output, index=False)
print(f"\nFinal cleaned CSV exported to: {final_output}")

print("\n🎉 Week 2 activities (1-18) completed. Check printed outputs and saved plots in the 'data' folder.")
print(" - If a step was skipped it was due to missing expected columns; check column names and adjust code accordingly.")
print(" - For debugging: use `print(df.columns)` to inspect names and match keys for merges.")
