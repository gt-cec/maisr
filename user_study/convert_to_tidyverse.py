import pandas as pd
import numpy as np


def convert_to_tidyverse_format(csv_file_path, output_file_path=None):
    """
    Convert user study data from long format to tidyverse (wide) format.

    Input format: subject_id, agent, metric, value
    Output format: subject_id, agent, episode_duration, targets_identified, threats_identified

    Args:
        csv_file_path (str): Path to the input CSV file
        output_file_path (str): Path for output CSV file (optional)

    Returns:
        pd.DataFrame: Converted dataframe in tidyverse format
    """

    # Load the dataset
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    # Display basic information about the dataset
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head(10))

    # Check unique metrics
    print(f"\nUnique metrics: {df['metric'].unique()}")
    print(f"Unique agents: {df['agent'].unique()}")
    print(f"Number of unique subjects: {df['subject_id'].nunique()}")

    # Verify data structure - each subject-agent combination should have 3 metrics
    combinations = df.groupby(['subject_id', 'agent']).size()
    print(f"\nMetrics per subject-agent combination:")
    print(combinations.value_counts())

    # Convert from long to wide format
    print("\nConverting to tidyverse format...")
    df_wide = df.pivot_table(
        index=['subject_id', 'agent'],
        columns='metric',
        values='value',
        aggfunc='first'  # Use first in case of duplicates
    ).reset_index()

    # Flatten column names (remove the multi-level structure)
    df_wide.columns.name = None

    # The exact column names will depend on what's in your 'metric' column
    # Common mapping based on typical user study metrics:
    metric_mapping = {
        'episode_duration': 'episode_duration',
        'targets_identified': 'targets_identified',
        'threats_identified': 'threats_identified',
        # Add other mappings as needed based on your actual metric names
    }

    # Rename columns if they match expected patterns
    current_columns = [col for col in df_wide.columns if col not in ['subject_id', 'agent']]
    print(f"\nCurrent metric columns: {current_columns}")

    # If we have exactly 3 metric columns, we can proceed
    if len(current_columns) == 3:
        # Try to intelligently map columns based on common patterns
        new_column_names = ['subject_id', 'agent']

        for col in current_columns:
            if 'duration' in col.lower() or 'time' in col.lower():
                new_column_names.append('episode_duration')
            elif 'target' in col.lower() and 'identified' in col.lower():
                new_column_names.append('targets_identified')
            elif 'threat' in col.lower() and 'identified' in col.lower():
                new_column_names.append('threats_identified')
            else:
                # Keep original name if we can't map it
                new_column_names.append(col)

        # Apply new column names
        df_wide.columns = new_column_names

    print(f"\nConverted data shape: {df_wide.shape}")
    print("Final columns:", list(df_wide.columns))
    print("\nFirst few rows of converted data:")
    print(df_wide.head())

    # Check for any missing values
    print(f"\nMissing values per column:")
    print(df_wide.isnull().sum())

    # Save to file if output path provided
    if output_file_path:
        df_wide.to_csv(output_file_path, index=False)
        print(f"\nConverted data saved to {output_file_path}")

    return df_wide


def validate_conversion(original_df, converted_df):
    """
    Validate that the conversion was successful.
    """
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    # Check row counts
    expected_rows = len(original_df) // 3  # Should be 1/3 of original since we're consolidating 3 rows into 1
    actual_rows = len(converted_df)

    print(f"Original rows: {len(original_df)}")
    print(f"Expected converted rows: {expected_rows}")
    print(f"Actual converted rows: {actual_rows}")
    print(f"Conversion ratio: {len(original_df) / len(converted_df):.1f}:1")

    if actual_rows == expected_rows:
        print("✓ Row count validation PASSED")
    else:
        print("✗ Row count validation FAILED")

    # Check for complete data
    total_missing = converted_df.isnull().sum().sum()
    if total_missing == 0:
        print("✓ No missing values - conversion complete")
    else:
        print(f"⚠ Warning: {total_missing} missing values found")


# Example usage
if __name__ == "__main__":
    # Convert the data
    input_file = "userstudy_data.csv"
    output_file = "userstudy_data_tidyverse.csv"

    try:
        # Load original data for validation
        original_df = pd.read_csv(input_file)

        # Convert to tidyverse format
        converted_df = convert_to_tidyverse_format(input_file, output_file)

        # Validate the conversion
        validate_conversion(original_df, converted_df)

        print("\n" + "=" * 50)
        print("CONVERSION COMPLETE!")
        print("=" * 50)

    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file}'")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        print("Please check your data format and try again.")