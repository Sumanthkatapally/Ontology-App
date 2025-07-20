#!/usr/bin/env python3
"""
convert_sav_to_csv_with_labels.py

This script:
 1. Reads an SPSS .sav file into a Pandas DataFrame using pyreadstat.
 2. Loads a JSON metadata file that contains, for each column, a 'labelled_values' dict.
 3. Replaces every cell in each column with its mapped label (if one exists), 
    otherwise keeps the original value.
 4. Writes the resulting DataFrame out to a CSV file.
"""

import pandas as pd
import pyreadstat
import json
import sys

def load_sav_file(file_path: str):
    """
    Load an SPSS .sav file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the .sav file.

    Returns:
        df (pd.DataFrame): The data.
        meta: Metadata object from pyreadstat.
    """
    try:
        df, meta = pyreadstat.read_sav(file_path)
        print(f"SAV file loaded successfully from '{file_path}'")
        return df, meta
    except Exception as e:
        print(f"Error loading .sav file '{file_path}': {e}")
        sys.exit(1)

def load_metadata(json_path: str) -> dict:
    """
    Load the JSON metadata file that contains labelled_values mappings.

    Args:
        json_path (str): Path to the JSON metadata file.

    Returns:
        metadata (dict): Loaded metadata dictionary.
    """
    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata loaded successfully from '{json_path}'")
        return metadata
    except Exception as e:
        print(f"Error loading metadata JSON '{json_path}': {e}")
        sys.exit(1)

def apply_labelled_values(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    For each column in df that has a 'labelled_values' dict in metadata,
    replace its cell values according to that mapping.

    Args:
        df (pd.DataFrame): The DataFrame to transform.
        metadata (dict): A dict where keys are column names and values contain
                         a 'labelled_values' dict of original->label.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    for col, col_info in metadata.items():
        labelled = col_info.get('labelled_values')
        if not isinstance(labelled, dict) or col not in df.columns:
            continue

        # Convert all values to string for lookup, map, then fall back to original
        mapped = (
            df[col]
            .astype(str)
            .map(labelled)
        )
        # Use the mapped label when available; otherwise keep original df[col]
        df[col] = mapped.where(mapped.notna(), df[col])
        print(f"  • Applied labels for column '{col}'")

    return df

def save_to_csv(df: pd.DataFrame, output_path: str):
    """
    Save the DataFrame to a CSV file (without the index).

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str): Path where the CSV will be written.
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"DataFrame written to CSV at '{output_path}'")
    except Exception as e:
        print(f"Error saving DataFrame to CSV '{output_path}': {e}")
        sys.exit(1)

def main():
    # ---- USER CONFIGURATION ----
    sav_path      = r'D:\MARIST\Research\Environment\myenv\Project\KnowledgeGraph\data\US_SURVEY\US20240903_FINAL.sav'         # Path to your .sav file
    metadata_path = r'D:\MARIST\Research\Environment\myenv\Project\KnowledgeGraph\src\class_names_direct.json'    # Path to your JSON metadata file
    output_csv    = r'D:\MARIST\Research\Environment\myenv\Project\KnowledgeGraph\data\US_SURVEY\US20240325_FINAL.csv'      # Desired output CSV file path
    # ----------------------------

    # 1. Load the .sav
    df, meta = load_sav_file(sav_path)

    # 2. Load the metadata (with labelled_values per column)
    metadata = load_metadata(metadata_path)

    # 3. Apply the labelled_values mappings
    df = apply_labelled_values(df, metadata)

    # 4. (Optional) Inspect the first few rows
    print("\nSample of the transformed DataFrame:")
    print(df.head())

    # 5. Save to CSV
    save_to_csv(df, output_csv)

if __name__ == "__main__":
    main()