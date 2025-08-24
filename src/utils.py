# src/utils.py
"""
Utility functions for loading and cleaning the Traffic Violations dataset.
"""

import pandas as pd

def load_data(path="../data/Indian_Traffic_Violations.csv"):
    """
    Load the traffic violations dataset.
    Args:
        path (str): Path to the dataset CSV file.
    Returns:
        DataFrame: Loaded dataset.
    """
    return pd.read_csv(path)

def clean_data(df):
    """
    Clean dataset: handle missing values, format dates, normalize text, remove duplicates.
    Args:
        df (DataFrame): Raw dataset.
    Returns:
        DataFrame: Cleaned dataset.
    """
    df = df.copy()

    # Fill missing values
    df['Helmet_Worn'] = df['Helmet_Worn'].fillna('Unknown')
    df['Seatbelt_Worn'] = df['Seatbelt_Worn'].fillna('Unknown')
    df['Comments'] = df['Comments'].fillna('No Comment')

    # Convert date and time
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.time
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')

    # Normalize Yes/No columns
    yes_no_columns = ['Helmet_Worn','Seatbelt_Worn','Towed','Fine_Paid','Court_Appearance_Required']
    for col in yes_no_columns:
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    # Drop duplicates
    df = df.drop_duplicates()

    return df
