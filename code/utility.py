"""Utility functions for saving and loading data."""

import json
import pandas as pd


def save_json(data, file_path):
    """Read data to specified filepath as a json"""
    with open(file_path, "w") as f:
        return json.dump(data, f, indent=4)


def load_json(file_path):
    """Read a json file and return the contents"""
    with open(file_path, "r") as f:
        return json.load(f)


def save_multiindex_df(df, file_path):
    """For a dataframe df whose columns are two-level MultiIndex where the second level may be empty,
    convert columns to a standard Index using mapping:
        MultiIndex: ('level_1', 'level_2')  --> Index: 'level_1.level_2'
        MultiIndex: ('level_1', '')         --> Index: 'level_1'
    Then save the converted df as an htsv file.
    """
    df.columns = [".".join(col) if col[1] else col[0] for col in df.columns]
    df.to_csv(file_path, sep="\t", index=False)


def load_multiindex_df(file_path):
    """Load a MultiIndex dataframe saved by the above save_multiindex_df function."""
    df = pd.read_csv(file_path, sep="\t")
    df.columns = pd.MultiIndex.from_tuples([col.split(".") if "." in col else [col, ""] for col in df.columns])
    return df
