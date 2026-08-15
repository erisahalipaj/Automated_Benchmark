"""
Clean and deduplicate analytics CSV files
"""

import pandas as pd
from pathlib import Path

ANALYTICS_DIR = Path("analytics")


def clean_token_scaling():
    """Remove duplicate rows from token_scaling data"""
    csv_path = ANALYTICS_DIR / "analysis_token_scaling.csv"
    df = pd.read_csv(csv_path)

    print(f"[INFO] Token Scaling: {len(df)} rows before cleaning")

    # Remove duplicates, keeping first occurrence
    # Group by model and max_tokens, keep only one entry per combination
    df_clean = df.groupby(["model", "max_tokens"]).first().reset_index()

    print(f"[INFO] Token Scaling: {len(df_clean)} rows after cleaning ({len(df) - len(df_clean)} duplicates removed)")

    # Save cleaned data
    df_clean.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")
    return df_clean


def clean_cohort_comparison():
    """Clean cohort comparison data"""
    csv_path = ANALYTICS_DIR / "analysis_cohort_comparison.csv"
    df = pd.read_csv(csv_path)

    print(f"[INFO] Cohort Comparison: {len(df)} rows before cleaning")

    # Remove any duplicates based on model
    df_clean = df.drop_duplicates(subset=["model"]).reset_index(drop=True)

    print(f"[INFO] Cohort Comparison: {len(df_clean)} rows after cleaning")

    # Save cleaned data
    df_clean.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")
    return df_clean


def clean_parameter_scaling():
    """Clean parameter scaling data"""
    csv_path = ANALYTICS_DIR / "analysis_parameter_scaling.csv"
    df = pd.read_csv(csv_path)

    print(f"[INFO] Parameter Scaling: {len(df)} rows before cleaning")

    # Remove any duplicates based on model
    df_clean = df.drop_duplicates(subset=["model"]).reset_index(drop=True)

    print(f"[INFO] Parameter Scaling: {len(df_clean)} rows after cleaning")

    # Save cleaned data
    df_clean.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")
    return df_clean


def clean_warmup():
    """Clean warmup data"""
    csv_path = ANALYTICS_DIR / "analysis_warmup.csv"
    df = pd.read_csv(csv_path)

    print(f"[INFO] Warmup: {len(df)} rows before cleaning")

    # Remove any duplicates based on model and config
    df_clean = df.drop_duplicates(subset=["model", "config"]).reset_index(drop=True)

    print(f"[INFO] Warmup: {len(df_clean)} rows after cleaning")

    # Save cleaned data
    df_clean.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")
    return df_clean


def clean_all():
    """Clean all analytics CSV files"""
    print("=" * 60)
    print("CLEANING ANALYTICS DATA")
    print("=" * 60)

    clean_token_scaling()
    print()
    clean_cohort_comparison()
    print()
    clean_parameter_scaling()
    print()
    clean_warmup()

    print("\n" + "=" * 60)
    print("✓ All analytics files cleaned successfully")
    print("=" * 60)


if __name__ == "__main__":
    clean_all()
