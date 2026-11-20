"""
Data loading and cleaning module.
Loads Telco customer churn data from Excel, cleans it, and loads to PostgreSQL.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Any

from config import TELCO_DATA_PATH, RANDOM_SEED
from db import load_dataframe_to_db, get_table_row_count, execute_sql_file, table_exists
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_telco_data(filepath: Path = TELCO_DATA_PATH) -> pd.DataFrame:
    """
    Load Telco customer churn data from Excel file.
    
    Args:
        filepath: Path to the Excel file
    
    Returns:
        pd.DataFrame: Raw data
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_excel(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess Telco customer churn data.
    
    Args:
        df: Raw DataFrame
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    df_clean = df.copy()
    
    # Standardize column names (lowercase with underscores)
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    
    # Ensure we have customer_id (might be customerid)
    if 'customerid' in df_clean.columns and 'customer_id' not in df_clean.columns:
        df_clean['customer_id'] = df_clean['customerid']
    
    logger.info(f"Columns after cleaning: {df_clean.columns.tolist()}")
    
    # Use churn_value (0/1) instead of churn_label (Yes/No)
    if 'churn_value' in df_clean.columns:
        df_clean['churn'] = df_clean['churn_value'].astype(int)
    elif 'churn_label' in df_clean.columns:
        df_clean['churn'] = (df_clean['churn_label'] == 'Yes').astype(int)
    
    # Handle tenure - may be called 'tenure' or 'tenure_months'
    if 'tenure_months' in df_clean.columns:
        df_clean['tenure'] = df_clean['tenure_months']
    
    # Handle Total Charges
    if 'total_charges' in df_clean.columns:
        logger.info(f"Total Charges dtype before cleaning: {df_clean['total_charges'].dtype}")
        
        # Replace empty strings and whitespace with NaN
        df_clean['total_charges'] = df_clean['total_charges'].replace(r'^\s*$', np.nan, regex=True)
        
        # Convert to numeric
        df_clean['total_charges'] = pd.to_numeric(df_clean['total_charges'], errors='coerce')
        
        # For customers with 0 tenure, set Total Charges to 0
        df_clean.loc[df_clean['tenure'] == 0, 'total_charges'] = 0.0
        
        # For others with missing Total Charges, impute with Monthly Charges * tenure
        missing_mask = df_clean['total_charges'].isna()
        if 'monthly_charges' in df_clean.columns:
            df_clean.loc[missing_mask, 'total_charges'] = (
                df_clean.loc[missing_mask, 'monthly_charges'] * 
                df_clean.loc[missing_mask, 'tenure']
            )
        
        logger.info(f"Imputed {missing_mask.sum()} missing Total Charges values")
    
    # Ensure numeric types for numeric columns
    # Handle senior_citizen - might be 0/1 or Yes/No
    if df_clean['senior_citizen'].dtype == 'object':
        df_clean['senior_citizen'] = (df_clean['senior_citizen'] == 'Yes').astype(int)
    else:
        df_clean['senior_citizen'] = df_clean['senior_citizen'].astype(int)
    
    df_clean['tenure'] = df_clean['tenure'].astype(int)
    
    if 'monthly_charges' in df_clean.columns:
        df_clean['monthly_charges'] = df_clean['monthly_charges'].astype(float)
    if 'total_charges' in df_clean.columns:
        df_clean['total_charges'] = df_clean['total_charges'].astype(float)
    
    # Data validation
    assert df_clean['churn'].isin([0, 1]).all(), "Churn must be 0 or 1"
    assert (df_clean['tenure'] >= 0).all(), "Tenure must be >= 0"
    if 'total_charges' in df_clean.columns:
        assert df_clean['total_charges'].isna().sum() == 0, "Still have missing Total Charges"
        assert (df_clean['total_charges'] >= 0).all(), "Total Charges must be >= 0"
    if 'monthly_charges' in df_clean.columns:
        assert (df_clean['monthly_charges'] >= 0).all(), "Monthly Charges must be >= 0"
    
    logger.info(f"Data cleaned successfully. Shape: {df_clean.shape}")
    logger.info(f"Churn rate: {df_clean['churn'].mean():.2%}")
    
    return df_clean


def verify_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform data quality checks and return statistics.
    
    Args:
        df: Cleaned DataFrame
    
    Returns:
        dict: Quality metrics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isna().sum().to_dict(),
        'duplicate_customer_ids': df['customer_id'].duplicated().sum() if 'customer_id' in df.columns else 0,
        'churn_rate': df['churn'].mean(),
        'tenure_stats': df['tenure'].describe().to_dict(),
    }
    
    if 'monthly_charges' in df.columns:
        stats['monthly_charges_stats'] = df['monthly_charges'].describe().to_dict()
    if 'total_charges' in df.columns:
        stats['total_charges_stats'] = df['total_charges'].describe().to_dict()
    
    logger.info("Data quality check:")
    logger.info(f"  Total rows: {stats['total_rows']}")
    logger.info(f"  Total columns: {stats['total_columns']}")
    logger.info(f"  Missing values: {sum(stats['missing_values'].values())}")
    logger.info(f"  Duplicate customer IDs: {stats['duplicate_customer_ids']}")
    logger.info(f"  Churn rate: {stats['churn_rate']:.2%}")
    
    return stats


def load_to_postgresql(df: pd.DataFrame, table_name: str = 'customers') -> int:
    """
    Load cleaned data to PostgreSQL.
    
    Args:
        df: Cleaned DataFrame
        table_name: Name of the table to load into
    
    Returns:
        int: Row count in table
    """
    logger.info(f"Loading data to PostgreSQL table '{table_name}'...")
    
    # Select only the columns we need for the customers table
    required_columns = [
        'customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
        'tenure', 'phone_service', 'multiple_lines', 'internet_service',
        'online_security', 'online_backup', 'device_protection', 'tech_support',
        'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing',
        'payment_method', 'monthly_charges', 'total_charges', 'churn'
    ]
    
    # Check which columns exist in the dataframe
    available_columns = [col for col in required_columns if col in df.columns]
    
    # Log what we found
    logger.info(f"Available columns: {available_columns}")
    logger.info(f"Missing columns: {[col for col in required_columns if col not in df.columns]}")
    
    # Make sure we have customer_id
    if 'customer_id' not in available_columns:
        logger.error("customer_id column is missing!")
        raise ValueError("customer_id column not found in dataframe")
    
    df_copy = df[available_columns].copy()
    
    # Load to database
    load_dataframe_to_db(df_copy, table_name, if_exists='append', index=False)
    
    # Verify
    row_count = get_table_row_count(table_name)
    logger.info(f"Verified: {row_count} rows in '{table_name}' table")
    
    return row_count


def setup_database_schema(schema_file: Path = None) -> None:
    """
    Create database schema from SQL file.
    
    Args:
        schema_file: Path to schema.sql file
    """
    if schema_file is None:
        schema_file = config.PROJECT_ROOT / "sql" / "schema.sql"
    
    logger.info(f"Setting up database schema from {schema_file}")
    execute_sql_file(str(schema_file))
    logger.info("Database schema created successfully")


def main():
    """
    Main ETL pipeline: Load, clean, and load to PostgreSQL.
    """
    logger.info("=" * 80)
    logger.info("STARTING DATA LOADING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load raw data
    df_raw = load_telco_data()
    logger.info(f"\nStep 1 complete: Loaded {len(df_raw)} rows")
    
    # Step 2: Clean data
    df_clean = clean_telco_data(df_raw)
    logger.info(f"\nStep 2 complete: Data cleaned")
    
    # Step 3: Verify data quality
    stats = verify_data_quality(df_clean)
    logger.info(f"\nStep 3 complete: Data quality verified")
    
    # Step 4: Setup database schema
    setup_database_schema()
    logger.info(f"\nStep 4 complete: Database schema created")
    
    # Step 5: Load to PostgreSQL
    row_count = load_to_postgresql(df_clean)
    logger.info(f"\nStep 5 complete: {row_count} rows loaded to PostgreSQL")
    
    logger.info("=" * 80)
    logger.info("DATA LOADING PIPELINE COMPLETE!")
    logger.info("=" * 80)
    
    return df_clean, stats


if __name__ == "__main__":
    df, stats = main()
    print("\n✓ Data loading complete!")
    print(f"  - Rows loaded: {len(df)}")
    print(f"  - Churn rate: {stats['churn_rate']:.2%}")
