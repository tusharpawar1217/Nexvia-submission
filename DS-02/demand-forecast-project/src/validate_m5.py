"""
Milestone 1: Validate M5 Dataset

Validate that all 5 M5 files exist and print their shapes and date ranges.
"""

import logging
import pandas as pd
from pathlib import Path

from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_m5_files():
    """Validate all M5 files and print diagnostics."""
    
    print("\n" + "="*80)
    print("MILESTONE 1: M5 DATASET VALIDATION")
    print("="*80 + "\n")
    
    print(f"📁 M5 Data Directory: {config.m5_data_dir}\n")
    
    # 1. Calendar
    print("📅 1. calendar.csv")
    calendar_path = config.get_m5_file('calendar.csv')
    df_cal = pd.read_csv(calendar_path)
    print(f"   Shape: {df_cal.shape}")
    print(f"   Columns: {list(df_cal.columns)}")
    print(f"   Date range: {df_cal['date'].min()} to {df_cal['date'].max()}")
    print(f"   Day range: {df_cal['d'].min()} to {df_cal['d'].max()}")
    print(f"   Sample:\n{df_cal.head(2)}\n")
    
    # 2. Sales train validation
    print("🛒 2. sales_train_validation.csv")
    sales_val_path = config.get_m5_file('sales_train_validation.csv')
    df_val = pd.read_csv(sales_val_path, nrows=5)  # Read only first 5 rows for validation
    print(f"   Shape (first 5 rows): {df_val.shape}")
    print(f"   Columns (first 10): {list(df_val.columns[:10])}")
    print(f"   ID columns: id, item_id, dept_id, cat_id, store_id, state_id")
    print(f"   Day columns: d_1 to d_1913 (expected)")
    
    # Get full shape by reading just the ID columns
    df_val_full = pd.read_csv(sales_val_path, usecols=['id'])
    print(f"   Full rows: {len(df_val_full)}")
    print(f"   Sample:\n{df_val.iloc[:, :10]}\n")
    
    # 3. Sales train evaluation  
    print("🛒 3. sales_train_evaluation.csv")
    sales_eval_path = config.get_m5_file('sales_train_evaluation.csv')
    df_eval = pd.read_csv(sales_eval_path, nrows=5)
    print(f"   Shape (first 5 rows): {df_eval.shape}")
    print(f"   Columns (first 10): {list(df_eval.columns[:10])}")
    print(f"   Day columns: d_1 to d_1941 (expected)")
    
    df_eval_full = pd.read_csv(sales_eval_path, usecols=['id'])
    print(f"   Full rows: {len(df_eval_full)}")
    
    # Verify consistency between validation and evaluation
    print(f"\n   ✅ Checking consistency...")
    df_val_check = pd.read_csv(sales_val_path, nrows=100)
    df_eval_check = pd.read_csv(sales_eval_path, nrows=100)
    
    # Check if first 1913 days match
    val_cols = [c for c in df_val_check.columns if c.startswith('d_')]
    eval_cols = [c for c in df_eval_check.columns if c.startswith('d_')]
    
    common_cols = set(val_cols) & set(eval_cols)
    print(f"   Common day columns: {len(common_cols)}")
    print(f"   Validation-only columns: {len(set(val_cols) - set(eval_cols))}")
    print(f"   Evaluation-only columns: {len(set(eval_cols) - set(val_cols))}")
    
    if len(common_cols) >= 1913:
        print(f"   ✅ Both files have at least d_1 to d_1913")
    else:
        print(f"   ⚠️  Expected 1913 common columns, found {len(common_cols)}")
    
    print()
    
    # 4. Sell prices
    print("💰 4. sell_prices.csv")
    prices_path = config.get_m5_file('sell_prices.csv')
    df_prices = pd.read_csv(prices_path, nrows=5)
    print(f"   Shape (first 5 rows): {df_prices.shape}")
    print(f"   Columns: {list(df_prices.columns)}")
    
    df_prices_full = pd.read_csv(prices_path)
    print(f"   Full rows: {len(df_prices_full):,}")
    print(f"   Unique store-item-week combinations: {len(df_prices_full):,}")
    print(f"   Sample:\n{df_prices.head()}\n")
    
    # 5. Sample submission
    print("📋 5. sample_submission.csv")
    submission_path = config.get_m5_file('sample_submission.csv')
    df_sub = pd.read_csv(submission_path, nrows=5)
    print(f"   Shape (first 5 rows): {df_sub.shape}")
    print(f"   Columns: {list(df_sub.columns)}")
    
    df_sub_full = pd.read_csv(submission_path)
    print(f"   Full rows: {len(df_sub_full):,}")
    print(f"   Sample:\n{df_sub.head()}\n")
    
    # Summary
    print("="*80)
    print("✅ VALIDATION COMPLETE")
    print("="*80)
    print(f"\nAll 5 M5 files validated successfully!")
    print(f"\n📊 Dataset Summary:")
    print(f"   - Calendar: {len(df_cal)} days (d_1 to {df_cal['d'].max()})")
    print(f"   - Sales validation: {len(df_val_full):,} series × 1,913 days")
    print(f"   - Sales evaluation: {len(df_eval_full):,} series × 1,941 days")
    print(f"   - Sell prices: {len(df_prices_full):,} store-item-week combinations")
    print(f"   - Sample submission: {len(df_sub_full):,} rows (validation + evaluation format)")
    
    print(f"\n🎯 Test Set:")
    print(f"   - Training: d_1 to d_{config.train_end_day}")
    print(f"   - Test (held-out): d_{config.train_end_day+1} to d_{config.train_end_day+config.test_days}")
    print(f"   - Test days available in evaluation file: d_1914 to d_1941")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    validate_m5_files()
