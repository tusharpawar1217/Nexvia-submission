"""
Milestone 2: Data Preparation & Hierarchy

Load M5 data, filter to configured subset, melt to long format,
join calendar and prices, and save as Parquet.

Memory-efficient approach:
1. Read with explicit dtypes
2. Filter to subset BEFORE melting
3. Handle missing prices (item not launched yet)
4. Save intermediate Parquet files
"""

import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np

from config import config, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_calendar() -> pd.DataFrame:
    """Load and preprocess calendar data."""
    logger.info("Loading calendar...")
    
    df = pd.read_csv(
        config.get_m5_file('calendar.csv'),
        dtype={
            'wm_yr_wk': 'int16',
            'wday': 'int8',
            'month': 'int8',
            'year': 'int16',
            'snap_CA': 'int8',
            'snap_TX': 'int8',
            'snap_WI': 'int8'
        },
        parse_dates=['date']
    )
    
    # Extract day number from 'd' column (d_1 -> 1)
    df['day_num'] = df['d'].str.extract(r'(\d+)').astype('int16')
    
    logger.info(f"  Loaded {len(df):,} days")
    return df


def load_prices() -> pd.DataFrame:
    """Load and preprocess sell prices."""
    logger.info("Loading sell prices...")
    
    df = pd.read_csv(
        config.get_m5_file('sell_prices.csv'),
        dtype={
            'store_id': 'category',
            'item_id': 'category',
            'wm_yr_wk': 'int16',
            'sell_price': 'float32'
        }
    )
    
    logger.info(f"  Loaded {len(df):,} price records")
    return df


def get_hierarchy_info() -> pd.DataFrame:
    """Extract hierarchy information from sales file (without loading all sales data)."""
    logger.info("Extracting hierarchy information...")
    
    # Read only the ID columns
    df = pd.read_csv(
        config.get_m5_file('sales_train_evaluation.csv'),
        usecols=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        dtype={
            'id': 'category',
            'item_id': 'category',
            'dept_id': 'category',
            'cat_id': 'category',
            'store_id': 'category',
            'state_id': 'category'
        }
    )
    
    logger.info(f"  Total series in M5: {len(df):,}")
    logger.info(f"  States: {df['state_id'].nunique()} | Stores: {df['store_id'].nunique()} | Items: {df['item_id'].nunique()}")
    logger.info(f"  Categories: {df['cat_id'].nunique()} | Departments: {df['dept_id'].nunique()}")
    
    return df


def filter_hierarchy(df_hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Filter hierarchy to configured subset."""
    logger.info(f"Filtering to profile: {config.profile}")
    
    # Filter by states
    states = config.states
    df = df_hierarchy[df_hierarchy['state_id'].isin(states)].copy()
    logger.info(f"  After state filter ({states}): {len(df):,} series")
    
    # Filter by stores if specified
    stores = config.stores
    if stores:
        df = df[df['store_id'].isin(stores)].copy()
        logger.info(f"  After store filter ({stores}): {len(df):,} series")
    
    # Sample items per category
    sample_frac = config.item_sample_fraction
    if sample_frac < 1.0:
        logger.info(f"  Sampling {sample_frac*100:.0f}% of items per category...")
        
        # Get unique items per category
        sampled_items = []
        for cat in df['cat_id'].unique():
            cat_items = df[df['cat_id'] == cat]['item_id'].unique()
            n_sample = max(1, int(len(cat_items) * sample_frac))
            sampled = np.random.choice(cat_items, size=n_sample, replace=False)
            sampled_items.extend(sampled)
            logger.info(f"    {cat}: {len(cat_items)} items -> sampled {n_sample}")
        
        df = df[df['item_id'].isin(sampled_items)].copy()
        logger.info(f"  After item sampling: {len(df):,} series")
    else:
        logger.info(f"  Using 100% of items")
    
    # Final summary
    logger.info(f"\n📊 Filtered Dataset:")
    logger.info(f"  Series: {len(df):,}")
    logger.info(f"  States: {df['state_id'].nunique()} - {sorted(df['state_id'].unique())}")
    logger.info(f"  Stores: {df['store_id'].nunique()} - {sorted(df['store_id'].unique())}")
    logger.info(f"  Categories: {df['cat_id'].nunique()} - {sorted(df['cat_id'].unique())}")
    logger.info(f"  Departments: {df['dept_id'].nunique()}")
    logger.info(f"  Items: {df['item_id'].nunique()}")
    
    return df


def load_and_melt_sales(filtered_ids: pd.DataFrame, use_evaluation: bool = True) -> pd.DataFrame:
    """
    Load sales data for filtered IDs and melt to long format.
    
    Args:
        filtered_ids: DataFrame with filtered hierarchy (id, item_id, etc.)
        use_evaluation: If True, use evaluation file (d_1 to d_1941), else validation (d_1 to d_1913)
    """
    filename = 'sales_train_evaluation.csv' if use_evaluation else 'sales_train_validation.csv'
    logger.info(f"Loading sales from {filename}...")
    
    # Read only the rows we need
    id_column = 'id'
    keep_ids = set(filtered_ids['id'].str.replace('_evaluation', '', regex=False).str.replace('_validation', '', regex=False))
    
    # Read in chunks to manage memory
    logger.info("  Reading sales file in chunks...")
    chunks = []
    chunksize = 10000
    
    for chunk in pd.read_csv(config.get_m5_file(filename), chunksize=chunksize):
        # Filter to our subset
        # The ID format in evaluation file has '_evaluation' suffix
        chunk['id_base'] = chunk['id'].str.replace('_evaluation', '', regex=False).str.replace('_validation', '', regex=False)
        chunk = chunk[chunk['id_base'].isin(keep_ids)]
        
        if len(chunk) > 0:
            chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Loaded {len(df):,} series")
    
    # Separate ID columns and day columns
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    day_cols = [c for c in df.columns if c.startswith('d_') and c[2:].isdigit()]
    
    logger.info(f"  Day columns: {len(day_cols)} (d_1 to d_{len(day_cols)})")
    
    # Convert to int16 to save memory
    for col in day_cols:
        df[col] = df[col].astype('int16')
    
    # Melt to long format
    logger.info("  Melting to long format...")
    df_long = df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name='d',
        value_name='sales'
    )
    
    # Convert sales to int16 (sales are integers)
    df_long['sales'] = df_long['sales'].astype('int16')
    
    # Extract day number
    df_long['day_num'] = df_long['d'].str.extract(r'(\d+)').astype('int16')
    
    logger.info(f"  Long format: {len(df_long):,} rows")
    
    # Report memory usage
    memory_mb = df_long.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"  Memory usage: {memory_mb:.1f} MB")
    
    return df_long


def join_calendar_and_prices(df_sales: pd.DataFrame, df_calendar: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """Join sales with calendar and price data."""
    logger.info("Joining calendar data...")
    
    # Join calendar on day_num
    df = df_sales.merge(
        df_calendar[['day_num', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
                     'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
                     'snap_CA', 'snap_TX', 'snap_WI']],
        on='day_num',
        how='left'
    )
    
    logger.info(f"  After calendar join: {len(df):,} rows")
    
    # Add SNAP flag for the relevant state
    logger.info("  Adding state-specific SNAP flags...")
    df['snap'] = 0
    df.loc[df['state_id'] == 'CA', 'snap'] = df.loc[df['state_id'] == 'CA', 'snap_CA']
    df.loc[df['state_id'] == 'TX', 'snap'] = df.loc[df['state_id'] == 'TX', 'snap_TX']
    df.loc[df['state_id'] == 'WI', 'snap'] = df.loc[df['state_id'] == 'WI', 'snap_WI']
    
    # Drop individual state SNAP columns
    df = df.drop(columns=['snap_CA', 'snap_TX', 'snap_WI'])
    
    # Join prices
    logger.info("Joining price data...")
    df = df.merge(
        df_prices,
        on=['store_id', 'item_id', 'wm_yr_wk'],
        how='left'
    )
    
    logger.info(f"  After price join: {len(df):,} rows")
    
    # Report missing prices
    missing_price_pct = df['sell_price'].isna().sum() / len(df) * 100
    logger.info(f"  Missing prices: {df['sell_price'].isna().sum():,} ({missing_price_pct:.1f}%)")
    logger.info(f"    Note: Missing prices typically mean item not yet launched")
    
    return df


def compute_weeks_since_launch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weeks since launch (first non-null price) for each item-store."""
    logger.info("Computing weeks since launch...")
    
    # Find first week with price for each item-store
    first_week = df[df['sell_price'].notna()].groupby(['store_id', 'item_id'])['wm_yr_wk'].min()
    first_week = first_week.reset_index()
    first_week.columns = ['store_id', 'item_id', 'first_wm_yr_wk']
    
    # Join back
    df = df.merge(first_week, on=['store_id', 'item_id'], how='left')
    
    # Compute weeks since launch (negative = not launched yet)
    df['weeks_since_launch'] = df['wm_yr_wk'] - df['first_wm_yr_wk']
    
    # Fill missing with -999 (not launched)
    df['weeks_since_launch'] = df['weeks_since_launch'].fillna(-999).astype('int16')
    
    # Report statistics
    not_launched = (df['weeks_since_launch'] < 0).sum()
    logger.info(f"  Rows before launch: {not_launched:,} ({not_launched/len(df)*100:.1f}%)")
    
    return df


def save_processed_data(df: pd.DataFrame, hierarchy: pd.DataFrame):
    """Save processed data as Parquet."""
    logger.info("Saving processed data...")
    
    # Save sales long format
    sales_path = PROCESSED_DATA_DIR / 'sales_long.parquet'
    df.to_parquet(sales_path, index=False, engine='pyarrow')
    logger.info(f"  ✅ Saved sales_long.parquet ({len(df):,} rows)")
    
    # Save hierarchy
    hierarchy_path = PROCESSED_DATA_DIR / 'hierarchy.parquet'
    hierarchy.to_parquet(hierarchy_path, index=False, engine='pyarrow')
    logger.info(f"  ✅ Saved hierarchy.parquet ({len(hierarchy):,} series)")
    
    # Report file sizes
    sales_size_mb = sales_path.stat().st_size / 1024 / 1024
    hierarchy_size_mb = hierarchy_path.stat().st_size / 1024 / 1024
    logger.info(f"  File sizes: sales={sales_size_mb:.1f}MB, hierarchy={hierarchy_size_mb:.1f}MB")


def main():
    """Main data preparation pipeline."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("MILESTONE 2: DATA PREPARATION & HIERARCHY")
    print("="*80 + "\n")
    
    # Set random seed
    np.random.seed(config.seed)
    
    # Step 1: Load calendar and prices (full dataset, small files)
    df_calendar = load_calendar()
    df_prices = load_prices()
    
    # Step 2: Get hierarchy and filter to subset
    df_hierarchy_full = get_hierarchy_info()
    df_hierarchy_filtered = filter_hierarchy(df_hierarchy_full)
    
    # Step 3: Load and melt sales for filtered subset
    df_sales_long = load_and_melt_sales(df_hierarchy_filtered, use_evaluation=True)
    
    # Step 4: Join calendar and prices
    df = join_calendar_and_prices(df_sales_long, df_calendar, df_prices)
    
    # Step 5: Compute weeks since launch
    df = compute_weeks_since_launch(df)
    
    # Step 6: Save processed data
    save_processed_data(df, df_hierarchy_filtered)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\n⏱️  Time: {elapsed:.0f}s")
    print(f"\n📊 Final Dataset:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Day range: d_{df['day_num'].min()} to d_{df['day_num'].max()}")
    print(f"\n📁 Saved to:")
    print(f"  - {PROCESSED_DATA_DIR / 'sales_long.parquet'}")
    print(f"  - {PROCESSED_DATA_DIR / 'hierarchy.parquet'}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
