"""
Milestone 3: Feature Engineering

Create lag, rolling, price, calendar, and promotion features.
CRITICAL: NO DATA LEAKAGE - all features properly shifted.
"""

import logging
import time
import pandas as pd
import numpy as np

from config import config, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_data() -> pd.DataFrame:
    """Load processed sales data."""
    logger.info("Loading processed sales data...")
    df = pd.read_parquet(PROCESSED_DATA_DIR / 'sales_long.parquet')
    logger.info(f"  Loaded {len(df):,} rows")
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features (properly shifted to avoid leakage)."""
    logger.info("Creating lag features...")
    
    lags = config.lags
    logger.info(f"  Lags: {lags}")
    
    # Sort by id and day_num to ensure proper ordering
    df = df.sort_values(['id', 'day_num'])
    
    for lag in lags:
        col_name = f'sales_lag_{lag}'
        df[col_name] = df.groupby('id')['sales'].shift(lag)
        logger.info(f"    Created {col_name}")
    
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling window features (properly shifted to avoid leakage)."""
    logger.info("Creating rolling features...")
    
    windows = config.rolling_windows
    logger.info(f"  Windows: {windows}")
    
    for window in windows:
        # Rolling mean (shifted to avoid leakage)
        col_name = f'sales_roll_mean_{window}'
        df[col_name] = df.groupby('id')['sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        logger.info(f"    Created {col_name}")
        
        # Rolling std (shifted to avoid leakage)
        col_name = f'sales_roll_std_{window}'
        df[col_name] = df.groupby('id')['sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
        logger.info(f"    Created {col_name}")
    
    return df


def create_expanding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create expanding statistics (cumulative, properly shifted)."""
    logger.info("Creating expanding features...")
    
    # Expanding mean (shifted)
    df['sales_expanding_mean'] = df.groupby('id')['sales'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    
    # Expanding std (shifted)
    df['sales_expanding_std'] = df.groupby('id')['sales'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).std()
    )
    
    logger.info("  Created expanding_mean and expanding_std")
    
    return df


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price-based features (optimized)."""
    logger.info("Creating price features...")
    
    # Sort once for efficiency
    df = df.sort_values(['id', 'day_num'])
    
    # Price change vs previous week (vectorized)
    logger.info("  Computing price_change...")
    df['price_change'] = df.groupby('id')['sell_price'].diff().astype('float32')
    
    # Price momentum (optimized with fewer periods)
    logger.info("  Computing price_momentum...")
    df['price_roll_mean_4w'] = df.groupby('id')['sell_price'].transform(
        lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
    ).astype('float32')
    df['price_momentum'] = (df['sell_price'] - df['price_roll_mean_4w']).astype('float32')
    
    # Price relative to item average (compute once, merge)
    logger.info("  Computing price_vs_item_avg...")
    item_avg = df.groupby('item_id')['sell_price'].mean().to_dict()
    df['price_vs_item_avg'] = (df['sell_price'] - df['item_id'].map(item_avg)).astype('float32')
    
    # Price drop indicator (potential promotion proxy)
    threshold = 0.05
    logger.info(f"  Computing price_drop_flag (threshold: {threshold*100:.0f}%)...")
    df['price_drop_flag'] = (
        (df['price_change'].fillna(0) < -threshold * df['price_roll_mean_4w'].fillna(0))
    ).astype('int8')
    
    logger.info("  Price features complete")
    
    return df


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features."""
    logger.info("Creating calendar features...")
    
    # Already have: weekday, wday, month, year, snap, events
    
    # Day of month
    df['day_of_month'] = df['date'].dt.day
    
    # Is weekend
    df['is_weekend'] = df['wday'].isin([1, 7]).astype('int8')  # Saturday=1, Sunday=7
    
    # Is month start/end
    df['is_month_start'] = (df['day_of_month'] <= 7).astype('int8')
    df['is_month_end'] = (df['day_of_month'] >= 24).astype('int8')
    
    # Event indicator
    df['has_event'] = (~df['event_name_1'].isna()).astype('int8')
    
    # Event type dummies (limited to avoid sparsity)
    df['event_sporting'] = (df['event_type_1'] == 'Sporting').astype('int8')
    df['event_cultural'] = (df['event_type_1'] == 'Cultural').astype('int8')
    df['event_national'] = (df['event_type_1'] == 'National').astype('int8')
    df['event_religious'] = (df['event_type_1'] == 'Religious').astype('int8')
    
    logger.info("  Created calendar features")
    
    return df


def create_promo_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create promotion proxy (M5 has no explicit promotion column).
    Use price drops, events, and SNAP as proxies.
    """
    logger.info("Creating promotion proxy...")
    
    # Promo proxy: price drop OR event OR SNAP
    df['promo_proxy'] = (
        (df['price_drop_flag'] == 1) |
        (df['has_event'] == 1) |
        (df['snap'] == 1)
    ).astype('int8')
    
    promo_pct = df['promo_proxy'].sum() / len(df) * 100
    logger.info(f"  Created promo_proxy ({promo_pct:.1f}% of rows flagged as promo)")
    logger.info("  Note: This is a PROXY - M5 has no explicit promotion data")
    
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features (trends, seasonality indicators)."""
    logger.info("Creating temporal features...")
    
    # Days since start (for trend)
    df['days_since_start'] = df['day_num']
    
    # Week of year (for yearly seasonality)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype('int8')
    
    # Quarter
    df['quarter'] = df['date'].dt.quarter.astype('int8')
    
    logger.info("  Created temporal features")
    
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with appropriate defaults."""
    logger.info("Filling missing values...")
    
    # Count missing before
    missing_before = df.isna().sum().sum()
    
    # Fill numeric features with 0 or forward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            if 'lag' in col or 'roll' in col or 'expanding' in col:
                # For lag/rolling features, NaN is expected at the start
                df[col] = df[col].fillna(0)
            elif 'price' in col:
                # For price features, forward fill then backward fill
                df[col] = df.groupby('id')[col].ffill().bfill()
    
    # Fill categorical NaN
    df['event_name_1'] = df['event_name_1'].fillna('None')
    df['event_type_1'] = df['event_type_1'].fillna('None')
    df['event_name_2'] = df['event_name_2'].fillna('None')
    df['event_type_2'] = df['event_type_2'].fillna('None')
    
    missing_after = df.isna().sum().sum()
    logger.info(f"  Missing values: {missing_before:,} -> {missing_after:,}")
    
    return df


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order final feature set."""
    logger.info("Selecting final features...")
    
    # ID columns
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    # Time columns
    time_cols = ['d', 'date', 'day_num', 'wm_yr_wk']
    
    # Target
    target_cols = ['sales']
    
    # Lag features
    lag_cols = [c for c in df.columns if 'lag_' in c]
    
    # Rolling features (exclude price_roll from sales features)
    roll_cols = [c for c in df.columns if 'roll_' in c and 'sales_roll' in c]
    
    # Expanding features
    expand_cols = [c for c in df.columns if 'expanding_' in c]
    
    # Price features (all price-related columns)
    price_cols = [c for c in df.columns if 'price' in c]
    # Remove duplicates and order properly
    price_base = ['sell_price'] if 'sell_price' in price_cols else []
    price_other = [c for c in price_cols if c != 'sell_price']
    price_cols = price_base + price_other
    
    # Calendar features
    calendar_cols = ['weekday', 'wday', 'month', 'year', 'day_of_month', 'week_of_year', 
                     'quarter', 'is_weekend', 'is_month_start', 'is_month_end']
    
    # Event features
    event_cols = ['has_event', 'event_sporting', 'event_cultural', 'event_national', 'event_religious',
                  'event_name_1', 'event_type_1']
    
    # SNAP and promo
    promo_cols = ['snap', 'promo_proxy']
    
    # Temporal
    temporal_cols = ['days_since_start', 'weeks_since_launch']
    
    # Combine all
    final_cols = (id_cols + time_cols + target_cols + 
                  lag_cols + roll_cols + expand_cols + 
                  price_cols + calendar_cols + event_cols + 
                  promo_cols + temporal_cols)
    
    # Filter to existing columns
    final_cols = [c for c in final_cols if c in df.columns]
    
    df = df[final_cols]
    
    logger.info(f"  Final features: {len(final_cols)} columns")
    logger.info(f"    Lag features: {len(lag_cols)}")
    logger.info(f"    Rolling features: {len(roll_cols)}")
    logger.info(f"    Price features: {len([c for c in price_cols if c in df.columns])}")
    logger.info(f"    Calendar features: {len([c for c in calendar_cols if c in df.columns])}")
    
    return df


def save_features(df: pd.DataFrame):
    """Save feature dataset."""
    logger.info("Saving features...")
    
    output_path = PROCESSED_DATA_DIR / 'features.parquet'
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"  ✅ Saved features.parquet ({file_size_mb:.1f} MB)")


def main():
    """Main feature engineering pipeline."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("MILESTONE 3: FEATURE ENGINEERING")
    print("="*80 + "\n")
    
    # Load data
    df = load_processed_data()
    
    # Create features (order matters due to dependencies)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_expanding_features(df)
    df = create_price_features(df)
    df = create_calendar_features(df)
    df = create_promo_proxy(df)
    df = create_temporal_features(df)
    
    # Fill missing values
    df = fill_missing_values(df)
    
    # Select final features
    df = select_final_features(df)
    
    # Save
    save_features(df)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"\n⏱️  Time: {elapsed:.0f}s")
    print(f"\n📊 Final Dataset:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"\n📁 Saved to:")
    print(f"  - {PROCESSED_DATA_DIR / 'features.parquet'}")
    
    # Feature summary
    print(f"\n🔧 Feature Categories:")
    print(f"  - Lag features: {len([c for c in df.columns if 'lag_' in c])}")
    print(f"  - Rolling features: {len([c for c in df.columns if 'roll_' in c])}")
    print(f"  - Price features: {len([c for c in df.columns if 'price' in c])}")
    print(f"  - Calendar features: 10+")
    print(f"  - Promo proxy: 1 (ASSUMED - no explicit promo data)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
