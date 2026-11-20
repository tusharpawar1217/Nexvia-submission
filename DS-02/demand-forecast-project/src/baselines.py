"""
Milestone 4a: Baseline Models

Naive, seasonal naive, and moving average forecasts.
These provide benchmarks for more sophisticated models.
"""

import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path

from config import config, PROCESSED_DATA_DIR, MODELS_DIR
from metrics import evaluate_forecast, print_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_features(sample_frac: float = 0.1) -> pd.DataFrame:
    """
    Load feature dataset with optional sampling for memory efficiency.
    
    Args:
        sample_frac: Fraction of series to sample (0.1 = 10% of series)
    """
    logger.info(f"Loading features (sampling {sample_frac*100:.0f}% of series)...")
    
    # Load hierarchy to get series list
    hierarchy = pd.read_parquet(PROCESSED_DATA_DIR / 'hierarchy.parquet')
    
    # Sample series
    if sample_frac < 1.0:
        n_sample = max(100, int(len(hierarchy) * sample_frac))
        sampled_ids = np.random.choice(hierarchy['id'].values, size=n_sample, replace=False)
        logger.info(f"  Sampled {len(sampled_ids):,} series (out of {len(hierarchy):,})")
    else:
        sampled_ids = hierarchy['id'].values
        logger.info(f"  Using all {len(sampled_ids):,} series")
    
    # Load features with filtering
    df = pd.read_parquet(
        PROCESSED_DATA_DIR / 'features.parquet',
        filters=[('id', 'in', set(sampled_ids))]
    )
    
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def split_train_test(df: pd.DataFrame, horizon: int = 28) -> tuple:
    """
    Split into train/test sets.
    
    Test set is last `horizon` days.
    """
    logger.info(f"Splitting data (horizon={horizon} days)...")
    
    max_day = df['day_num'].max()
    split_day = max_day - horizon + 1
    
    train = df[df['day_num'] < split_day].copy()
    test = df[df['day_num'] >= split_day].copy()
    
    logger.info(f"  Train: d_1 to d_{split_day-1} ({len(train):,} rows)")
    logger.info(f"  Test:  d_{split_day} to d_{max_day} ({len(test):,} rows)")
    
    return train, test


def naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Naive forecast: last observed value.
    
    For each series, predict the last value from training set.
    """
    logger.info("Computing naive forecasts...")
    
    # Get last value per series
    last_values = train.groupby('id')['sales'].last().to_dict()
    
    # Apply to test set
    test = test.copy()
    test['forecast_naive'] = test['id'].map(last_values)
    
    logger.info("  ✅ Naive forecast complete")
    
    return test


def seasonal_naive_forecast(train: pd.DataFrame, test: pd.DataFrame, 
                          season: int = 7) -> pd.DataFrame:
    """
    Seasonal naive forecast: value from `season` periods ago.
    
    Args:
        season: Seasonality period (7 for weekly)
    """
    logger.info(f"Computing seasonal naive forecasts (season={season})...")
    
    # For each test point, get value from `season` days ago
    test = test.copy()
    
    # Create lookup from (id, day_num) -> sales
    train_lookup = train.set_index(['id', 'day_num'])['sales'].to_dict()
    
    def get_seasonal_value(row):
        key = (row['id'], row['day_num'] - season)
        return train_lookup.get(key, 0)  # Default to 0 if not found
    
    test['forecast_snaive'] = test.apply(get_seasonal_value, axis=1)
    
    logger.info("  ✅ Seasonal naive forecast complete")
    
    return test


def moving_average_forecast(train: pd.DataFrame, test: pd.DataFrame,
                           window: int = 28) -> pd.DataFrame:
    """
    Moving average forecast: average of last `window` values.
    """
    logger.info(f"Computing moving average forecasts (window={window})...")
    
    # Get last `window` values per series
    last_values = train.groupby('id')['sales'].apply(
        lambda x: x.tail(window).mean()
    ).to_dict()
    
    test = test.copy()
    test['forecast_ma'] = test['id'].map(last_values)
    
    logger.info("  ✅ Moving average forecast complete")
    
    return test


def evaluate_baselines(test: pd.DataFrame) -> dict:
    """Evaluate all baseline models."""
    logger.info("Evaluating baseline models...")
    
    results = {}
    
    for model in ['naive', 'snaive', 'ma']:
        col = f'forecast_{model}'
        if col in test.columns:
            metrics = evaluate_forecast(
                y_true=test['sales'].values,
                y_pred=test[col].values
            )
            results[model] = metrics
            print_metrics(metrics, title=f"{model.upper()} Model")
    
    return results


def save_baselines(test: pd.DataFrame):
    """Save baseline forecasts."""
    logger.info("Saving baseline forecasts...")
    
    # Save full test set with all forecasts
    output_path = MODELS_DIR / 'baselines_forecasts.parquet'
    test.to_parquet(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"  ✅ Saved {output_path.name} ({file_size_mb:.1f} MB)")


def main():
    """Main baseline pipeline."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("MILESTONE 4A: BASELINE MODELS")
    print("="*80 + "\n")
    
    # Load data
    df = load_features()
    
    # Split train/test
    horizon = config.get('forecast.horizon', 28)
    train, test = split_train_test(df, horizon=horizon)
    
    # Compute baselines
    test = naive_forecast(train, test)
    test = seasonal_naive_forecast(train, test, season=7)
    test = moving_average_forecast(train, test, window=28)
    
    # Evaluate
    results = evaluate_baselines(test)
    
    # Save
    save_baselines(test)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("✅ BASELINE MODELS COMPLETE")
    print("="*80)
    print(f"\n⏱️  Time: {elapsed:.0f}s")
    print(f"\n📊 Best Model (by WAPE):")
    best_model = min(results, key=lambda k: results[k]['wape'])
    print(f"  {best_model.upper()}: WAPE = {results[best_model]['wape']:.4f}")
    print(f"\n📁 Saved to:")
    print(f"  - {MODELS_DIR / 'baselines_forecasts.parquet'}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
