"""
Milestone 4b: LightGBM Model

Global forecasting model with quantile regression for prediction intervals.
"""

import logging
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import joblib

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


def prepare_lgbm_data(df: pd.DataFrame, horizon: int = 28) -> tuple:
    """
    Prepare data for LightGBM.
    
    Args:
        df: Feature dataframe
        horizon: Forecast horizon (test set size)
    
    Returns:
        X_train, y_train, X_test, y_test, feature_names, test_ids
    """
    logger.info("Preparing LightGBM data...")
    
    # Split train/test by day
    max_day = df['day_num'].max()
    split_day = max_day - horizon + 1
    
    train = df[df['day_num'] < split_day].copy()
    test = df[df['day_num'] >= split_day].copy()
    
    logger.info(f"  Train: {len(train):,} rows")
    logger.info(f"  Test:  {len(test):,} rows")
    
    # Define features to use
    exclude_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                    'd', 'date', 'sales', 'event_name_1', 'event_type_1', 
                    'event_name_2', 'event_type_2', 'weekday']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    logger.info(f"  Features: {len(feature_cols)}")
    
    # Prepare X, y
    X_train = train[feature_cols]
    y_train = train['sales'].values  # Convert to numpy array
    X_test = test[feature_cols]
    y_test = test['sales'].values  # Convert to numpy array
    test_ids = test[['id', 'd', 'day_num']].copy()
    
    logger.info(f"  ✅ Data prepared")
    
    return X_train, y_train, X_test, y_test, feature_cols, test_ids


def train_lgbm_quantile(X_train: pd.DataFrame, y_train: np.ndarray,
                        X_test: pd.DataFrame, y_test: np.ndarray,
                        quantile: float = 0.5) -> lgb.Booster:
    """
    Train LightGBM with quantile loss.
    
    Args:
        quantile: Target quantile (0.5 for median)
    """
    logger.info(f"Training LightGBM (quantile={quantile})...")
    
    # LightGBM parameters
    params = {
        'objective': 'quantile',
        'alpha': quantile,
        'metric': 'quantile',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': config.seed
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    logger.info(f"  ✅ Training complete (best iteration: {model.best_iteration})")
    
    return model


def train_lgbm_ensemble(X_train, y_train, X_test, y_test) -> dict:
    """
    Train ensemble of quantile models.
    
    Returns:
        Dictionary of {quantile: model}
    """
    logger.info("Training LightGBM ensemble...")
    
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    models = {}
    
    for q in quantiles:
        logger.info(f"\n--- Quantile {q} ---")
        models[q] = train_lgbm_quantile(X_train, y_train, X_test, y_test, quantile=q)
    
    logger.info("\n✅ Ensemble training complete")
    
    return models


def predict_with_ensemble(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions from ensemble.
    
    Returns:
        DataFrame with columns: q05, q25, q50, q75, q95
    """
    logger.info("Generating ensemble predictions...")
    
    predictions = pd.DataFrame()
    
    for q, model in models.items():
        col_name = f'q{int(q*100):02d}'
        predictions[col_name] = model.predict(X)
    
    logger.info("  ✅ Predictions complete")
    
    return predictions


def evaluate_lgbm(y_true: np.ndarray, predictions: pd.DataFrame) -> dict:
    """Evaluate LightGBM ensemble."""
    logger.info("Evaluating LightGBM...")
    
    # Point forecast is median (q50)
    y_pred = predictions['q50'].values
    
    # Quantiles for interval metrics
    quantiles = {
        0.05: predictions['q05'].values,
        0.25: predictions['q25'].values,
        0.50: predictions['q50'].values,
        0.75: predictions['q75'].values,
        0.95: predictions['q95'].values,
    }
    
    metrics = evaluate_forecast(y_true, y_pred, quantiles=quantiles)
    
    print_metrics(metrics, title="LightGBM Model")
    
    return metrics


def save_lgbm_models(models: dict, feature_names: list):
    """Save trained models."""
    logger.info("Saving LightGBM models...")
    
    # Save individual quantile models
    for q, model in models.items():
        model_path = MODELS_DIR / f'lgbm_q{int(q*100):02d}.txt'
        model.save_model(str(model_path))
        logger.info(f"  Saved {model_path.name}")
    
    # Save feature names
    feature_path = MODELS_DIR / 'lgbm_features.pkl'
    joblib.dump(feature_names, feature_path)
    logger.info(f"  Saved {feature_path.name}")
    
    logger.info("  ✅ Models saved")


def save_forecasts(test_ids: pd.DataFrame, y_test: np.ndarray, 
                   predictions: pd.DataFrame):
    """Save forecasts with metadata."""
    logger.info("Saving forecasts...")
    
    # Combine everything
    forecasts = test_ids.copy()
    forecasts['actual'] = y_test
    forecasts = pd.concat([forecasts, predictions], axis=1)
    
    # Save
    output_path = MODELS_DIR / 'lgbm_forecasts.parquet'
    forecasts.to_parquet(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"  ✅ Saved {output_path.name} ({file_size_mb:.1f} MB)")


def main():
    """Main LightGBM pipeline."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("MILESTONE 4B: LIGHTGBM MODEL")
    print("="*80 + "\n")
    
    # Load data
    df = load_features()
    
    # Prepare for LightGBM
    horizon = config.get('forecast.horizon', 28)
    X_train, y_train, X_test, y_test, feature_names, test_ids = prepare_lgbm_data(
        df, horizon=horizon
    )
    
    # Train ensemble
    models = train_lgbm_ensemble(X_train, y_train, X_test, y_test)
    
    # Generate predictions
    predictions = predict_with_ensemble(models, X_test)
    
    # Evaluate
    metrics = evaluate_lgbm(y_test, predictions)
    
    # Save models and forecasts
    save_lgbm_models(models, feature_names)
    save_forecasts(test_ids, y_test, predictions)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("✅ LIGHTGBM MODEL COMPLETE")
    print("="*80)
    print(f"\n⏱️  Time: {elapsed:.0f}s")
    print(f"\n📊 Performance:")
    print(f"  WAPE: {metrics['wape']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  Coverage 90%: {metrics.get('coverage_90', 0):.2%}")
    print(f"\n📁 Saved to:")
    print(f"  - {MODELS_DIR / 'lgbm_q*.txt'} (5 models)")
    print(f"  - {MODELS_DIR / 'lgbm_forecasts.parquet'}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
