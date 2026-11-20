"""
Evaluation metrics for forecasting.

Includes WAPE, RMSSE, Pinball Loss, and others.
"""

import numpy as np
import pandas as pd
from typing import Optional


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.
    
    WAPE = sum(|y_true - y_pred|) / sum(|y_true|)
    
    Better than MAPE as it doesn't blow up for small values.
    """
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-10)


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, 
          y_train: Optional[np.ndarray] = None, scale: Optional[float] = None) -> float:
    """
    Root Mean Squared Scaled Error.
    
    Used in M5 competition. Scale-independent metric.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Training data for computing scale (naive forecast MSE)
        scale: Pre-computed scale factor (if y_train not provided)
    """
    mse = np.mean((y_true - y_pred) ** 2)
    
    if scale is None:
        if y_train is None:
            raise ValueError("Either y_train or scale must be provided")
        # Naive forecast scale: MSE of one-step-ahead naive forecast on train
        scale = np.mean(np.diff(y_train) ** 2)
    
    return np.sqrt(mse / (scale + 1e-10))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Pinball loss for quantile regression.
    
    Used to evaluate prediction intervals.
    
    Args:
        y_true: Actual values
        y_pred: Predicted quantile
        quantile: Target quantile (e.g., 0.5 for median, 0.9 for 90th percentile)
    """
    error = y_true - y_pred
    return np.mean(np.where(error >= 0, quantile * error, (quantile - 1) * error))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    
    Warning: Can blow up for small y_true values.
    """
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    More stable than MAPE.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / (denominator + 1e-10)) * 100


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Prediction interval coverage.
    
    Proportion of actual values that fall within [lower, upper].
    """
    return np.mean((y_true >= lower) & (y_true <= upper))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average width of prediction intervals."""
    return np.mean(upper - lower)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray,
                     y_train: Optional[np.ndarray] = None,
                     quantiles: Optional[dict] = None) -> dict:
    """
    Compute all relevant metrics for a forecast.
    
    Args:
        y_true: Actual values
        y_pred: Point forecast (median or mean)
        y_train: Training data for RMSSE
        quantiles: Dict of quantile predictions {0.1: array, 0.9: array, ...}
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'wape': wape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
    }
    
    # Add RMSSE if training data provided
    if y_train is not None:
        metrics['rmsse'] = rmsse(y_true, y_pred, y_train=y_train)
    
    # Add quantile metrics if provided
    if quantiles:
        for q, q_pred in quantiles.items():
            metrics[f'pinball_{int(q*100)}'] = pinball_loss(y_true, q_pred, q)
        
        # Coverage for 80% and 90% intervals
        if 0.1 in quantiles and 0.9 in quantiles:
            metrics['coverage_80'] = coverage(y_true, quantiles[0.1], quantiles[0.9])
            metrics['width_80'] = interval_width(quantiles[0.1], quantiles[0.9])
        
        if 0.05 in quantiles and 0.95 in quantiles:
            metrics['coverage_90'] = coverage(y_true, quantiles[0.05], quantiles[0.95])
            metrics['width_90'] = interval_width(quantiles[0.05], quantiles[0.95])
    
    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Metrics"):
    """Pretty print metrics."""
    print(f"\n{title}")
    print("=" * 50)
    for metric, value in metrics.items():
        if 'coverage' in metric:
            print(f"  {metric:20s}: {value:.2%}")
        elif 'width' in metric:
            print(f"  {metric:20s}: {value:.2f}")
        else:
            print(f"  {metric:20s}: {value:.4f}")
    print("=" * 50)
