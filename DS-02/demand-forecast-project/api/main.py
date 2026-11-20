"""
FastAPI Backend for Retail Demand Forecasting & Inventory Optimization

Endpoints:
1. GET /health - Health check
2. GET /hierarchy - Get hierarchy information
3. GET /forecast/{series_id} - Get forecast for a series
4. GET /metrics - Get model performance metrics
5. POST /optimize - Get optimal order quantities
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import config, PROCESSED_DATA_DIR, MODELS_DIR

# Create app
app = FastAPI(
    title=config.get('api.title', 'Demand Forecasting API'),
    version=config.get('api.version', '1.0.0'),
    description=config.get('api.description', 'FastAPI backend for M5 forecasting')
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data on startup
hierarchy_df = None
forecasts_df = None
baselines_df = None

@app.on_event("startup")
async def load_data():
    """Load data on startup."""
    global hierarchy_df, forecasts_df, baselines_df
    
    try:
        hierarchy_df = pd.read_parquet(PROCESSED_DATA_DIR / 'hierarchy.parquet')
        print(f"✅ Loaded hierarchy: {len(hierarchy_df):,} series")
    except:
        print("⚠️  Warning: hierarchy.parquet not found")
        hierarchy_df = pd.DataFrame()
    
    try:
        forecasts_df = pd.read_parquet(MODELS_DIR / 'lgbm_forecasts.parquet')
        print(f"✅ Loaded LightGBM forecasts: {len(forecasts_df):,} rows")
    except:
        print("⚠️  Warning: lgbm_forecasts.parquet not found")
        forecasts_df = pd.DataFrame()
    
    try:
        baselines_df = pd.read_parquet(MODELS_DIR / 'baselines_forecasts.parquet')
        print(f"✅ Loaded baseline forecasts: {len(baselines_df):,} rows")
    except:
        print("⚠️  Warning: baselines_forecasts.parquet not found")
        baselines_df = pd.DataFrame()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Retail Demand Forecasting & Inventory Optimization API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/hierarchy",
            "/forecast/{series_id}",
            "/metrics",
            "/optimize"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "hierarchy_loaded": not hierarchy_df.empty,
        "forecasts_loaded": not forecasts_df.empty,
        "baselines_loaded": not baselines_df.empty,
        "series_count": len(hierarchy_df) if not hierarchy_df.empty else 0,
        "forecast_rows": len(forecasts_df) if not forecasts_df.empty else 0
    }


@app.get("/hierarchy")
async def get_hierarchy(
    state_id: Optional[str] = None,
    store_id: Optional[str] = None,
    cat_id: Optional[str] = None,
    limit: int = Query(100, le=1000)
):
    """
    Get hierarchy information.
    
    Query params:
    - state_id: Filter by state (CA, TX, WI)
    - store_id: Filter by store (CA_1, etc.)
    - cat_id: Filter by category (FOODS, HOUSEHOLD, HOBBIES)
    - limit: Max results (default: 100)
    """
    if hierarchy_df.empty:
        raise HTTPException(status_code=503, detail="Hierarchy data not loaded")
    
    df = hierarchy_df.copy()
    
    # Apply filters
    if state_id:
        df = df[df['state_id'] == state_id]
    if store_id:
        df = df[df['store_id'] == store_id]
    if cat_id:
        df = df[df['cat_id'] == cat_id]
    
    # Limit results
    df = df.head(limit)
    
    return {
        "total": len(df),
        "series": df.to_dict(orient='records')
    }


@app.get("/forecast/{series_id}")
async def get_forecast(series_id: str):
    """
    Get forecast for a specific series.
    
    Returns:
    - actual: Actual sales values
    - forecast: Median forecast (q50)
    - q05, q25, q75, q95: Quantile forecasts
    - dates: Forecast dates
    """
    if forecasts_df.empty:
        raise HTTPException(status_code=503, detail="Forecast data not loaded")
    
    # Filter to series
    series_forecasts = forecasts_df[forecasts_df['id'] == series_id]
    
    if len(series_forecasts) == 0:
        raise HTTPException(status_code=404, detail=f"No forecast found for series {series_id}")
    
    # Sort by day
    series_forecasts = series_forecasts.sort_values('day_num')
    
    # Replace NaN/inf with None for JSON serialization
    series_forecasts = series_forecasts.replace([np.nan, np.inf, -np.inf], None)
    
    # Extract data
    result = {
        "series_id": series_id,
        "forecast_days": int(len(series_forecasts)),
        "dates": series_forecasts['d'].tolist(),
        "actual": series_forecasts['actual'].tolist(),
        "forecast": series_forecasts['q50'].tolist(),
        "q05": series_forecasts['q05'].tolist(),
        "q25": series_forecasts['q25'].tolist(),
        "q75": series_forecasts['q75'].tolist(),
        "q95": series_forecasts['q95'].tolist(),
    }
    
    # Add series metadata if available
    if not hierarchy_df.empty:
        series_info = hierarchy_df[hierarchy_df['id'] == series_id]
        if len(series_info) > 0:
            result["metadata"] = series_info.iloc[0].to_dict()
    
    return result


@app.get("/metrics")
async def get_metrics(model: str = Query("lgbm", regex="^(lgbm|naive|snaive|ma)$")):
    """
    Get model performance metrics.
    
    Query params:
    - model: Model name (lgbm, naive, snaive, ma)
    """
    if model == "lgbm":
        if forecasts_df.empty:
            raise HTTPException(status_code=503, detail="Forecast data not loaded")
        df = forecasts_df
        y_true = df['actual'].values
        y_pred = df['q50'].values
    else:
        if baselines_df.empty:
            raise HTTPException(status_code=503, detail="Baseline data not loaded")
        df = baselines_df
        y_true = df['sales'].values
        y_pred = df[f'forecast_{model}'].values
    
    # Compute metrics
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-10))
    
    # Add quantile metrics for LightGBM
    if model == "lgbm" and 'q05' in df.columns and 'q95' in df.columns:
        coverage_90 = float(np.mean((y_true >= df['q05']) & (y_true <= df['q95'])))
        width_90 = float(np.mean(df['q95'] - df['q05']))
        
        return {
            "model": model,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "wape": round(wape, 4),
            "coverage_90": round(coverage_90, 4),
            "width_90": round(width_90, 2),
            "sample_size": len(y_true)
        }
    
    return {
        "model": model,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4),
        "sample_size": len(y_true)
    }


@app.post("/optimize")
async def optimize_inventory(
    series_id: str,
    service_level: float = Query(0.95, ge=0.5, le=0.99),
    budget: Optional[float] = None
):
    """
    Get optimal order quantity for a series.
    
    Body params:
    - series_id: Series to optimize
    - service_level: Target service level (0.95 = 95%)
    - budget: Budget constraint (optional)
    
    Returns:
    - order_qty: Recommended order quantity
    - expected_demand: Forecasted demand
    - safety_stock: Safety stock level
    - costs: Cost breakdown
    
    Note: Costs are ASSUMED (not in M5 dataset)
    """
    if forecasts_df.empty:
        raise HTTPException(status_code=503, detail="Forecast data not loaded")
    
    # Get forecast for series
    series_forecasts = forecasts_df[forecasts_df['id'] == series_id]
    
    if len(series_forecasts) == 0:
        raise HTTPException(status_code=404, detail=f"No forecast found for series {series_id}")
    
    # Get series metadata
    if hierarchy_df.empty:
        raise HTTPException(status_code=503, detail="Hierarchy data not loaded")
    
    series_info = hierarchy_df[hierarchy_df['id'] == series_id]
    if len(series_info) == 0:
        raise HTTPException(status_code=404, detail=f"Series {series_id} not in hierarchy")
    
    series_info = series_info.iloc[0]
    category = series_info['cat_id']
    
    # Expected demand (sum of median forecasts)
    expected_demand = float(series_forecasts['q50'].sum())
    
    # Demand variability (use std of forecast)
    demand_std = float(series_forecasts['q50'].std())
    
    # Safety stock (z-score for service level * std)
    from scipy.stats import norm
    z_score = norm.ppf(service_level)
    safety_stock = z_score * demand_std
    
    # Order quantity = expected demand + safety stock
    order_qty = int(np.ceil(expected_demand + safety_stock))
    
    # Get costs (ASSUMED - not in M5)
    unit_costs = config.get('inventory.unit_cost_by_category', {})
    unit_cost = unit_costs.get(category, config.get('inventory.unit_cost_default', 10.0))
    
    holding_cost_pct = config.get('inventory.holding_cost_pct_per_day', 0.001)
    holding_cost_total = order_qty * unit_cost * holding_cost_pct * 28  # 28 days
    
    stockout_penalties = config.get('inventory.stockout_penalty_by_category', {})
    stockout_penalty = stockout_penalties.get(category, config.get('inventory.stockout_penalty_default', 25.0))
    
    # Total order cost
    order_cost = order_qty * unit_cost
    
    # Check budget constraint
    if budget and order_cost > budget:
        order_qty = int(budget / unit_cost)
        order_cost = order_qty * unit_cost
        holding_cost_total = order_qty * unit_cost * holding_cost_pct * 28
    
    return {
        "series_id": series_id,
        "category": category,
        "recommendation": {
            "order_qty": order_qty,
            "expected_demand": round(expected_demand, 2),
            "safety_stock": round(safety_stock, 2),
            "service_level": service_level
        },
        "costs": {
            "unit_cost": round(unit_cost, 2),
            "order_cost": round(order_cost, 2),
            "holding_cost_28d": round(holding_cost_total, 2),
            "stockout_penalty": round(stockout_penalty, 2),
            "note": "ASSUMED - M5 dataset has no cost data"
        },
        "budget_used": round(order_cost, 2) if budget else None,
        "budget_remaining": round(budget - order_cost, 2) if budget else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.get('api.host', '127.0.0.1'),
        port=config.get('api.port', 8000),
        reload=True
    )
