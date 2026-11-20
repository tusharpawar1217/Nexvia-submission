"""
FastAPI backend for Customer Churn ROI Prediction System.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from db import execute_query, get_psycopg2_connection
from config import (
    CHURN_MODEL_PATH, 
    SURVIVAL_MODEL_PATH,
    SEGMENTATION_MODEL_PATH,
    SCALER_PATH
)

app = FastAPI(
    title="Customer Churn ROI API",
    description="ML-powered customer churn prediction and ROI analysis",
    version="1.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
try:
    churn_model = joblib.load(CHURN_MODEL_PATH)
    survival_model = joblib.load(SURVIVAL_MODEL_PATH)
    segmentation_model = joblib.load(SEGMENTATION_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load models - {e}")
    churn_model = None
    survival_model = None
    segmentation_model = None
    scaler = None


# Pydantic models
class CustomerInput(BaseModel):
    customer_id: str
    
class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    expected_lifetime_months: float
    clv_predicted: float
    segment_name: str
    top_features: List[Dict[str, float]]

class SegmentProfile(BaseModel):
    segment_name: str
    count: int
    avg_clv: float
    avg_churn_prob: float
    avg_tenure: float
    avg_monthly_charges: float

class ROIInput(BaseModel):
    campaign_cost: float
    retention_rate: float
    target_segment: Optional[str] = None
    
class ROIResponse(BaseModel):
    total_customers: int
    campaign_cost: float
    expected_revenue: float
    net_roi: float
    roi_percentage: float
    customers_retained: int


# Health check
@app.get("/")
def read_root():
    return {
        "message": "Customer Churn ROI API",
        "status": "active",
        "models_loaded": churn_model is not None
    }


@app.get("/health")
def health_check():
    """Health check endpoint for Vercel."""
    return {
        "status": "healthy",
        "models": {
            "churn_model": churn_model is not None,
            "survival_model": survival_model is not None,
            "segmentation_model": segmentation_model is not None,
            "scaler": scaler is not None
        }
    }


@app.get("/api/stats")
def get_stats():
    """Get overall statistics."""
    try:
        query = """
            SELECT 
                COUNT(*) as total_customers,
                AVG(churn_probability) as avg_churn_prob,
                AVG(clv_predicted) as avg_clv,
                SUM(CASE WHEN churn_probability > 0.5 THEN 1 ELSE 0 END) as high_risk_count
            FROM predictions
        """
        result = execute_query(query)
        if result.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        stats = result.iloc[0].to_dict()
        return {
            "total_customers": int(stats["total_customers"]),
            "avg_churn_probability": float(stats["avg_churn_prob"]),
            "avg_clv": float(stats["avg_clv"]),
            "high_risk_customers": int(stats["high_risk_count"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerInput):
    """Get churn prediction for a specific customer."""
    try:
        query = f"""
            SELECT 
                c.customer_id,
                p.churn_probability,
                p.expected_lifetime_months,
                p.clv_predicted,
                p.segment_name,
                c.tenure,
                c.monthly_charges,
                f.service_count
            FROM customers c
            JOIN predictions p ON c.customer_id = p.customer_id
            JOIN features f ON c.customer_id = f.customer_id
            WHERE c.customer_id = '{customer.customer_id}'
        """
        result = execute_query(query)
        
        if result.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        customer_data = result.iloc[0]
        
        # Determine risk level
        prob = customer_data["churn_probability"]
        if prob > 0.7:
            risk_level = "High"
        elif prob > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Mock top features (in production, use SHAP values)
        top_features = [
            {"feature": "Contract Type", "impact": 0.25},
            {"feature": "Tenure", "impact": -0.18},
            {"feature": "Monthly Charges", "impact": 0.15},
        ]
        
        return ChurnPredictionResponse(
            customer_id=customer_data["customer_id"],
            churn_probability=float(customer_data["churn_probability"]),
            risk_level=risk_level,
            expected_lifetime_months=float(customer_data["expected_lifetime_months"]),
            clv_predicted=float(customer_data["clv_predicted"]),
            segment_name=customer_data["segment_name"],
            top_features=top_features
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/segments", response_model=List[SegmentProfile])
def get_segments():
    """Get all customer segments with profiles."""
    try:
        query = """
            SELECT 
                p.segment_name,
                COUNT(*) as count,
                AVG(p.churn_probability) as avg_churn_prob,
                AVG(p.clv_predicted) as avg_clv,
                AVG(c.tenure) as avg_tenure,
                AVG(c.monthly_charges) as avg_monthly_charges
            FROM predictions p
            JOIN customers c ON p.customer_id = c.customer_id
            GROUP BY p.segment_name
            ORDER BY avg_clv DESC
        """
        result = execute_query(query)
        
        segments = []
        for _, row in result.iterrows():
            segments.append(SegmentProfile(
                segment_name=row["segment_name"],
                count=int(row["count"]),
                avg_clv=float(row["avg_clv"]),
                avg_churn_prob=float(row["avg_churn_prob"]),
                avg_tenure=float(row["avg_tenure"]),
                avg_monthly_charges=float(row["avg_monthly_charges"])
            ))
        
        return segments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/high-risk")
def get_high_risk_customers(limit: int = 100):
    """Get high-risk customers for retention campaigns."""
    try:
        query = f"""
            SELECT 
                c.customer_id,
                p.churn_probability,
                p.clv_predicted,
                p.segment_name,
                c.tenure,
                c.monthly_charges
            FROM customers c
            JOIN predictions p ON c.customer_id = p.customer_id
            WHERE p.churn_probability > 0.5
            ORDER BY p.clv_predicted DESC, p.churn_probability DESC
            LIMIT {limit}
        """
        result = execute_query(query)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/roi", response_model=ROIResponse)
def calculate_roi(roi_input: ROIInput):
    """Calculate ROI for a retention campaign."""
    try:
        # Get target customers
        if roi_input.target_segment:
            query = f"""
                SELECT COUNT(*) as count, AVG(clv_predicted) as avg_clv
                FROM predictions
                WHERE segment_name = '{roi_input.target_segment}'
                AND churn_probability > 0.5
            """
        else:
            query = """
                SELECT COUNT(*) as count, AVG(clv_predicted) as avg_clv
                FROM predictions
                WHERE churn_probability > 0.5
            """
        
        result = execute_query(query)
        
        if result.empty:
            raise HTTPException(status_code=404, detail="No customers found")
        
        total_customers = int(result.iloc[0]["count"])
        avg_clv = float(result.iloc[0]["avg_clv"])
        
        # Calculate ROI
        customers_retained = int(total_customers * roi_input.retention_rate)
        expected_revenue = customers_retained * avg_clv
        total_cost = total_customers * roi_input.campaign_cost
        net_roi = expected_revenue - total_cost
        roi_percentage = (net_roi / total_cost * 100) if total_cost > 0 else 0
        
        return ROIResponse(
            total_customers=total_customers,
            campaign_cost=total_cost,
            expected_revenue=expected_revenue,
            net_roi=net_roi,
            roi_percentage=roi_percentage,
            customers_retained=customers_retained
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/customers/search")
def search_customers(query: str = "", limit: int = 50):
    """Search customers by ID."""
    try:
        sql = f"""
            SELECT 
                c.customer_id,
                p.churn_probability,
                p.segment_name,
                c.tenure,
                c.monthly_charges
            FROM customers c
            JOIN predictions p ON c.customer_id = p.customer_id
            WHERE c.customer_id LIKE '%{query}%'
            ORDER BY p.churn_probability DESC
            LIMIT {limit}
        """
        result = execute_query(sql)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
