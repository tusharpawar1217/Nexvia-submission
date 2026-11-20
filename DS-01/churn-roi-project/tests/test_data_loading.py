"""
Tests for data loading and cleaning functions.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_data import clean_telco_data


def test_clean_telco_data_converts_churn():
    """Test that churn is converted from Yes/No to 1/0."""
    df = pd.DataFrame({
        'Churn': ['Yes', 'No', 'Yes', 'No'],
        'customerID': ['001', '002', '003', '004'],
        'SeniorCitizen': [0, 1, 0, 1],
        'tenure': [1, 12, 24, 36],
        'MonthlyCharges': [50.0, 60.0, 70.0, 80.0],
        'TotalCharges': [50.0, 720.0, 1680.0, 2880.0],
    })
    
    # Add minimal required columns
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']:
        df[col] = 'No'
    
    result = clean_telco_data(df)
    
    assert 'churn' in result.columns
    assert result['churn'].dtype == int
    assert set(result['churn'].unique()) == {0, 1}
    assert result['churn'].sum() == 2  # Two 'Yes' values


def test_clean_telco_data_handles_missing_totalcharges():
    """Test that missing TotalCharges are imputed correctly."""
    df = pd.DataFrame({
        'customerID': ['001', '002', '003'],
        'SeniorCitizen': [0, 1, 0],
        'tenure': [5, 0, 10],
        'MonthlyCharges': [50.0, 60.0, 70.0],
        'TotalCharges': [250.0, ' ', '700.0'],  # Blank value
        'Churn': ['No', 'Yes', 'No'],
    })
    
    # Add minimal required columns
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']:
        df[col] = 'No'
    
    result = clean_telco_data(df)
    
    assert result['totalcharges'].isna().sum() == 0
    assert result.loc[1, 'totalcharges'] == 0.0  # tenure=0, so TotalCharges=0
    assert result['totalcharges'].dtype == float


def test_clean_telco_data_converts_numeric_types():
    """Test that numeric columns have correct types."""
    df = pd.DataFrame({
        'customerID': ['001'],
        'SeniorCitizen': [0],
        'tenure': [12],
        'MonthlyCharges': [50.0],
        'TotalCharges': [600.0],
        'Churn': ['No'],
    })
    
    # Add minimal required columns
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']:
        df[col] = 'No'
    
    result = clean_telco_data(df)
    
    assert result['senior_citizen'].dtype == int
    assert result['tenure'].dtype == int
    assert result['monthlycharges'].dtype == float
    assert result['totalcharges'].dtype == float
    assert result['churn'].dtype == int


def test_clean_telco_data_validates_constraints():
    """Test that data constraints are enforced."""
    df = pd.DataFrame({
        'customerID': ['001', '002'],
        'SeniorCitizen': [0, 1],
        'tenure': [12, 24],
        'MonthlyCharges': [50.0, 60.0],
        'TotalCharges': [600.0, 1440.0],
        'Churn': ['Yes', 'No'],
    })
    
    # Add minimal required columns
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']:
        df[col] = 'No'
    
    result = clean_telco_data(df)
    
    # All values should be >= 0
    assert (result['tenure'] >= 0).all()
    assert (result['monthlycharges'] >= 0).all()
    assert (result['totalcharges'] >= 0).all()
    
    # Churn should be 0 or 1
    assert result['churn'].isin([0, 1]).all()


def test_clean_telco_data_preserves_customer_ids():
    """Test that customer IDs are preserved during cleaning."""
    df = pd.DataFrame({
        'customerID': ['A001', 'B002', 'C003'],
        'SeniorCitizen': [0, 1, 0],
        'tenure': [12, 24, 36],
        'MonthlyCharges': [50.0, 60.0, 70.0],
        'TotalCharges': [600.0, 1440.0, 2520.0],
        'Churn': ['Yes', 'No', 'Yes'],
    })
    
    # Add minimal required columns
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']:
        df[col] = 'No'
    
    result = clean_telco_data(df)
    
    assert 'customerid' in result.columns
    assert len(result) == 3
    assert set(result['customerid']) == {'A001', 'B002', 'C003'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
