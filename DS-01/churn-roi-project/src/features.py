"""
Feature engineering module for customer churn prediction.
Creates derived features from raw customer data.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

from config import AUTOPAY_METHODS, TENURE_BUCKETS
from db import execute_query, load_dataframe_to_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_tenure_bucket(tenure: int) -> str:
    """
    Categorize tenure into buckets.
    
    Args:
        tenure: Tenure in months
    
    Returns:
        str: Tenure bucket ('0-12', '13-24', '25-48', '49+')
    """
    if tenure <= 12:
        return '0-12'
    elif tenure <= 24:
        return '13-24'
    elif tenure <= 48:
        return '25-48'
    else:
        return '49+'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw customer data.
    
    Args:
        df: DataFrame with raw customer data
    
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    logger.info("Engineering features...")
    features = pd.DataFrame()
    
    # Customer ID
    features['customer_id'] = df['customer_id']
    
    # === TENURE FEATURES ===
    features['tenure_months'] = df['tenure']
    features['tenure_bucket'] = df['tenure'].apply(create_tenure_bucket)
    
    # === FINANCIAL FEATURES ===
    # Charges per tenure month (average monthly spend over lifetime)
    features['charges_per_tenure_month'] = np.where(
        df['tenure'] > 0,
        df['total_charges'] / df['tenure'],
        df['monthly_charges']  # For tenure=0, use monthly charges
    )
    
    # Charge change ratio: current monthly / historical average
    # >1 means spending increased, <1 means decreased
    features['charge_change_ratio'] = np.where(
        features['charges_per_tenure_month'] > 0,
        df['monthly_charges'] / features['charges_per_tenure_month'],
        1.0
    )
    
    # === SERVICE FEATURES ===
    # Count total number of services
    service_columns = [
        'phone_service', 'internet_service', 'online_security', 
        'online_backup', 'device_protection', 'tech_support',
        'streaming_tv', 'streaming_movies'
    ]
    
    features['service_count'] = 0
    
    # Phone service
    if 'phone_service' in df.columns:
        features['has_phone_service'] = (df['phone_service'] == 'Yes').astype(int)
        features['service_count'] += features['has_phone_service']
    
    # Internet service
    if 'internet_service' in df.columns:
        features['has_internet_service'] = (df['internet_service'] != 'No').astype(int)
        features['service_count'] += features['has_internet_service']
    
    # Online security
    if 'online_security' in df.columns:
        features['has_online_security'] = (df['online_security'] == 'Yes').astype(int)
        features['service_count'] += features['has_online_security']
    
    # Tech support
    if 'tech_support' in df.columns:
        features['has_tech_support'] = (df['tech_support'] == 'Yes').astype(int)
        features['service_count'] += features['has_tech_support']
    
    # Streaming services (TV or Movies)
    if 'streaming_tv' in df.columns and 'streaming_movies' in df.columns:
        features['has_streaming_services'] = (
            ((df['streaming_tv'] == 'Yes') | (df['streaming_movies'] == 'Yes')).astype(int)
        )
        features['service_count'] += (df['streaming_tv'] == 'Yes').astype(int)
        features['service_count'] += (df['streaming_movies'] == 'Yes').astype(int)
    
    # === CONTRACT FEATURES ===
    if 'contract' in df.columns:
        features['is_month_to_month'] = (df['contract'] == 'Month-to-month').astype(int)
        features['is_one_year'] = (df['contract'] == 'One year').astype(int)
        features['is_two_year'] = (df['contract'] == 'Two year').astype(int)
    
    # === PAYMENT FEATURES ===
    if 'payment_method' in df.columns:
        features['has_autopay'] = df['payment_method'].isin(AUTOPAY_METHODS).astype(int)
    
    # === COMPLAINT PROXY ===
    # Flag: month-to-month + no tech support + no online security
    # This is a proxy for potentially dissatisfied customers
    features['potential_complaint_flag'] = (
        (features.get('is_month_to_month', 0) == 1) &
        (features.get('has_tech_support', 0) == 0) &
        (features.get('has_online_security', 0) == 0)
    ).astype(int)
    
    logger.info(f"Created {len(features.columns)} features for {len(features)} customers")
    logger.info(f"Feature columns: {features.columns.tolist()}")
    
    return features


def get_feature_statistics(features: pd.DataFrame) -> Dict:
    """
    Calculate statistics for engineered features.
    
    Args:
        features: DataFrame with engineered features
    
    Returns:
        dict: Feature statistics
    """
    stats = {
        'total_customers': len(features),
        'tenure_bucket_distribution': features['tenure_bucket'].value_counts().to_dict(),
        'avg_service_count': features['service_count'].mean(),
        'month_to_month_pct': features['is_month_to_month'].mean() * 100,
        'autopay_pct': features['has_autopay'].mean() * 100,
        'potential_complaints_pct': features['potential_complaint_flag'].mean() * 100,
    }
    
    logger.info("\nFeature Statistics:")
    logger.info(f"  Total customers: {stats['total_customers']}")
    logger.info(f"  Average services per customer: {stats['avg_service_count']:.2f}")
    logger.info(f"  Month-to-month contracts: {stats['month_to_month_pct']:.1f}%")
    logger.info(f"  Autopay adoption: {stats['autopay_pct']:.1f}%")
    logger.info(f"  Potential complaint flags: {stats['potential_complaints_pct']:.1f}%")
    logger.info(f"\nTenure distribution:")
    for bucket, count in sorted(stats['tenure_bucket_distribution'].items()):
        pct = count / stats['total_customers'] * 100
        logger.info(f"    {bucket}: {count} ({pct:.1f}%)")
    
    return stats


def load_customers_from_db() -> pd.DataFrame:
    """
    Load customer data from PostgreSQL.
    
    Returns:
        pd.DataFrame: Customer data
    """
    logger.info("Loading customer data from database...")
    query = "SELECT * FROM customers"
    df = execute_query(query)
    logger.info(f"Loaded {len(df)} customers from database")
    return df


def save_features_to_db(features: pd.DataFrame) -> None:
    """
    Save engineered features to PostgreSQL features table.
    
    Args:
        features: DataFrame with engineered features
    """
    logger.info("Saving features to database...")
    
    # First, truncate the features table to clear old data
    from db import get_psycopg2_connection
    try:
        with get_psycopg2_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE features;")
            conn.commit()
        logger.info("Truncated existing features table")
    except Exception as e:
        logger.warning(f"Could not truncate features table: {e}")
    
    # Now append the new features
    load_dataframe_to_db(features, 'features', if_exists='append', index=False)
    logger.info(f"Saved {len(features)} feature records to database")


def main():
    """
    Main feature engineering pipeline.
    """
    logger.info("=" * 80)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load customer data
    df = load_customers_from_db()
    logger.info(f"\nStep 1 complete: Loaded {len(df)} customers")
    
    # Step 2: Engineer features
    features = engineer_features(df)
    logger.info(f"\nStep 2 complete: Created {len(features.columns)} features")
    
    # Step 3: Calculate statistics
    stats = get_feature_statistics(features)
    logger.info(f"\nStep 3 complete: Feature statistics calculated")
    
    # Step 4: Save to database
    save_features_to_db(features)
    logger.info(f"\nStep 4 complete: Features saved to database")
    
    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING PIPELINE COMPLETE!")
    logger.info("=" * 80)
    
    return features, stats


if __name__ == "__main__":
    features, stats = main()
    print(f"\n✓ Feature engineering complete!")
    print(f"  - Total features: {len(features.columns)}")
    print(f"  - Customers processed: {len(features)}")
    print(f"  - Average services: {stats['avg_service_count']:.2f}")
