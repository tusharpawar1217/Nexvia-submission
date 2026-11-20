"""
Customer Lifetime Value (CLV) calculation and customer segmentation.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import logging
from typing import Dict, Tuple

from config import RANDOM_SEED, DISCOUNT_RATE, MONTHLY_MARGIN, SEGMENTATION_MODEL_PATH, SCALER_PATH
from db import execute_query, get_psycopg2_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_for_clv() -> pd.DataFrame:
    """
    Load customer data with predictions for CLV calculation.
    
    Returns:
        pd.DataFrame: Complete customer data
    """
    logger.info("Loading data for CLV calculation...")
    query = """
        SELECT 
            c.customer_id,
            c.tenure,
            c.monthly_charges,
            c.total_charges,
            c.contract,
            c.churn,
            p.churn_probability,
            p.expected_lifetime_months,
            p.median_survival_months,
            f.service_count,
            f.has_autopay,
            f.is_month_to_month
        FROM customers c
        JOIN predictions p ON c.customer_id = p.customer_id
        JOIN features f ON c.customer_id = f.customer_id
    """
    df = execute_query(query)
    logger.info(f"Loaded {len(df)} customers")
    return df


def calculate_clv(df: pd.DataFrame, discount_rate: float = DISCOUNT_RATE, 
                  margin: float = MONTHLY_MARGIN) -> pd.DataFrame:
    """
    Calculate Customer Lifetime Value (CLV).
    
    CLV = Expected Lifetime (months) × Monthly Revenue × Margin × Discount Factor
    
    Args:
        df: Dataframe with customer data
        discount_rate: Annual discount rate (default from config)
        margin: Profit margin percentage (default from config)
    
    Returns:
        pd.DataFrame: DataFrame with CLV calculations
    """
    logger.info(f"Calculating CLV (discount rate: {discount_rate:.1%}, margin: {margin:.1%})...")
    
    clv_df = df.copy()
    
    # Monthly discount rate
    monthly_discount = discount_rate / 12
    
    # Calculate CLV components
    clv_df['monthly_revenue'] = clv_df['monthly_charges']
    clv_df['monthly_profit'] = clv_df['monthly_revenue'] * margin
    
    # Predicted CLV using expected remaining lifetime from survival model
    # For churned customers, use actual tenure; for active, use predicted lifetime
    clv_df['remaining_lifetime'] = np.where(
        clv_df['churn'] == 1,
        0,  # Already churned
        clv_df['expected_lifetime_months'] - clv_df['tenure']  # Remaining time
    )
    
    # Simple discount factor: 1 / (1 + r)^t for each month
    # For simplicity, use average discount over the period
    clv_df['discount_factor'] = np.where(
        clv_df['remaining_lifetime'] > 0,
        (1 - (1 + monthly_discount) ** (-clv_df['remaining_lifetime'])) / monthly_discount,
        0
    )
    
    # Predicted CLV (discounted future value)
    clv_df['clv_predicted'] = (
        clv_df['monthly_profit'] * clv_df['discount_factor']
    )
    
    # Historical CLV (what they've already contributed)
    clv_df['clv_historical'] = clv_df['total_charges'] * margin
    
    # Total CLV (historical + predicted)
    clv_df['clv_total'] = clv_df['clv_historical'] + clv_df['clv_predicted']
    
    logger.info(f"CLV calculated:")
    logger.info(f"  Average predicted CLV: ${clv_df['clv_predicted'].mean():.2f}")
    logger.info(f"  Average historical CLV: ${clv_df['clv_historical'].mean():.2f}")
    logger.info(f"  Average total CLV: ${clv_df['clv_total'].mean():.2f}")
    logger.info(f"  Max total CLV: ${clv_df['clv_total'].max():.2f}")
    logger.info(f"  Min total CLV: ${clv_df['clv_total'].min():.2f}")
    
    return clv_df


def select_optimal_clusters(X_scaled: np.ndarray, max_k: int = 10) -> int:
    """
    Select optimal number of clusters using elbow method and silhouette score.
    
    Args:
        X_scaled: Scaled feature matrix
        max_k: Maximum number of clusters to try
    
    Returns:
        int: Optimal number of clusters
    """
    logger.info(f"Finding optimal number of clusters (max_k={max_k})...")
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Find elbow using rate of change
    inertia_diffs = np.diff(inertias)
    elbow_k = np.argmax(inertia_diffs) + 2  # +2 because we start from k=2 and diff reduces by 1
    
    # Find best silhouette
    best_silhouette_k = list(K_range)[np.argmax(silhouette_scores)]
    
    logger.info(f"\nCluster evaluation:")
    for i, k in enumerate(K_range):
        logger.info(f"  k={k}: inertia={inertias[i]:.0f}, silhouette={silhouette_scores[i]:.3f}")
    
    logger.info(f"\nElbow method suggests: k={elbow_k}")
    logger.info(f"Best silhouette score: k={best_silhouette_k}")
    
    # Use silhouette if it's reasonable (4-6 clusters), otherwise use elbow
    if 4 <= best_silhouette_k <= 6:
        optimal_k = best_silhouette_k
    else:
        optimal_k = min(max(elbow_k, 4), 6)  # Constrain to 4-6 clusters
    
    logger.info(f"Selected: k={optimal_k}")
    
    return optimal_k


def perform_clustering(df: pd.DataFrame, n_clusters: int = None) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Perform K-Means clustering on customer features.
    
    Args:
        df: DataFrame with CLV and features
        n_clusters: Number of clusters (if None, will be determined automatically)
    
    Returns:
        tuple: (DataFrame with segments, KMeans model, Scaler)
    """
    logger.info("Performing customer segmentation...")
    
    # Select features for clustering
    cluster_features = [
        'clv_total',
        'churn_probability',
        'tenure',
        'monthly_charges',
        'service_count',
        'expected_lifetime_months'
    ]
    
    X = df[cluster_features].copy()
    
    # Handle any NaN values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal clusters if not specified
    if n_clusters is None:
        n_clusters = select_optimal_clusters(X_scaled, max_k=8)
    
    # Fit K-Means
    logger.info(f"Fitting K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    df['segment_id'] = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, df['segment_id'])
    logger.info(f"Silhouette score: {silhouette:.3f}")
    
    return df, kmeans, scaler


def profile_segments(df: pd.DataFrame) -> Dict:
    """
    Profile and name each customer segment.
    
    Args:
        df: DataFrame with segments
    
    Returns:
        dict: Segment profiles with names and characteristics
    """
    logger.info("Profiling customer segments...")
    
    segment_profiles = {}
    
    for seg_id in sorted(df['segment_id'].unique()):
        seg_df = df[df['segment_id'] == seg_id]
        
        profile = {
            'segment_id': int(seg_id),
            'size': len(seg_df),
            'size_pct': len(seg_df) / len(df) * 100,
            'avg_clv': seg_df['clv_total'].mean(),
            'avg_churn_prob': seg_df['churn_probability'].mean(),
            'avg_tenure': seg_df['tenure'].mean(),
            'avg_monthly_charges': seg_df['monthly_charges'].mean(),
            'avg_service_count': seg_df['service_count'].mean(),
            'churn_rate': seg_df['churn'].mean(),
            'month_to_month_pct': seg_df['is_month_to_month'].mean() * 100
        }
        
        # Assign descriptive name based on characteristics
        if profile['avg_clv'] > df['clv_total'].quantile(0.75):
            if profile['avg_churn_prob'] > 0.5:
                name = "High-Value At-Risk"
            else:
                name = "Champions"
        elif profile['avg_clv'] > df['clv_total'].median():
            if profile['avg_churn_prob'] > 0.5:
                name = "Mid-Value At-Risk"
            else:
                name = "Loyal Customers"
        else:
            if profile['avg_churn_prob'] > 0.5:
                name = "Low-Value At-Risk"
            else:
                name = "Budget Customers"
        
        profile['name'] = name
        segment_profiles[seg_id] = profile
        
        logger.info(f"\nSegment {seg_id}: {name}")
        logger.info(f"  Size: {profile['size']:,} ({profile['size_pct']:.1f}%)")
        logger.info(f"  Avg CLV: ${profile['avg_clv']:.2f}")
        logger.info(f"  Avg Churn Prob: {profile['avg_churn_prob']:.1%}")
        logger.info(f"  Avg Tenure: {profile['avg_tenure']:.1f} months")
        logger.info(f"  Avg Monthly Charges: ${profile['avg_monthly_charges']:.2f}")
        logger.info(f"  Month-to-Month: {profile['month_to_month_pct']:.1f}%")
    
    return segment_profiles


def assign_segment_names(df: pd.DataFrame, segment_profiles: Dict) -> pd.DataFrame:
    """
    Assign segment names to dataframe.
    
    Args:
        df: DataFrame with segment_id
        segment_profiles: Dictionary of segment profiles
    
    Returns:
        pd.DataFrame: DataFrame with segment names
    """
    df['segment_name'] = df['segment_id'].map(
        {sid: profile['name'] for sid, profile in segment_profiles.items()}
    )
    return df


def save_clv_and_segments(df: pd.DataFrame) -> None:
    """
    Save CLV and segment data to database.
    
    Args:
        df: DataFrame with CLV and segments
    """
    logger.info("Saving CLV and segments to database...")
    
    # Update predictions table with CLV and segments
    with get_psycopg2_connection() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                update_sql = """
                    UPDATE predictions
                    SET 
                        clv_predicted = %s,
                        clv_historical = %s,
                        segment_id = %s,
                        segment_name = %s
                    WHERE customer_id = %s
                """
                cur.execute(update_sql, (
                    float(row['clv_predicted']),
                    float(row['clv_historical']),
                    int(row['segment_id']),
                    row['segment_name'],
                    row['customer_id']
                ))
        conn.commit()
    
    logger.info(f"Updated {len(df)} customer CLV and segments")


def main():
    """
    Main CLV and segmentation pipeline.
    """
    logger.info("=" * 80)
    logger.info("STARTING CLV & SEGMENTATION PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load data
    df = load_data_for_clv()
    logger.info(f"\nStep 1 complete: Loaded {len(df)} customers")
    
    # Step 2: Calculate CLV
    df = calculate_clv(df)
    logger.info(f"\nStep 2 complete: CLV calculated")
    
    # Step 3: Perform clustering
    df, kmeans, scaler = perform_clustering(df)
    logger.info(f"\nStep 3 complete: Customer segmentation done")
    
    # Step 4: Profile segments
    segment_profiles = profile_segments(df)
    logger.info(f"\nStep 4 complete: Segments profiled")
    
    # Step 5: Assign segment names
    df = assign_segment_names(df, segment_profiles)
    logger.info(f"\nStep 5 complete: Segment names assigned")
    
    # Step 6: Save models
    joblib.dump(kmeans, SEGMENTATION_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Models saved to {SEGMENTATION_MODEL_PATH} and {SCALER_PATH}")
    logger.info(f"\nStep 6 complete: Models saved")
    
    # Step 7: Save to database
    save_clv_and_segments(df)
    logger.info(f"\nStep 7 complete: CLV and segments saved to database")
    
    logger.info("=" * 80)
    logger.info("CLV & SEGMENTATION PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nFINAL RESULTS:")
    logger.info(f"  Average total CLV: ${df['clv_total'].mean():.2f}")
    logger.info(f"  Number of segments: {df['segment_id'].nunique()}")
    logger.info(f"\nSegment summary:")
    for seg_id, profile in segment_profiles.items():
        logger.info(f"  {profile['name']}: {profile['size']:,} customers (${profile['avg_clv']:.2f} avg CLV)")
    
    return df, segment_profiles


if __name__ == "__main__":
    df, profiles = main()
    print(f"\nCLV & Segmentation complete!")
    print(f"  Avg CLV: ${df['clv_total'].mean():.2f}")
    print(f"  Segments: {df['segment_id'].nunique()}")
