"""
Survival analysis module using Lifelines.
Estimates customer lifetime and survival probabilities.
"""
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import restricted_mean_survival_time
import matplotlib.pyplot as plt
import joblib
import logging
from typing import Dict, Tuple

from config import RANDOM_SEED, SURVIVAL_MODEL_PATH
from db import execute_query, get_psycopg2_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_for_survival() -> pd.DataFrame:
    """
    Load customer data for survival analysis.
    
    Returns:
        pd.DataFrame: Customer data with event indicator
    """
    logger.info("Loading data for survival analysis...")
    query = """
        SELECT 
            c.customer_id,
            c.tenure,
            c.churn as event,  -- 1 = churned (event occurred), 0 = censored
            c.contract,
            c.internet_service,
            c.payment_method,
            c.monthly_charges,
            c.total_charges,
            f.service_count,
            f.has_autopay,
            f.is_month_to_month,
            f.potential_complaint_flag
        FROM customers c
        JOIN features f ON c.customer_id = f.customer_id
    """
    df = execute_query(query)
    logger.info(f"Loaded {len(df)} customers")
    logger.info(f"Event rate (churn): {df['event'].mean():.2%}")
    logger.info(f"Censored rate (retained): {(1-df['event'].mean()):.2%}")
    return df


def kaplan_meier_overall(df: pd.DataFrame) -> KaplanMeierFitter:
    """
    Fit overall Kaplan-Meier survival curve.
    
    Args:
        df: Dataframe with tenure and event columns
    
    Returns:
        KaplanMeierFitter: Fitted model
    """
    logger.info("Fitting overall Kaplan-Meier curve...")
    
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df['tenure'], event_observed=df['event'], label='Overall')
    
    # Calculate metrics
    median_survival = kmf.median_survival_time_
    logger.info(f"Median survival time: {median_survival:.1f} months")
    
    # Survival at key timepoints
    survival_12m = kmf.survival_function_at_times(12).values[0] if 12 in kmf.survival_function_at_times(12).index else None
    survival_24m = kmf.survival_function_at_times(24).values[0] if 24 in kmf.survival_function_at_times(24).index else None
    survival_36m = kmf.survival_function_at_times(36).values[0] if 36 in kmf.survival_function_at_times(36).index else None
    
    logger.info(f"Survival at 12 months: {survival_12m:.2%}" if survival_12m else "12m not available")
    logger.info(f"Survival at 24 months: {survival_24m:.2%}" if survival_24m else "24m not available")
    logger.info(f"Survival at 36 months: {survival_36m:.2%}" if survival_36m else "36m not available")
    
    return kmf


def kaplan_meier_by_group(df: pd.DataFrame, group_col: str) -> Dict[str, KaplanMeierFitter]:
    """
    Fit Kaplan-Meier curves by group.
    
    Args:
        df: Dataframe with tenure and event columns
        group_col: Column name to group by
    
    Returns:
        dict: Dictionary of fitted models by group
    """
    logger.info(f"Fitting Kaplan-Meier curves by {group_col}...")
    
    kmf_dict = {}
    groups = df[group_col].unique()
    
    for group in groups:
        group_df = df[df[group_col] == group]
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group_df['tenure'],
            event_observed=group_df['event'],
            label=str(group)
        )
        kmf_dict[str(group)] = kmf
        
        median = kmf.median_survival_time_
        logger.info(f"  {group}: median survival = {median:.1f} months, n={len(group_df)}")
    
    # Log-rank test
    result = multivariate_logrank_test(
        df['tenure'],
        df[group_col],
        df['event']
    )
    logger.info(f"Log-rank test p-value: {result.p_value:.4f}")
    if result.p_value < 0.05:
        logger.info(f"  → Significant difference in survival between {group_col} groups!")
    
    return kmf_dict


def fit_cox_model(df: pd.DataFrame) -> CoxPHFitter:
    """
    Fit Cox Proportional Hazards model.
    
    Args:
        df: Dataframe with tenure, event, and covariates
    
    Returns:
        CoxPHFitter: Fitted model
    """
    logger.info("Fitting Cox Proportional Hazards model...")
    
    # Prepare data for Cox model - select non-collinear features
    cox_data = df.copy()
    
    # Encode categorical variables
    cox_data = pd.get_dummies(
        cox_data,
        columns=['contract', 'internet_service'],
        drop_first=True
    )
    
    # Select key features (avoid multicollinearity)
    # Don't include is_month_to_month since it's encoded in contract dummies
    feature_cols = [
        'monthly_charges',
        'service_count',
        'has_autopay',
        'potential_complaint_flag',
        'contract_One year',
        'contract_Two year',
        'internet_service_Fiber optic',
        'internet_service_No'
    ]
    
    cox_df = cox_data[['tenure', 'event'] + feature_cols].copy()
    
    # Fit model with penalization to handle any remaining collinearity
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col='tenure', event_col='event')
    
    # Model summary
    logger.info(f"Concordance index (C-index): {cph.concordance_index_:.4f}")
    logger.info(f"Log-likelihood: {cph.log_likelihood_:.2f}")
    
    # Significant coefficients
    summary = cph.summary
    significant = summary[summary['p'] < 0.05].sort_values('coef', ascending=False)
    
    logger.info("\nSignificant predictors (p < 0.05):")
    for idx, row in significant.iterrows():
        hazard_ratio = np.exp(row['coef'])
        logger.info(f"  {idx}: HR={hazard_ratio:.3f}, p={row['p']:.4f}")
    
    # Check proportional hazards assumption
    try:
        cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
        logger.info("Proportional hazards assumption checked")
    except Exception as e:
        logger.warning(f"Could not check PH assumption: {e}")
    
    return cph


def predict_customer_survival(
    cph: CoxPHFitter,
    df: pd.DataFrame,
    timepoints: list = [12, 24, 36, 48, 60]
) -> pd.DataFrame:
    """
    Predict survival probabilities for each customer.
    
    Args:
        cph: Fitted Cox model
        df: Original dataframe with customer data
        timepoints: List of timepoints to predict survival at
    
    Returns:
        pd.DataFrame: Survival predictions
    """
    logger.info("Generating per-customer survival predictions...")
    
    # Prepare data (same encoding as training)
    cox_data = df.copy()
    cox_data = pd.get_dummies(
        cox_data,
        columns=['contract', 'internet_service'],
        drop_first=True
    )
    
    # Ensure all columns from training are present
    for col in cph.params_.index:
        if col not in cox_data.columns:
            cox_data[col] = 0
    
    # Select features used in training
    feature_cols = cph.params_.index.tolist()
    X = cox_data[feature_cols]
    
    # Predict survival function
    survival_funcs = cph.predict_survival_function(X)
    
    # Extract survival at specific timepoints
    predictions = pd.DataFrame({
        'customer_id': df['customer_id'].values
    })
    
    for t in timepoints:
        # Get survival probability at time t
        if t <= survival_funcs.index.max():
            closest_idx = survival_funcs.index.get_indexer([t], method='nearest')[0]
            predictions[f'survival_{t}m'] = survival_funcs.iloc[closest_idx].values
        else:
            # Extrapolate with last known value
            predictions[f'survival_{t}m'] = survival_funcs.iloc[-1].values
    
    # Calculate median survival time
    median_survival = []
    for i in range(len(survival_funcs.columns)):
        sf = survival_funcs.iloc[:, i]
        # Find first time where survival < 0.5
        below_half = sf[sf < 0.5]
        if len(below_half) > 0:
            median_survival.append(float(below_half.index[0]))
        else:
            median_survival.append(float(sf.index[-1]))  # Censored - use max observed
    
    predictions['median_survival_months'] = median_survival
    
    # Calculate expected remaining lifetime (restricted mean up to max observed time)
    max_time = min(72, df['tenure'].max())  # Cap at 72 months (6 years)
    expected_lifetime = []
    for i in range(len(survival_funcs.columns)):
        sf = survival_funcs.iloc[:, i]
        # Restrict to observed time range
        sf_restricted = sf[sf.index <= max_time]
        if len(sf_restricted) > 0:
            # Calculate area under curve (expected lifetime)
            expected_lifetime.append(float(np.trapezoid(sf_restricted.values, sf_restricted.index)))
        else:
            expected_lifetime.append(float(max_time))
    
    predictions['expected_lifetime_months'] = expected_lifetime
    
    logger.info(f"Generated predictions for {len(predictions)} customers")
    logger.info(f"Average expected lifetime: {predictions['expected_lifetime_months'].mean():.1f} months")
    logger.info(f"Average median survival: {predictions['median_survival_months'].mean():.1f} months")
    
    return predictions


def save_survival_predictions(predictions: pd.DataFrame) -> None:
    """
    Save survival predictions to database.
    
    Args:
        predictions: Survival predictions dataframe
    """
    logger.info("Saving survival predictions to database...")
    
    # Update predictions table with survival metrics
    with get_psycopg2_connection() as conn:
        with conn.cursor() as cur:
            for _, row in predictions.iterrows():
                update_sql = """
                    UPDATE predictions
                    SET 
                        median_survival_months = %s,
                        expected_lifetime_months = %s,
                        survival_12m = %s,
                        survival_24m = %s,
                        survival_36m = %s
                    WHERE customer_id = %s
                """
                cur.execute(update_sql, (
                    float(row['median_survival_months']),
                    float(row['expected_lifetime_months']),
                    float(row.get('survival_12m', None)) if 'survival_12m' in row else None,
                    float(row.get('survival_24m', None)) if 'survival_24m' in row else None,
                    float(row.get('survival_36m', None)) if 'survival_36m' in row else None,
                    row['customer_id']
                ))
        conn.commit()
    
    logger.info(f"Updated {len(predictions)} customer survival predictions")


def main():
    """
    Main survival analysis pipeline.
    """
    logger.info("=" * 80)
    logger.info("STARTING SURVIVAL ANALYSIS PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load data
    df = load_data_for_survival()
    logger.info(f"\nStep 1 complete: Loaded {len(df)} customers")
    
    # Step 2: Overall Kaplan-Meier
    kmf_overall = kaplan_meier_overall(df)
    logger.info(f"\nStep 2 complete: Overall KM curve fitted")
    
    # Step 3: Kaplan-Meier by contract type
    kmf_contract = kaplan_meier_by_group(df, 'contract')
    logger.info(f"\nStep 3 complete: KM curves by contract type")
    
    # Step 4: Kaplan-Meier by internet service
    kmf_internet = kaplan_meier_by_group(df, 'internet_service')
    logger.info(f"\nStep 4 complete: KM curves by internet service")
    
    # Step 5: Fit Cox Proportional Hazards model
    cph = fit_cox_model(df)
    logger.info(f"\nStep 5 complete: Cox model fitted (C-index: {cph.concordance_index_:.4f})")
    
    # Step 6: Save Cox model
    joblib.dump(cph, SURVIVAL_MODEL_PATH)
    logger.info(f"Cox model saved to {SURVIVAL_MODEL_PATH}")
    logger.info(f"\nStep 6 complete: Model saved")
    
    # Step 7: Generate per-customer predictions
    predictions = predict_customer_survival(cph, df)
    logger.info(f"\nStep 7 complete: Customer survival predictions generated")
    
    # Step 8: Save predictions to database
    save_survival_predictions(predictions)
    logger.info(f"\nStep 8 complete: Predictions saved to database")
    
    logger.info("=" * 80)
    logger.info("SURVIVAL ANALYSIS PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nFINAL RESULTS:")
    logger.info(f"  Cox model C-index: {cph.concordance_index_:.4f}")
    logger.info(f"  Average expected lifetime: {predictions['expected_lifetime_months'].mean():.1f} months")
    logger.info(f"  Median survival (overall): {kmf_overall.median_survival_time_:.1f} months")
    
    return cph, predictions


if __name__ == "__main__":
    cph, predictions = main()
    print(f"\nSurvival analysis complete!")
    print(f"  C-index: {cph.concordance_index_:.4f}")
    print(f"  Avg expected lifetime: {predictions['expected_lifetime_months'].mean():.1f} months")
