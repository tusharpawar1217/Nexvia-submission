"""
Churn classification model using LightGBM with SHAP explanations.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve
)
import shap
import joblib
import logging
from typing import Tuple, Dict
import matplotlib.pyplot as plt

from config import RANDOM_SEED, TEST_SIZE, VAL_SIZE, CHURN_MODEL_PATH, CATEGORICAL_FEATURES
from db import execute_query, load_dataframe_to_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_for_modeling() -> pd.DataFrame:
    """
    Load customer data with features from database.
    
    Returns:
        pd.DataFrame: Complete dataset with features
    """
    logger.info("Loading data from database...")
    query = """
        SELECT 
            c.customer_id,
            c.gender,
            c.senior_citizen,
            c.partner,
            c.dependents,
            c.tenure,
            c.phone_service,
            c.multiple_lines,
            c.internet_service,
            c.online_security,
            c.online_backup,
            c.device_protection,
            c.tech_support,
            c.streaming_tv,
            c.streaming_movies,
            c.contract,
            c.paperless_billing,
            c.payment_method,
            c.monthly_charges,
            c.total_charges,
            c.churn,
            f.tenure_bucket,
            f.service_count,
            f.charges_per_tenure_month,
            f.charge_change_ratio,
            f.has_autopay,
            f.is_month_to_month,
            f.potential_complaint_flag
        FROM customers c
        JOIN features f ON c.customer_id = f.customer_id
    """
    df = execute_query(query)
    logger.info(f"Loaded {len(df)} customers")
    logger.info(f"Churn rate: {df['churn'].mean():.2%}")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Prepare features for modeling.
    
    Args:
        df: Raw dataframe
    
    Returns:
        tuple: (X, y, feature_names)
    """
    logger.info("Preparing features...")
    
    # Define feature columns
    feature_cols = [
        # Demographics
        'senior_citizen', 'partner', 'dependents',
        # Tenure
        'tenure',
        # Services
        'phone_service', 'internet_service', 'online_security',
        'online_backup', 'device_protection', 'tech_support',
        'streaming_tv', 'streaming_movies', 'service_count',
        # Contract & Payment
        'contract', 'paperless_billing', 'payment_method',
        'has_autopay', 'is_month_to_month',
        # Financial
        'monthly_charges', 'total_charges',
        'charges_per_tenure_month', 'charge_change_ratio',
        # Flags
        'potential_complaint_flag'
    ]
    
    # Select features that exist in dataframe
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].copy()
    y = df['churn'].copy()
    
    # Encode categorical variables
    categorical_cols = [
        'partner', 'dependents', 'phone_service', 'internet_service',
        'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies',
        'contract', 'paperless_billing', 'payment_method'
    ]
    
    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns and X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    logger.info(f"Features prepared: {len(available_features)} features")
    logger.info(f"Feature names: {available_features}")
    
    return X, y, available_features


def split_data(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Split data into train/validation/test sets with stratification.
    
    Args:
        X: Features
        y: Target
    
    Returns:
        dict: Dictionary with train/val/test splits
    """
    logger.info("Splitting data...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )
    
    # Second split: separate train and validation
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=RANDOM_SEED
    )
    
    logger.info(f"Train set: {len(X_train)} samples ({y_train.mean():.2%} churn)")
    logger.info(f"Validation set: {len(X_val)} samples ({y_val.mean():.2%} churn)")
    logger.info(f"Test set: {len(X_test)} samples ({y_test.mean():.2%} churn)")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def train_lightgbm(data_splits: Dict) -> lgb.Booster:
    """
    Train LightGBM model with optimized hyperparameters.
    
    Args:
        data_splits: Dictionary with train/val/test splits
    
    Returns:
        lgb.Booster: Trained model
    """
    logger.info("Training LightGBM model...")
    
    # Prepare datasets
    train_data = lgb.Dataset(
        data_splits['X_train'],
        label=data_splits['y_train']
    )
    val_data = lgb.Dataset(
        data_splits['X_val'],
        label=data_splits['y_val'],
        reference=train_data
    )
    
    # Handle class imbalance
    scale_pos_weight = (len(data_splits['y_train']) - data_splits['y_train'].sum()) / data_splits['y_train'].sum()
    
    # Hyperparameters (tuned for binary classification with imbalanced data)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_SEED,
        'verbose': -1
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    
    logger.info(f"Training complete. Best iteration: {model.best_iteration}")
    
    return model


def evaluate_model(model: lgb.Booster, data_splits: Dict, feature_names: list) -> Dict:
    """
    Evaluate model performance on all splits.
    
    Args:
        model: Trained LightGBM model
        data_splits: Dictionary with train/val/test splits
        feature_names: List of feature names
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model...")
    
    results = {}
    
    for split_name in ['train', 'val', 'test']:
        X = data_splits[f'X_{split_name}']
        y_true = data_splits[f'y_{split_name}']
        
        # Predictions
        y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        results[split_name] = metrics
        
        logger.info(f"\n{split_name.upper()} SET METRICS:")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    results['feature_importance'] = feature_importance.to_dict('records')
    
    return results


def save_model_and_predictions(
    model: lgb.Booster,
    df_original: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    results: Dict
) -> None:
    """
    Save model and predictions to disk and database.
    
    Args:
        model: Trained model
        df_original: Original dataframe with customer_id
        X: Features used for training
        y: True labels
        results: Evaluation results
    """
    logger.info("Saving model and predictions...")
    
    # Save model
    joblib.dump(model, CHURN_MODEL_PATH)
    logger.info(f"Model saved to {CHURN_MODEL_PATH}")
    
    # Generate predictions for all customers
    y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Create predictions dataframe
    predictions = pd.DataFrame({
        'customer_id': df_original['customer_id'],
        'churn_probability': y_pred_proba,
        'churn_prediction': y_pred,
        'churn_model_version': 'lightgbm_v1.0'
    })
    
    # Truncate existing predictions table first
    from db import get_psycopg2_connection
    try:
        with get_psycopg2_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE predictions;")
            conn.commit()
        logger.info("Truncated existing predictions table")
    except Exception as e:
        logger.warning(f"Could not truncate predictions table: {e}")
    
    # Save to database (append to empty table)
    load_dataframe_to_db(predictions, 'predictions', if_exists='append', index=False)
    logger.info(f"Predictions saved to database: {len(predictions)} records")


def generate_shap_explanations(model: lgb.Booster, X: pd.DataFrame, sample_size: int = 100) -> shap.Explainer:
    """
    Generate SHAP explanations for model.
    
    Args:
        model: Trained model
        X: Features
        sample_size: Number of samples for SHAP background
    
    Returns:
        shap.Explainer: SHAP explainer object
    """
    logger.info("Generating SHAP explanations...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample (full dataset can be slow)
    sample_idx = np.random.RandomState(RANDOM_SEED).choice(len(X), min(sample_size, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]
    
    shap_values = explainer.shap_values(X_sample)
    
    logger.info("SHAP explanations generated")
    
    # Save explainer
    explainer_path = CHURN_MODEL_PATH.parent / 'shap_explainer.pkl'
    joblib.dump(explainer, explainer_path)
    logger.info(f"SHAP explainer saved to {explainer_path}")
    
    return explainer


def main():
    """
    Main churn modeling pipeline.
    """
    logger.info("=" * 80)
    logger.info("STARTING CHURN MODELING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load data
    df = load_data_for_modeling()
    logger.info(f"\nStep 1 complete: Loaded {len(df)} customers")
    
    # Step 2: Prepare features
    X, y, feature_names = prepare_features(df)
    logger.info(f"\nStep 2 complete: Prepared {len(feature_names)} features")
    
    # Step 3: Split data
    data_splits = split_data(X, y)
    logger.info(f"\nStep 3 complete: Data split into train/val/test")
    
    # Step 4: Train model
    model = train_lightgbm(data_splits)
    logger.info(f"\nStep 4 complete: Model trained")
    
    # Step 5: Evaluate model
    results = evaluate_model(model, data_splits, feature_names)
    logger.info(f"\nStep 5 complete: Model evaluated")
    
    # Step 6: Save model and predictions
    save_model_and_predictions(model, df, X, y, results)
    logger.info(f"\nStep 6 complete: Model and predictions saved")
    
    # Step 7: Generate SHAP explanations
    explainer = generate_shap_explanations(model, X)
    logger.info(f"\nStep 7 complete: SHAP explanations generated")
    
    logger.info("=" * 80)
    logger.info("CHURN MODELING PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nFINAL TEST SET RESULTS:")
    logger.info(f"  ROC-AUC: {results['test']['roc_auc']:.4f}")
    logger.info(f"  PR-AUC: {results['test']['pr_auc']:.4f}")
    logger.info(f"  F1 Score: {results['test']['f1']:.4f}")
    
    return model, results, explainer


if __name__ == "__main__":
    model, results, explainer = main()
    print(f"\nChurn model training complete!")
    print(f"  Test ROC-AUC: {results['test']['roc_auc']:.4f}")
    print(f"  Test PR-AUC: {results['test']['pr_auc']:.4f}")
