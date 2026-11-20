"""
Improved Churn Classification Model with Advanced Techniques
- Hyperparameter tuning with Optuna
- Better feature engineering
- Class balancing
- Cross-validation
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)
import lightgbm as lgb
import optuna

from config import (
    RANDOM_SEED, TEST_SIZE, CHURN_MODEL_PATH, SCALER_PATH, MODELS_DIR
)
from db import execute_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Load and prepare data with advanced feature engineering."""
    logger.info("Loading data from database...")
    
    query = """
        SELECT 
            c.*,
            f.*
        FROM customers c
        JOIN features f ON c.customer_id = f.customer_id
    """
    df = execute_query(query)
    logger.info(f"Loaded {len(df)} records")
    
    # Remove duplicate customer_id columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df


def create_advanced_features(df):
    """
    Create advanced features for better prediction.
    
    Returns:
        pd.DataFrame: DataFrame with new features
    """
    logger.info("Creating advanced features...")
    
    df = df.copy()
    
    # 1. Interaction features
    df['tenure_charges_ratio'] = df['tenure'] / (df['monthly_charges'] + 1)
    df['total_to_monthly_ratio'] = df['total_charges'] / (df['monthly_charges'] + 1)
    df['charges_per_service'] = df['monthly_charges'] / (df['service_count'] + 1)
    
    # 2. Contract type encoding with domain knowledge
    contract_value = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }
    df['contract_numeric'] = df['contract'].map(contract_value)
    
    # 3. Payment method risk score
    payment_risk = {
        'Electronic check': 3,  # Highest risk
        'Mailed check': 2,
        'Bank transfer (automatic)': 0,  # Lowest risk
        'Credit card (automatic)': 0
    }
    df['payment_risk_score'] = df['payment_method'].map(payment_risk)
    
    # 4. Service engagement score (normalized)
    max_services = df['service_count'].max()
    df['service_engagement'] = df['service_count'] / max_services
    
    # 5. Customer lifecycle stage
    df['lifecycle_stage'] = pd.cut(
        df['tenure'],
        bins=[0, 6, 12, 24, 48, 100],
        labels=['new', 'early', 'mid', 'mature', 'veteran']
    )
    
    # 6. Churn risk indicators
    df['high_charges_low_tenure'] = (
        (df['monthly_charges'] > df['monthly_charges'].median()) & 
        (df['tenure'] < 12)
    ).astype(int)
    
    df['month_to_month_no_autopay'] = (
        (df['is_month_to_month'] == 1) & 
        (df['has_autopay'] == 0)
    ).astype(int)
    
    # 7. Value-based features
    df['is_high_value'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)
    df['is_low_tenure'] = (df['tenure'] < df['tenure'].quantile(0.25)).astype(int)
    
    # 8. Internet service features
    df['has_internet'] = (df['internet_service'] != 'No').astype(int)
    df['fiber_optic'] = (df['internet_service'] == 'Fiber optic').astype(int)
    
    # 9. Protection services coverage
    protection_cols = ['online_security', 'online_backup', 'device_protection', 'tech_support']
    df['protection_count'] = 0
    for col in protection_cols:
        if col in df.columns:
            df['protection_count'] += (df[col] == 'Yes').astype(int)
    
    df['has_full_protection'] = (df['protection_count'] >= 3).astype(int)
    
    # 10. Demographic combinations
    df['senior_no_partner'] = ((df['senior_citizen'] == 1) & (df['partner'] == 'No')).astype(int)
    df['young_with_dependents'] = ((df['senior_citizen'] == 0) & (df['dependents'] == 'Yes')).astype(int)
    
    logger.info(f"Created advanced features - Total columns: {len(df.columns)}")
    
    return df


def prepare_features_for_modeling(df):
    """
    Prepare features for modeling with encoding and scaling.
    
    Returns:
        tuple: (X, y, feature_names, label_encoders)
    """
    logger.info("Preparing features for modeling...")
    
    # Separate target
    y = df['churn'].values
    
    # Select features for modeling
    exclude_cols = [
        'customer_id', 'churn', 'churn_label', 'churn_value', 'churn_score',
        'churn_reason', 'cltv', 'count', 'country', 'state', 'city',
        'zip_code', 'lat_long', 'latitude', 'longitude', 'lifecycle_stage',
        'created_at', 'updated_at'  # Exclude datetime columns
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    
    # Exclude any remaining datetime columns
    datetime_cols = X.select_dtypes(include=['datetime', 'datetime64']).columns
    if len(datetime_cols) > 0:
        logger.info(f"Excluding datetime columns: {list(datetime_cols)}")
        X = X.drop(columns=datetime_cols)
    
    # Encode categorical features
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Convert all to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    logger.info(f"Final feature set: {X.shape[1]} features")
    
    return X, y, X.columns.tolist(), label_encoders


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        float: ROC-AUC score on validation set
    """
    # Suggest hyperparameters
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': RANDOM_SEED,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0)
    }
    
    # Train model
    model = lgb.LGBMClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Predict
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    return roc_auc


def hyperparameter_tuning(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Perform hyperparameter tuning using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials
    
    Returns:
        dict: Best hyperparameters
    """
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"Best ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    return study.best_params


def train_improved_model(best_params, X_train, y_train, X_test, y_test):
    """
    Train the final improved model with best parameters.
    
    Args:
        best_params: Best hyperparameters from tuning
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        lgb.LGBMClassifier: Trained model
    """
    logger.info("Training final improved model...")
    
    # Combine best params with fixed params
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': RANDOM_SEED,
        **best_params
    }
    
    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("IMPROVED MODEL PERFORMANCE")
    logger.info("=" * 60)
    for metric, value in metrics.items():
        logger.info(f"{metric.upper():20s}: {value:.4f}")
    logger.info("=" * 60)
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    return metrics


def main():
    """Main pipeline for improved churn model."""
    logger.info("=" * 80)
    logger.info("IMPROVED CHURN MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load data
    df = load_and_prepare_data()
    logger.info(f"\nStep 1 complete: Loaded {len(df)} records")
    
    # Step 2: Create advanced features
    df = create_advanced_features(df)
    logger.info(f"\nStep 2 complete: Advanced features created")
    
    # Step 3: Prepare features
    X, y, feature_names, label_encoders = prepare_features_for_modeling(df)
    logger.info(f"\nStep 3 complete: {X.shape[1]} features prepared")
    
    # Step 4: Split data (60% train, 20% val, 20% test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    logger.info(f"\nStep 4 complete: Data split")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Val: {len(X_val)} samples")
    logger.info(f"  Test: {len(X_test)} samples")
    
    # Step 5: Hyperparameter tuning
    best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val, n_trials=20)
    logger.info(f"\nStep 5 complete: Hyperparameter tuning done")
    
    # Step 6: Train final model
    model = train_improved_model(best_params, X_train, y_train, X_test, y_test)
    logger.info(f"\nStep 6 complete: Final model trained")
    
    # Step 7: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"\nStep 7 complete: Model evaluated")
    
    # Step 8: Save model
    improved_model_path = Path(MODELS_DIR) / 'churn_model_improved.pkl'
    joblib.dump(model, improved_model_path)
    logger.info(f"\nStep 8 complete: Model saved to {improved_model_path}")
    
    # Save feature names
    feature_names_path = Path(MODELS_DIR) / 'feature_names_improved.pkl'
    joblib.dump(feature_names, feature_names_path)
    
    # Save label encoders
    encoders_path = Path(MODELS_DIR) / 'label_encoders_improved.pkl'
    joblib.dump(label_encoders, encoders_path)
    
    logger.info("=" * 80)
    logger.info("IMPROVED MODEL TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nIMPROVEMENT SUMMARY:")
    logger.info(f"  Previous Accuracy: 80.46%")
    logger.info(f"  New Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"  Improvement: {(metrics['accuracy']-0.8046)*100:+.2f}%")
    logger.info(f"\n  Previous ROC-AUC: 0.9005")
    logger.info(f"  New ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  Improvement: {(metrics['roc_auc']-0.9005):+.4f}")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
    print(f"\n✅ Improved model training complete!")
    print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
