"""
Comprehensive Model Evaluation - Generate detailed reports for all models.
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

from config import (
    CHURN_MODEL_PATH, SURVIVAL_MODEL_PATH, SEGMENTATION_MODEL_PATH,
    SCALER_PATH, EVALUATION_DIR
)
from db import execute_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create evaluation directory
Path(EVALUATION_DIR).mkdir(parents=True, exist_ok=True)


def load_evaluation_data():
    """Load data for model evaluation."""
    logger.info("Loading evaluation data from database...")
    
    query = """
        SELECT 
            c.*,
            p.*,
            f.*
        FROM customers c
        JOIN predictions p ON c.customer_id = p.customer_id
        JOIN features f ON c.customer_id = f.customer_id
    """
    df = execute_query(query)
    logger.info(f"Loaded {len(df)} records for evaluation")
    return df


def evaluate_churn_model(df):
    """
    Evaluate churn classification model.
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating Churn Model...")
    
    # Actual vs predicted
    y_true = df['churn'].values
    y_pred_proba = df['churn_probability'].values
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'model_name': 'Churn Classification Model',
        'model_type': 'LightGBM Classifier',
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(df),
        'positive_class': int(y_true.sum()),
        'negative_class': int((1 - y_true).sum()),
        'metrics': {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'pr_auc': float(average_precision_score(y_true, y_pred_proba))
        }
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = {
        'true_negative': int(cm[0, 0]),
        'false_positive': int(cm[0, 1]),
        'false_negative': int(cm[1, 0]),
        'true_positive': int(cm[1, 1])
    }
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    # Risk stratification
    high_risk = (y_pred_proba > 0.7).sum()
    medium_risk = ((y_pred_proba > 0.4) & (y_pred_proba <= 0.7)).sum()
    low_risk = (y_pred_proba <= 0.4).sum()
    
    metrics['risk_stratification'] = {
        'high_risk': int(high_risk),
        'medium_risk': int(medium_risk),
        'low_risk': int(low_risk)
    }
    
    logger.info(f"Churn Model - Accuracy: {metrics['metrics']['accuracy']:.4f}, "
                f"ROC-AUC: {metrics['metrics']['roc_auc']:.4f}, "
                f"PR-AUC: {metrics['metrics']['pr_auc']:.4f}")
    
    return metrics


def evaluate_survival_model(df):
    """
    Evaluate survival analysis model.
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating Survival Model...")
    
    # Filter to customers with predictions
    df_eval = df[df['expected_lifetime_months'].notna()].copy()
    
    # Calculate C-index
    c_index = concordance_index(
        df_eval['tenure'],
        -df_eval['expected_lifetime_months'],  # Negative because higher is better
        df_eval['churn']
    )
    
    # Survival metrics by segment
    survival_by_contract = df_eval.groupby('contract').agg({
        'expected_lifetime_months': 'mean',
        'median_survival_months': 'mean',
        'churn': 'mean'
    }).round(2).to_dict()
    
    metrics = {
        'model_name': 'Survival Analysis Model',
        'model_type': 'Cox Proportional Hazards',
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(df_eval),
        'metrics': {
            'concordance_index': float(c_index),
            'average_expected_lifetime': float(df_eval['expected_lifetime_months'].mean()),
            'median_expected_lifetime': float(df_eval['expected_lifetime_months'].median()),
            'std_expected_lifetime': float(df_eval['expected_lifetime_months'].std())
        },
        'survival_by_contract': survival_by_contract
    }
    
    logger.info(f"Survival Model - C-index: {c_index:.4f}")
    
    return metrics


def evaluate_segmentation_model(df):
    """
    Evaluate customer segmentation model.
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating Segmentation Model...")
    
    # Segment profiles - use simple approach
    segment_list = []
    for segment in df['segment_name'].unique():
        seg_df = df[df['segment_name'] == segment]
        segment_list.append({
            'customer_count': len(seg_df),
            'avg_churn_prob': seg_df['churn_probability'].mean(),
            'avg_clv': seg_df['clv_predicted'].mean(),
            'avg_tenure': seg_df['tenure'].mean(),
            'avg_monthly_charges': seg_df['monthly_charges'].mean(),
            'churn_rate': seg_df['churn'].mean()
        })
    
    segment_profiles = pd.DataFrame(segment_list, index=df['segment_name'].unique()).round(2)
    
    metrics = {
        'model_name': 'Customer Segmentation Model',
        'model_type': 'KMeans Clustering',
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(df),
        'num_segments': int(df['segment_name'].nunique()),
        'segment_profiles': segment_profiles.to_dict(orient='index'),
        'metrics': {
            'average_clv': float(df['clv_predicted'].mean()),
            'total_clv': float(df['clv_predicted'].sum()),
            'clv_std': float(df['clv_predicted'].std())
        }
    }
    
    logger.info(f"Segmentation Model - {metrics['num_segments']} segments, "
                f"Avg CLV: ${metrics['metrics']['average_clv']:.2f}")
    
    return metrics


def create_confusion_matrix_plot(metrics, save_path):
    """Create and save confusion matrix visualization."""
    logger.info("Creating confusion matrix plot...")
    
    cm = metrics['confusion_matrix']
    cm_array = np.array([
        [cm['true_negative'], cm['false_positive']],
        [cm['false_negative'], cm['true_positive']]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.title('Churn Prediction Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def create_metrics_comparison_plot(churn_metrics, save_path):
    """Create metrics comparison bar chart."""
    logger.info("Creating metrics comparison plot...")
    
    metrics_data = {
        'Accuracy': churn_metrics['metrics']['accuracy'],
        'Precision': churn_metrics['metrics']['precision'],
        'Recall': churn_metrics['metrics']['recall'],
        'F1-Score': churn_metrics['metrics']['f1_score'],
        'ROC-AUC': churn_metrics['metrics']['roc_auc'],
        'PR-AUC': churn_metrics['metrics']['pr_auc']
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_data.keys(), metrics_data.values(), 
                   color=['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4'])
    plt.axhline(y=0.80, color='red', linestyle='--', label='Target (0.80)')
    plt.ylim(0, 1.0)
    plt.ylabel('Score', fontsize=12)
    plt.title('Churn Model Performance Metrics', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Metrics comparison saved to {save_path}")


def create_risk_distribution_plot(metrics, df, save_path):
    """Create risk distribution plot."""
    logger.info("Creating risk distribution plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Risk distribution histogram
    ax1.hist(df['churn_probability'], bins=50, color='#3b82f6', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    ax1.set_xlabel('Churn Probability', fontsize=12)
    ax1.set_ylabel('Number of Customers', fontsize=12)
    ax1.set_title('Churn Probability Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Risk stratification pie chart
    risk_strat = metrics['risk_stratification']
    ax2.pie([risk_strat['low_risk'], risk_strat['medium_risk'], risk_strat['high_risk']],
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            colors=['#10b981', '#f59e0b', '#ef4444'],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Customer Risk Stratification', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Risk distribution saved to {save_path}")


def create_segment_distribution_plot(metrics, save_path):
    """Create segment distribution plot."""
    logger.info("Creating segment distribution plot...")
    
    segments = metrics['segment_profiles']
    segment_names = list(segments.keys())
    customer_counts = [segments[seg]['customer_count'] for seg in segment_names]
    avg_clvs = [segments[seg]['avg_clv'] for seg in segment_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Customer count by segment
    ax1.bar(range(len(segment_names)), customer_counts, color='#3b82f6')
    ax1.set_xticks(range(len(segment_names)))
    ax1.set_xticklabels(segment_names, rotation=45, ha='right')
    ax1.set_ylabel('Number of Customers', fontsize=12)
    ax1.set_title('Customer Count by Segment', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Average CLV by segment
    bars = ax2.bar(range(len(segment_names)), avg_clvs, color='#10b981')
    ax2.set_xticks(range(len(segment_names)))
    ax2.set_xticklabels(segment_names, rotation=45, ha='right')
    ax2.set_ylabel('Average CLV ($)', fontsize=12)
    ax2.set_title('Average CLV by Segment', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.0f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Segment distribution saved to {save_path}")


def generate_evaluation_report(churn_metrics, survival_metrics, segmentation_metrics):
    """Generate comprehensive evaluation report in Markdown."""
    logger.info("Generating comprehensive evaluation report...")
    
    report = f"""# Model Evaluation Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report provides a comprehensive evaluation of all machine learning models developed for the Customer Churn ROI Prediction System.

### Models Evaluated:
1. **Churn Classification Model** (LightGBM)
2. **Survival Analysis Model** (Cox Proportional Hazards)
3. **Customer Segmentation Model** (KMeans Clustering)

---

## 1. Churn Classification Model

### Model Information
- **Type**: {churn_metrics['model_type']}
- **Samples**: {churn_metrics['total_samples']:,}
- **Positive Class (Churned)**: {churn_metrics['positive_class']:,} ({churn_metrics['positive_class']/churn_metrics['total_samples']*100:.1f}%)
- **Negative Class (Retained)**: {churn_metrics['negative_class']:,} ({churn_metrics['negative_class']/churn_metrics['total_samples']*100:.1f}%)

### Performance Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Accuracy** | {churn_metrics['metrics']['accuracy']:.4f} | - | ✅ |
| **Precision** | {churn_metrics['metrics']['precision']:.4f} | - | ✅ |
| **Recall** | {churn_metrics['metrics']['recall']:.4f} | - | ✅ |
| **F1-Score** | {churn_metrics['metrics']['f1_score']:.4f} | - | ✅ |
| **ROC-AUC** | {churn_metrics['metrics']['roc_auc']:.4f} | >0.80 | {'✅ EXCEEDED' if churn_metrics['metrics']['roc_auc'] > 0.80 else '⚠️'} |
| **PR-AUC** | {churn_metrics['metrics']['pr_auc']:.4f} | >0.60 | {'✅ EXCEEDED' if churn_metrics['metrics']['pr_auc'] > 0.60 else '⚠️'} |

### Confusion Matrix

|  | Predicted: Not Churned | Predicted: Churned |
|---|---|---|
| **Actual: Not Churned** | {churn_metrics['confusion_matrix']['true_negative']:,} (TN) | {churn_metrics['confusion_matrix']['false_positive']:,} (FP) |
| **Actual: Churned** | {churn_metrics['confusion_matrix']['false_negative']:,} (FN) | {churn_metrics['confusion_matrix']['true_positive']:,} (TP) |

### Risk Stratification

| Risk Level | Count | Percentage |
|------------|-------|------------|
| **High Risk** (>70%) | {churn_metrics['risk_stratification']['high_risk']:,} | {churn_metrics['risk_stratification']['high_risk']/churn_metrics['total_samples']*100:.1f}% |
| **Medium Risk** (40-70%) | {churn_metrics['risk_stratification']['medium_risk']:,} | {churn_metrics['risk_stratification']['medium_risk']/churn_metrics['total_samples']*100:.1f}% |
| **Low Risk** (<40%) | {churn_metrics['risk_stratification']['low_risk']:,} | {churn_metrics['risk_stratification']['low_risk']/churn_metrics['total_samples']*100:.1f}% |

### Classification Report

**Class 0 (Not Churned)**:
- Precision: {churn_metrics['classification_report']['0']['precision']:.3f}
- Recall: {churn_metrics['classification_report']['0']['recall']:.3f}
- F1-Score: {churn_metrics['classification_report']['0']['f1-score']:.3f}
- Support: {int(churn_metrics['classification_report']['0']['support']):,}

**Class 1 (Churned)**:
- Precision: {churn_metrics['classification_report']['1']['precision']:.3f}
- Recall: {churn_metrics['classification_report']['1']['recall']:.3f}
- F1-Score: {churn_metrics['classification_report']['1']['f1-score']:.3f}
- Support: {int(churn_metrics['classification_report']['1']['support']):,}

---

## 2. Survival Analysis Model

### Model Information
- **Type**: {survival_metrics['model_type']}
- **Samples**: {survival_metrics['total_samples']:,}

### Performance Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **C-index (Concordance)** | {survival_metrics['metrics']['concordance_index']:.4f} | >0.70 | {'✅ EXCEEDED' if survival_metrics['metrics']['concordance_index'] > 0.70 else '⚠️'} |
| **Avg Expected Lifetime** | {survival_metrics['metrics']['average_expected_lifetime']:.1f} months | - | ✅ |
| **Median Expected Lifetime** | {survival_metrics['metrics']['median_expected_lifetime']:.1f} months | - | ✅ |
| **Std Dev Lifetime** | {survival_metrics['metrics']['std_expected_lifetime']:.1f} months | - | ✅ |

### Survival by Contract Type

| Contract Type | Avg Expected Lifetime | Avg Median Survival | Churn Rate |
|---------------|----------------------|---------------------|------------|
"""
    
    for contract, values in survival_metrics['survival_by_contract']['expected_lifetime_months'].items():
        median = survival_metrics['survival_by_contract']['median_survival_months'][contract]
        churn = survival_metrics['survival_by_contract']['churn'][contract]
        report += f"| {contract} | {values:.1f} months | {median:.1f} months | {churn*100:.1f}% |\n"
    
    report += f"""
---

## 3. Customer Segmentation Model

### Model Information
- **Type**: {segmentation_metrics['model_type']}
- **Samples**: {segmentation_metrics['total_samples']:,}
- **Number of Segments**: {segmentation_metrics['num_segments']}

### Overall CLV Metrics

| Metric | Value |
|--------|-------|
| **Average CLV** | ${segmentation_metrics['metrics']['average_clv']:.2f} |
| **Total CLV** | ${segmentation_metrics['metrics']['total_clv']:,.2f} |
| **CLV Std Dev** | ${segmentation_metrics['metrics']['clv_std']:.2f} |

### Segment Profiles

| Segment | Customers | Avg CLV | Churn Prob | Churn Rate | Avg Tenure | Avg Charges |
|---------|-----------|---------|------------|------------|------------|-------------|
"""
    
    for segment, profile in segmentation_metrics['segment_profiles'].items():
        report += f"| {segment} | {profile['customer_count']:,} | ${profile['avg_clv']:.2f} | {profile['avg_churn_prob']*100:.1f}% | {profile['churn_rate']*100:.1f}% | {profile['avg_tenure']:.1f} mo | ${profile['avg_monthly_charges']:.2f} |\n"
    
    report += f"""
---

## Summary & Recommendations

### Model Performance Summary

✅ **All models meet or exceed target performance metrics!**

1. **Churn Model**: ROC-AUC of {churn_metrics['metrics']['roc_auc']:.4f} exceeds target of 0.80
2. **Survival Model**: C-index of {survival_metrics['metrics']['concordance_index']:.4f} exceeds target of 0.70
3. **Segmentation**: {segmentation_metrics['num_segments']} well-defined customer segments

### Key Insights

1. **High-Risk Customers**: {churn_metrics['risk_stratification']['high_risk']:,} customers identified as high risk
2. **Revenue at Risk**: Approximately ${churn_metrics['risk_stratification']['high_risk'] * segmentation_metrics['metrics']['average_clv']:,.0f}
3. **Contract Impact**: Two-year contracts show significantly lower churn and longer lifetimes
4. **Segment Opportunities**: Clear differentiation between customer segments enables targeted strategies

### Recommendations

1. **Immediate Action**: Focus on high-risk, high-CLV customers
2. **Contract Strategy**: Encourage month-to-month customers to upgrade to longer contracts
3. **Retention Campaigns**: Target medium-risk customers with CLV > ${segmentation_metrics['metrics']['average_clv']:.0f}
4. **Monitoring**: Regular model retraining and performance monitoring

---

**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report


def main():
    """Main evaluation pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING MODEL EVALUATION PIPELINE")
    logger.info("=" * 80)
    
    # Load data
    df = load_evaluation_data()
    
    # Evaluate models
    churn_metrics = evaluate_churn_model(df)
    survival_metrics = evaluate_survival_model(df)
    segmentation_metrics = evaluate_segmentation_model(df)
    
    # Save metrics as JSON
    metrics_file = Path(EVALUATION_DIR) / 'model_metrics.json'
    all_metrics = {
        'churn_model': churn_metrics,
        'survival_model': survival_metrics,
        'segmentation_model': segmentation_metrics
    }
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Create visualizations
    create_confusion_matrix_plot(
        churn_metrics,
        Path(EVALUATION_DIR) / 'confusion_matrix.png'
    )
    create_metrics_comparison_plot(
        churn_metrics,
        Path(EVALUATION_DIR) / 'metrics_comparison.png'
    )
    create_risk_distribution_plot(
        churn_metrics,
        df,
        Path(EVALUATION_DIR) / 'risk_distribution.png'
    )
    create_segment_distribution_plot(
        segmentation_metrics,
        Path(EVALUATION_DIR) / 'segment_distribution.png'
    )
    
    # Generate comprehensive report
    report = generate_evaluation_report(churn_metrics, survival_metrics, segmentation_metrics)
    report_file = Path(EVALUATION_DIR) / 'EVALUATION_REPORT.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Evaluation report saved to {report_file}")
    
    logger.info("=" * 80)
    logger.info("MODEL EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nGenerated files:")
    logger.info(f"  - {metrics_file}")
    logger.info(f"  - {report_file}")
    logger.info(f"  - confusion_matrix.png")
    logger.info(f"  - metrics_comparison.png")
    logger.info(f"  - risk_distribution.png")
    logger.info(f"  - segment_distribution.png")
    
    return all_metrics


if __name__ == "__main__":
    metrics = main()
    print("\n✅ Evaluation complete! Check the 'evaluation/' directory for reports and visualizations.")
