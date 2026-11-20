"""
Generate figures for documentation and portfolio.

Creates:
1. Model comparison chart
2. Sample forecast visualization
3. Prediction interval coverage
4. Feature importance
5. Pipeline architecture diagram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

FIGURES_DIR = Path('reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("GENERATING FIGURES FOR PORTFOLIO")
print("=" * 60)


# ============================================================================
# Figure 1: Model Comparison
# ============================================================================
def create_model_comparison():
    """Bar chart comparing all models."""
    print("\n1. Creating model comparison chart...")
    
    models = ['Naive', 'Seasonal\nNaive', 'Moving\nAverage', 'LightGBM']
    wape = [0.9993, 0.9700, 0.7571, 0.6595]
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, wape, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, wape):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Highlight best model
    bars[-1].set_edgecolor('#27ae60')
    bars[-1].set_linewidth(3)
    
    ax.set_ylabel('WAPE (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison\nLightGBM achieves 13% improvement over best baseline',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add improvement annotation
    ax.annotate('13% improvement\nover best baseline',
                xy=(3, 0.6595), xytext=(2.5, 0.85),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: model_comparison.png")
    plt.close()


# ============================================================================
# Figure 2: Sample Forecast with Prediction Intervals
# ============================================================================
def create_forecast_visualization():
    """Forecast visualization with prediction intervals."""
    print("\n2. Creating forecast visualization...")
    
    try:
        # Load actual forecast data
        forecasts = pd.read_parquet('models/lgbm_forecasts.parquet')
        
        # Get a sample series (FOODS category)
        sample_series = forecasts[forecasts['id'].str.startswith('FOODS')]['id'].iloc[0]
        data = forecasts[forecasts['id'] == sample_series].sort_values('day_num')
        
        # Replace None with 0 for plotting
        for col in ['actual', 'q05', 'q25', 'q50', 'q75', 'q95']:
            data[col] = data[col].fillna(0)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = range(len(data))
        
        # Plot prediction intervals
        ax.fill_between(x, data['q05'], data['q95'], alpha=0.2, color='blue', label='90% PI')
        ax.fill_between(x, data['q25'], data['q75'], alpha=0.3, color='blue', label='50% PI')
        
        # Plot forecast and actual
        ax.plot(x, data['q50'], 'b--', linewidth=2, label='Forecast (Median)')
        ax.plot(x, data['actual'], 'k-', linewidth=2, marker='o', markersize=4, 
                label='Actual', alpha=0.7)
        
        ax.set_xlabel('Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales', fontsize=12, fontweight='bold')
        ax.set_title(f'28-Day Forecast with Prediction Intervals\n{sample_series}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add coverage text
        actual = data['actual'].values
        q05 = data['q05'].values
        q95 = data['q95'].values
        coverage = np.mean((actual >= q05) & (actual <= q95)) * 100
        
        ax.text(0.02, 0.98, f'90% PI Coverage: {coverage:.1f}%\nTarget: 90%',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'forecast_example.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: forecast_example.png")
        plt.close()
        
    except Exception as e:
        print(f"   ⚠️  Could not create forecast visualization: {e}")


# ============================================================================
# Figure 3: Metrics Summary Table
# ============================================================================
def create_metrics_table():
    """Create metrics comparison table as image."""
    print("\n3. Creating metrics summary table...")
    
    # Data
    metrics_data = {
        'Model': ['Naive', 'Seasonal Naive', 'Moving Average', 'LightGBM'],
        'WAPE': [0.9993, 0.9700, 0.7571, 0.6595],
        'RMSE': [2.8894, 3.4518, 2.0337, 1.8763],
        'MAE': [1.3344, 1.2953, 1.0110, 0.9124],
        'Coverage 90%': ['N/A', 'N/A', 'N/A', '94.52%']
    }
    
    df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style best model row
    for i in range(len(df.columns)):
        table[(4, i)].set_facecolor('#d5f4e6')
        table[(4, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(1, 4):
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    plt.title('Model Performance Metrics Comparison', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'metrics_table.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: metrics_table.png")
    plt.close()


# ============================================================================
# Figure 4: Data Processing Pipeline
# ============================================================================
def create_pipeline_diagram():
    """Visual representation of data pipeline."""
    print("\n4. Creating pipeline diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define stages
    stages = [
        {'y': 12.5, 'title': 'M5 Raw Data', 'desc': '30,490 series × 1,941 days\n5 CSV files', 'color': '#e74c3c'},
        {'y': 10.5, 'title': 'Data Preparation', 'desc': '41,426,763 rows\nFilter, melt, join', 'color': '#e67e22'},
        {'y': 8.5, 'title': 'Feature Engineering', 'desc': '47 features\nLags, rolling, price, calendar', 'color': '#f39c12'},
        {'y': 6.5, 'title': 'Model Training', 'desc': 'Baselines + LightGBM\n5 quantile models', 'color': '#27ae60'},
        {'y': 4.5, 'title': 'Prediction Intervals', 'desc': 'Q05, Q25, Q50, Q75, Q95\n94.52% coverage', 'color': '#3498db'},
        {'y': 2.5, 'title': 'API & Dashboard', 'desc': 'FastAPI + Plotly Dash\n5 endpoints, 3 tabs', 'color': '#9b59b6'},
    ]
    
    # Draw stages
    for stage in stages:
        # Box
        rect = plt.Rectangle((1, stage['y']-0.4), 8, 0.8, 
                            facecolor=stage['color'], edgecolor='black', 
                            linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Title
        ax.text(5, stage['y']+0.1, stage['title'], 
               ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Description
        ax.text(5, stage['y']-0.2, stage['desc'], 
               ha='center', va='center', fontsize=9, color='white')
        
        # Arrow to next stage
        if stage != stages[-1]:
            ax.arrow(5, stage['y']-0.5, 0, -0.8, head_width=0.3, head_length=0.2,
                    fc='black', ec='black', linewidth=2)
    
    # Title
    ax.text(5, 13.5, 'Data Processing Pipeline', 
           ha='center', fontsize=16, fontweight='bold')
    
    # Time annotations
    times = ['30s', '8 min', '10 min', '8 min', 'included', 'deployed']
    for i, (stage, time) in enumerate(zip(stages, times)):
        ax.text(9.5, stage['y'], f'⏱️ {time}', 
               ha='left', va='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pipeline_diagram.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: pipeline_diagram.png")
    plt.close()


# ============================================================================
# Figure 5: Dataset Overview
# ============================================================================
def create_dataset_overview():
    """Dataset statistics visualization."""
    print("\n5. Creating dataset overview...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Series by Category
    categories = ['FOODS', 'HOUSEHOLD', 'HOBBIES']
    series_count = [997, 753, 384]
    colors1 = ['#3498db', '#e74c3c', '#f39c12']
    
    ax1.bar(categories, series_count, color=colors1, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Series (10% sample)', fontweight='bold')
    ax1.set_title('Series Distribution by Category', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(series_count):
        ax1.text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Pipeline Timing
    stages = ['Validation', 'Data Prep', 'Features', 'Baselines', 'LightGBM']
    times = [0.5, 477, 631, 22, 474]
    colors2 = ['#27ae60', '#3498db', '#e67e22', '#9b59b6', '#e74c3c']
    
    ax2.barh(stages, times, color=colors2, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Time (seconds)', fontweight='bold')
    ax2.set_title('Pipeline Execution Time', fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(times):
        ax2.text(v + 10, i, f'{v}s', va='center', fontweight='bold')
    
    # 3. Model Performance
    models = ['Naive', 'S-Naive', 'MA', 'LightGBM']
    improvements = [32.0, 27.1, 14.8, 0.0]  # % worse than LightGBM
    colors3 = ['#e74c3c', '#e74c3c', '#e67e22', '#27ae60']
    
    bars = ax3.barh(models, improvements, color=colors3, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('WAPE % Worse than LightGBM', fontweight='bold')
    ax3.set_title('Model Performance Gap', fontweight='bold', fontsize=12)
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_xaxis()
    for i, v in enumerate(improvements):
        if v > 0:
            ax3.text(v - 1, i, f'+{v:.1f}%', va='center', ha='right', fontweight='bold')
        else:
            ax3.text(1, i, 'BEST', va='center', ha='left', fontweight='bold', color='white')
    
    # 4. Data Volume
    data_stages = ['Raw CSV', 'Processed', 'Features', 'Models']
    sizes_mb = [120, 288, 1040, 3]
    colors4 = ['#95a5a6', '#3498db', '#e67e22', '#27ae60']
    
    ax4.bar(data_stages, sizes_mb, color=colors4, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Size (MB)', fontweight='bold')
    ax4.set_title('Data Volume at Each Stage', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    for i, v in enumerate(sizes_mb):
        ax4.text(i, v + 20, f'{v} MB', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Project Dataset Overview', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'dataset_overview.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: dataset_overview.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    """Generate all figures."""
    
    # Create all figures
    create_model_comparison()
    create_forecast_visualization()
    create_metrics_table()
    create_pipeline_diagram()
    create_dataset_overview()
    
    print("\n" + "=" * 60)
    print("✅ ALL FIGURES GENERATED!")
    print("=" * 60)
    print(f"\nSaved to: {FIGURES_DIR.absolute()}")
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  - {f.name}")
    print("\n💡 Use these figures in:")
    print("  - README.md")
    print("  - Portfolio presentations")
    print("  - GitHub repository")
    print("  - Job applications")
    print("=" * 60)


if __name__ == "__main__":
    main()
