import pandas as pd

# Load forecasts
df = pd.read_parquet('models/lgbm_forecasts.parquet')

print("Categories in forecast data:")
print("="*60)

for cat in ['FOODS', 'HOUSEHOLD', 'HOBBIES']:
    cat_series = df[df['id'].str.startswith(cat)]['id'].unique()
    print(f"\n{cat}: {len(cat_series)} series")
    if len(cat_series) > 0:
        print(f"  Examples: {cat_series[:3].tolist()}")
    else:
        print(f"  Examples: NONE - No {cat} series in forecast!")

print(f"\n{'='*60}")
print(f"Total unique series: {df['id'].nunique()}")
print(f"Total forecast rows: {len(df)}")
