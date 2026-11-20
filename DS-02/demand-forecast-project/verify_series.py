import pandas as pd
import requests

API_BASE = "http://127.0.0.1:8000"

# Load forecast data
forecasts = pd.read_parquet('models/lgbm_forecasts.parquet')
available_series = set(forecasts['id'].unique())

print("Testing series from each category:")
print("="*60)

test_series = {
    'HOBBIES_1_006_CA_1_evaluation': 'HOBBIES',
    'HOUSEHOLD_1_024_CA_1_evaluation': 'HOUSEHOLD',
    'FOODS_1_001_TX_1_evaluation': 'FOODS'
}

for series_id, category in test_series.items():
    print(f"\n{category}: {series_id}")
    
    # Check if in forecast data
    if series_id in available_series:
        print(f"  ✅ Found in forecast data")
        
        # Test API
        try:
            response = requests.get(f"{API_BASE}/forecast/{series_id}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ API works - {data['forecast_days']} days")
            else:
                print(f"  ❌ API error: {response.status_code}")
                print(f"     Response: {response.text[:100]}")
        except Exception as e:
            print(f"  ❌ API exception: {e}")
    else:
        print(f"  ❌ NOT in forecast data")
        # Find a similar one that exists
        similar = [s for s in available_series if category in s][:3]
        print(f"  💡 Try instead: {similar}")

print("\n" + "="*60)
print(f"\nRecommended series to use:")
print("  FOODS: FOODS_1_001_TX_1_evaluation")
print("  HOUSEHOLD: HOUSEHOLD_1_002_CA_4_evaluation")  
print("  HOBBIES: HOBBIES_1_001_TX_2_evaluation")
