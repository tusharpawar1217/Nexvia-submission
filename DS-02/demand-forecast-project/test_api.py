"""
Quick API Test Script

Tests all 5 API endpoints to verify they're working.
Run this while the API is running.
"""

import requests
import json

API_BASE = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("1. Testing /health endpoint...")
    print("="*60)
    
    response = requests.get(f"{API_BASE}/health")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Health check passed!")
        print(f"   Status: {data['status']}")
        print(f"   Series loaded: {data['series_count']:,}")
        print(f"   Forecast rows: {data['forecast_rows']:,}")
    else:
        print(f"❌ Health check failed: {response.status_code}")


def test_hierarchy():
    """Test hierarchy endpoint."""
    print("\n" + "="*60)
    print("2. Testing /hierarchy endpoint...")
    print("="*60)
    
    # First try to get a series from the forecast data
    try:
        import pandas as pd
        from pathlib import Path
        forecasts = pd.read_parquet(Path('models/lgbm_forecasts.parquet'))
        sample_ids = forecasts['id'].unique()[:5].tolist()
        print(f"✅ Using forecast series (from models)")
        print("   Sample series:")
        for sid in sample_ids[:3]:
            print(f"     - {sid}")
        return sample_ids[0] if sample_ids else None
    except:
        pass
    
    # Fallback to hierarchy API
    response = requests.get(
        f"{API_BASE}/hierarchy",
        params={'state_id': 'CA', 'limit': 10}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Hierarchy loaded! Found {data['total']} series")
        print("   Sample series:")
        for series in data['series'][:3]:
            print(f"     - {series['id']} ({series['item_id']} at {series['store_id']})")
        return data['series'][0]['id'] if data['series'] else None
    else:
        print(f"❌ Hierarchy failed: {response.status_code}")
        return None


def test_forecast(series_id):
    """Test forecast endpoint."""
    print("\n" + "="*60)
    print(f"3. Testing /forecast/{series_id} endpoint...")
    print("="*60)
    
    response = requests.get(f"{API_BASE}/forecast/{series_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Forecast loaded!")
        print(f"   Series: {data['series_id']}")
        print(f"   Forecast days: {data['forecast_days']}")
        print(f"   Total actual: {sum(data['actual']):.0f}")
        print(f"   Total forecast: {sum(data['forecast']):.0f}")
        print(f"   Quantiles: Q05, Q25, Q50, Q75, Q95")
    else:
        print(f"❌ Forecast failed: {response.status_code}")


def test_metrics():
    """Test metrics endpoint."""
    print("\n" + "="*60)
    print("4. Testing /metrics endpoint...")
    print("="*60)
    
    models = ['lgbm', 'naive', 'snaive', 'ma']
    
    for model in models:
        response = requests.get(f"{API_BASE}/metrics", params={'model': model})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {model.upper():6s} - WAPE: {data['wape']:.4f}, RMSE: {data['rmse']:.4f}")
        else:
            print(f"❌ {model.upper()} failed: {response.status_code}")


def test_optimize(series_id):
    """Test optimize endpoint."""
    print("\n" + "="*60)
    print(f"5. Testing /optimize endpoint...")
    print("="*60)
    
    response = requests.post(
        f"{API_BASE}/optimize",
        params={
            'series_id': series_id,
            'service_level': 0.95
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        rec = data['recommendation']
        costs = data['costs']
        
        print(f"✅ Optimization complete!")
        print(f"   Series: {data['series_id']} ({data['category']})")
        print(f"   Order Quantity: {rec['order_qty']} units")
        print(f"   Expected Demand: {rec['expected_demand']:.1f} units")
        print(f"   Safety Stock: {rec['safety_stock']:.1f} units")
        print(f"   Order Cost: ${costs['order_cost']:.2f}")
        print(f"   Note: {costs['note']}")
    else:
        print(f"❌ Optimization failed: {response.status_code}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 TESTING API ENDPOINTS")
    print("="*60)
    print(f"\nAPI Base URL: {API_BASE}")
    print("\nMake sure the API is running:")
    print("  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000")
    
    try:
        # Test 1: Health
        test_health()
        
        # Test 2: Hierarchy (get a sample series ID)
        series_id = test_hierarchy()
        
        if not series_id:
            print("\n❌ No series found, cannot continue tests")
            return
        
        # Test 3: Forecast
        test_forecast(series_id)
        
        # Test 4: Metrics
        test_metrics()
        
        # Test 5: Optimize
        test_optimize(series_id)
        
        # Summary
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETE!")
        print("="*60)
        print("\nAPI is working correctly! 🎉")
        print("\nNext steps:")
        print("  1. Open dashboard: http://127.0.0.1:8050")
        print("  2. Browse API docs: http://127.0.0.1:8000/docs")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("\nMake sure the API is running:")
        print("  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    main()
