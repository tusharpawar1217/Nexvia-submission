"""
Quick setup validation script.
Tests database connection and environment configuration.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required packages are installed."""
    print("Testing package imports...")
    try:
        import pandas
        import numpy
        import psycopg2
        import sqlalchemy
        import lightgbm
        import sklearn
        from dotenv import load_dotenv
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False


def test_env_file():
    """Test that .env file exists and has required variables."""
    print("\nTesting .env configuration...")
    env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        print("✗ .env file not found")
        print("  Copy .env.example to .env and fill in your credentials")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    import os
    required_vars = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"✗ Missing environment variables: {', '.join(missing)}")
        return False
    
    print("✓ .env file configured correctly")
    return True


def test_data_file():
    """Test that the Telco dataset exists."""
    print("\nTesting dataset availability...")
    data_path = Path(__file__).parent / "data" / "raw" / "Telco_customer_churn.xlsx"
    
    if not data_path.exists():
        print(f"✗ Dataset not found at {data_path}")
        return False
    
    print(f"✓ Dataset found: {data_path}")
    return True


def test_database_connection():
    """Test PostgreSQL connection."""
    print("\nTesting database connection...")
    try:
        from src import db
        if db.test_connection():
            print("✓ Database connection successful")
            return True
        else:
            print("✗ Database connection failed")
            return False
    except Exception as e:
        print(f"✗ Database connection error: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running")
        print("  2. Database 'churn_db' exists")
        print("  3. User 'churn_user' has access")
        print("\nSee setup_database.md for detailed instructions")
        return False


def main():
    """Run all setup tests."""
    print("=" * 70)
    print("CHURN ROI PROJECT - SETUP VALIDATION")
    print("=" * 70)
    
    tests = [
        ("Package Imports", test_imports),
        ("Environment Configuration", test_env_file),
        ("Dataset Availability", test_data_file),
        ("Database Connection", test_database_connection),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to run the data loading pipeline.")
        print("\nNext step: Run the data loading pipeline")
        print("  cd src")
        print("  python load_data.py")
    else:
        print("\n✗ Some tests failed. Please fix the issues above before continuing.")
        print("\nSee README.md and setup_database.md for detailed setup instructions.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
