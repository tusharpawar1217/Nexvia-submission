"""
Configuration module for the Customer Churn ROI project.
Loads environment variables and provides centralized configuration.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Ensure directories exist
for dir_path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, EVALUATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "churn_db"),
    "user": os.getenv("DB_USER", "churn_user"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# SQLAlchemy connection string
DB_CONNECTION_STRING = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# Model configuration
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Feature engineering configuration
TENURE_BUCKETS = {
    "0-12": (0, 12),
    "13-24": (13, 24),
    "25-48": (25, 48),
    "49+": (49, float("inf")),
}

# CLV configuration
DISCOUNT_RATE = float(os.getenv("DISCOUNT_RATE", "0.10"))
MONTHLY_MARGIN = float(os.getenv("MONTHLY_MARGIN", "0.30"))

# Model file paths
CHURN_MODEL_PATH = MODELS_DIR / "churn_model.pkl"
SURVIVAL_MODEL_PATH = MODELS_DIR / "survival_model.pkl"
UPLIFT_MODEL_PATH = MODELS_DIR / "uplift_model.pkl"
SEGMENTATION_MODEL_PATH = MODELS_DIR / "segmentation_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Data file paths
TELCO_DATA_PATH = DATA_RAW_DIR / "Telco_customer_churn.xlsx"
HILLSTROM_DATA_PATH = DATA_RAW_DIR / "hillstrom.csv"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Autopay payment methods (for feature engineering)
AUTOPAY_METHODS = ["Bank transfer (automatic)", "Credit card (automatic)"]

# Feature lists for modeling
CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
]

# Target variable
TARGET = "churn"

def get_feature_columns():
    """Return list of all feature columns for modeling."""
    return CATEGORICAL_FEATURES + NUMERIC_FEATURES


def validate_config():
    """Validate that required configuration is present."""
    errors = []
    
    if not DB_CONFIG["password"]:
        errors.append("DB_PASSWORD not set in environment variables")
    
    if not TELCO_DATA_PATH.exists():
        errors.append(f"Telco data file not found at {TELCO_DATA_PATH}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Database: {DB_CONFIG['database']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Discount rate: {DISCOUNT_RATE}")
    print(f"Monthly margin: {MONTHLY_MARGIN}")
