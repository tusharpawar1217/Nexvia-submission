"""
Configuration management using pathlib, python-dotenv, and config.yaml.

Loads M5_DATA_DIR from .env and validates that all required files exist.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Logger
logger = logging.getLogger(__name__)


class Config:
    """Configuration singleton that loads from config.yaml and .env"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    _m5_data_dir: Path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
            cls._instance._validate_m5_data()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = PROJECT_ROOT / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
    
    def _validate_m5_data(self) -> None:
        """Validate M5 dataset location and files from .env."""
        m5_dir_str = os.getenv('M5_DATA_DIR')
        
        if not m5_dir_str:
            raise EnvironmentError(
                "M5_DATA_DIR not set in .env file!\n"
                "Please create .env from .env.example and set M5_DATA_DIR"
            )
        
        # Convert to pathlib.Path (handles Windows paths correctly)
        self._m5_data_dir = Path(m5_dir_str)
        
        if not self._m5_data_dir.exists():
            raise FileNotFoundError(
                f"M5 data directory does not exist: {self._m5_data_dir}\n"
                f"Please check M5_DATA_DIR in .env"
            )
        
        # Required M5 files
        required_files = [
            'calendar.csv',
            'sales_train_validation.csv',
            'sales_train_evaluation.csv',
            'sell_prices.csv',
            'sample_submission.csv'
        ]
        
        missing_files = []
        for filename in required_files:
            filepath = self._m5_data_dir / filename
            if not filepath.exists():
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required M5 files in {self._m5_data_dir}:\n" +
                "\n".join(f"  - {f}" for f in missing_files)
            )
        
        logger.info(f"✅ All M5 files found in {self._m5_data_dir}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self.get(key)
    
    @property
    def m5_data_dir(self) -> Path:
        """M5 dataset directory from .env"""
        return self._m5_data_dir
    
    def get_m5_file(self, filename: str) -> Path:
        """Get full path to an M5 file."""
        return self._m5_data_dir / filename
    
    # Data configuration
    
    @property
    def profile(self) -> str:
        """Active data profile ('tiny', 'small', 'medium', 'full')"""
        return self.get('data.profile', 'medium')
    
    @property
    def states(self) -> List[str]:
        """States to include based on active profile"""
        profile_config = self.get(f'data.profiles.{self.profile}', {})
        return profile_config.get('states', ['CA', 'TX'])
    
    @property
    def stores(self) -> List[str]:
        """Store IDs to include (empty = all stores in selected states)"""
        profile_config = self.get(f'data.profiles.{self.profile}', {})
        return profile_config.get('stores', [])
    
    @property
    def item_sample_fraction(self) -> float:
        """Fraction of items to sample per category"""
        profile_config = self.get(f'data.profiles.{self.profile}', {})
        return profile_config.get('item_sample_fraction', 1.0)
    
    @property
    def seed(self) -> int:
        """Random seed for reproducibility"""
        return self.get('data.seed', 42)
    
    # Date configuration
    
    @property
    def train_end_day(self) -> int:
        """Last day number for training (d_XXXX)"""
        return self.get('dates.train_end_day', 1913)
    
    @property
    def test_days(self) -> int:
        """Number of days in test set"""
        return self.get('dates.test_days', 28)
    
    @property
    def backtest_folds(self) -> int:
        """Number of backtest folds"""
        return self.get('dates.backtest_folds', 2)
    
    @property
    def fold_stride(self) -> int:
        """Days between backtest fold starts"""
        return self.get('dates.fold_stride', 28)
    
    # Model configuration
    
    @property
    def horizon(self) -> int:
        """Forecast horizon in days"""
        return self.get('models.horizon', 28)
    
    @property
    def lags(self) -> List[int]:
        """Lag periods for features"""
        return self.get('features.lags', [7, 14, 28])
    
    @property
    def rolling_windows(self) -> List[int]:
        """Rolling window sizes for features"""
        return self.get('features.rolling_windows', [7, 28])
    
    # Inventory optimization parameters (ALL ASSUMED)
    
    @property
    def unit_cost_default(self) -> float:
        """Default unit cost (ASSUMED)"""
        return self.get('inventory.unit_cost_default', 10.0)
    
    @property
    def unit_cost_by_category(self) -> Dict[str, float]:
        """Unit cost by category (ASSUMED)"""
        return self.get('inventory.unit_cost_by_category', {})
    
    @property
    def holding_cost_pct_per_day(self) -> float:
        """Holding cost as % of unit cost per day (ASSUMED)"""
        return self.get('inventory.holding_cost_pct_per_day', 0.001)
    
    @property
    def stockout_penalty_default(self) -> float:
        """Default stockout penalty per unit (ASSUMED)"""
        return self.get('inventory.stockout_penalty_default', 25.0)
    
    @property
    def stockout_penalty_by_category(self) -> Dict[str, float]:
        """Stockout penalty by category (ASSUMED)"""
        return self.get('inventory.stockout_penalty_by_category', {})
    
    @property
    def lead_time_default(self) -> int:
        """Default lead time in days (ASSUMED)"""
        return self.get('inventory.lead_time_default', 7)
    
    @property
    def shelf_life_by_category(self) -> Dict[str, int]:
        """Shelf life by category in days (ASSUMED)"""
        return self.get('inventory.shelf_life_by_category', {})
    
    @property
    def store_capacity(self) -> float:
        """Store storage capacity (ASSUMED)"""
        return self.get('inventory.store_capacity', 100000)
    
    @property
    def budget_default(self) -> float:
        """Default budget constraint (ASSUMED)"""
        return self.get('inventory.budget_default', 500000)
    
    # API configuration
    
    @property
    def api_host(self) -> str:
        """API host"""
        return os.getenv('API_HOST', self.get('api.host', '127.0.0.1'))
    
    @property
    def api_port(self) -> int:
        """API port"""
        return int(os.getenv('API_PORT', self.get('api.port', 8000)))


# Singleton instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    logging.basicConfig(level=logging.INFO)
    
    print(f"\n{'='*80}")
    print("CONFIGURATION TEST")
    print(f"{'='*80}\n")
    
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📁 M5 data directory: {config.m5_data_dir}")
    print(f"\n📊 Active profile: {config.profile}")
    print(f"   States: {config.states}")
    print(f"   Stores: {config.stores if config.stores else 'All in selected states'}")
    print(f"   Item sample: {config.item_sample_fraction*100:.0f}%")
    print(f"\n📅 Date configuration:")
    print(f"   Train end: d_{config.train_end_day}")
    print(f"   Test days: {config.test_days} (d_{config.train_end_day+1} to d_{config.train_end_day+config.test_days})")
    print(f"   Horizon: {config.horizon} days")
    print(f"\n💰 Inventory assumptions (ASSUMED - not from M5):")
    print(f"   Unit cost default: ${config.unit_cost_default}")
    print(f"   Stockout penalty default: ${config.stockout_penalty_default}")
    print(f"   Holding cost: {config.holding_cost_pct_per_day*100:.2f}% per day")
    print(f"\n{'='*80}\n")
