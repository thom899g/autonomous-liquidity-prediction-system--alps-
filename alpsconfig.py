"""
Configuration management for ALPS with environment-based settings
and rigorous validation.
"""
import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

@dataclass
class ExchangeConfig:
    """Configuration for a single exchange connection."""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    rate_limit: int = 10  # requests per second
    enabled: bool = True
    
    def validate(self) -> bool:
        """Validate exchange configuration."""
        if not self.name:
            raise ValueError("Exchange name cannot be empty")
        if self.rate_limit <= 0:
            raise ValueError(f"Invalid rate limit for {self.name}")
        return True

@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    window_size: int = 100  # Number of data points for prediction
    prediction_horizon: int = 10  # Steps ahead to predict
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2
    use_gpu: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "prediction_horizon": self.prediction_horizon,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "use_gpu": self.use_gpu
        }

@dataclass
class ALPSConfig:
    """Main configuration class for ALPS system."""
    
    # System settings
    system_id: str = "alps_v1"
    log_level: str = "INFO"
    max_workers: int = 4
    
    # Exchange configurations
    exchanges: List[ExchangeConfig] = field(default_factory=list)
    
    # Model configuration
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Firebase configuration
    firebase_project_id: Optional[str] = None
    firestore_collection: str = "alps_state"
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    
    # Data collection
    data_update_interval: int = 5  # seconds
    historical_days: int = 30
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        self._load_from_env()
        self._validate()
        
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Firebase
        self.firebase_project_id = os.getenv("FIREBASE_PROJECT_ID", self.firebase_project_id)
        
        # Exchanges
        exchange_names = os.getenv("EXCHANGES", "binance,kraken").split(",")
        for name in exchange_names:
            api_key = os.getenv(f"{name.upper()}_API_KEY")
            api_secret = os.getenv(f"{name.upper()}_API_SECRET")
            self.exchanges.append(ExchangeConfig(
                name=name.strip(),
                api_key=api_key,
                api_secret=api_secret
            ))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        
    def _validate(self):
        """Validate entire configuration."""
        if not self.exchanges:
            raise ValueError("At least one exchange must be configured")
        
        for exchange in self.exchanges:
            exchange.validate()
            
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
            
        if not self.firebase_project_id:
            logging.warning("Firebase project ID not set - state persistence disabled")
            
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {