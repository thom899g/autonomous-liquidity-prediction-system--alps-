"""
Autonomous Liquidity Prediction System (ALPS)
A self-evolving AI system for predicting and optimizing trading strategies
based on real-time market liquidity analysis.
"""

__version__ = "1.0.0"
__author__ = "Evolution Ecosystem - Autonomous Architect"

from .config import ALPSConfig
from .data_collector import MarketDataCollector
from .liquidity_analyzer import LiquidityAnalyzer
from .predictor import LiquidityPredictor
from .strategy_optimizer import StrategyOptimizer
from .state_manager import StateManager
from .main import ALPSCore

__all__ = [
    "ALPSConfig",
    "MarketDataCollector",
    "LiquidityAnalyzer", 
    "LiquidityPredictor",
    "StrategyOptimizer",
    "StateManager",
    "ALPSCore"
]