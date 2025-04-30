"""
Stockformer: Multivariate Stock Prediction with Cross-Ticker Attention

This package provides a PyTorch implementation of the Stockformer model
for multivariate stock prediction with support for Bybit exchange data.
"""

from .model import StockformerConfig, StockformerModel
from .data import BybitClient, MultiAssetDataset, FeatureEngineering
from .strategy import SignalGenerator, Backtester

__version__ = "0.1.0"
__all__ = [
    "StockformerConfig",
    "StockformerModel",
    "BybitClient",
    "MultiAssetDataset",
    "FeatureEngineering",
    "SignalGenerator",
    "Backtester",
]
