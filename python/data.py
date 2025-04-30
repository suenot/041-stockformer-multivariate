"""
Data loading and processing for Stockformer

Provides:
- Bybit API client for cryptocurrency data
- Multi-asset dataset handling
- Feature engineering utilities
"""

import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import time


@dataclass
class Kline:
    """OHLCV candlestick data"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


class BybitClient:
    """
    HTTP client for Bybit exchange API

    Example:
        client = BybitClient()
        klines = client.get_klines("BTCUSDT", "60", limit=100)
    """

    BASE_URL = "https://api.bybit.com"

    # Interval mapping
    INTERVALS = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
        "1d": "D", "1w": "W", "1M": "M"
    }

    def __init__(self, rate_limit_delay: float = 0.1):
        """
        Initialize Bybit client

        Args:
            rate_limit_delay: Delay between requests in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Kline]:
        """
        Get historical klines (candlesticks)

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Time interval ("1m", "5m", "1h", "1d", etc.)
            limit: Number of candles (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            List of Kline objects
        """
        interval_code = self.INTERVALS.get(interval, interval)

        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval_code,
            "limit": min(limit, 1000)
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        response = self._request("/v5/market/kline", params)

        klines = []
        if response and "result" in response and "list" in response["result"]:
            for item in response["result"]["list"]:
                klines.append(Kline(
                    timestamp=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    turnover=float(item[6])
                ))

        # API returns newest first, reverse for chronological order
        return list(reversed(klines))

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get current ticker information

        Args:
            symbol: Trading pair

        Returns:
            Ticker dict with price, volume, etc.
        """
        params = {"category": "spot", "symbol": symbol}
        response = self._request("/v5/market/tickers", params)

        if response and "result" in response and "list" in response["result"]:
            if response["result"]["list"]:
                return response["result"]["list"][0]
        return None

    def get_orderbook(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """
        Get order book

        Args:
            symbol: Trading pair
            limit: Depth limit (5, 10, 25, 50)

        Returns:
            Dict with bids and asks
        """
        params = {"category": "spot", "symbol": symbol, "limit": limit}
        response = self._request("/v5/market/orderbook", params)

        if response and "result" in response:
            return response["result"]
        return None

    def _request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request with rate limiting"""
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(f"{self.BASE_URL}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API Error: {e}")
            return None


class FeatureEngineering:
    """
    Feature engineering utilities for financial time series

    Example:
        fe = FeatureEngineering()
        returns = fe.log_returns(prices)
        rsi = fe.rsi(prices, period=14)
    """

    @staticmethod
    def log_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate log returns"""
        prices = np.asarray(prices)
        returns = np.zeros_like(prices)
        returns[1:] = np.log(prices[1:] / prices[:-1])
        return returns

    @staticmethod
    def simple_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate simple returns"""
        prices = np.asarray(prices)
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        return returns

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index

        Args:
            prices: Price array
            period: RSI period

        Returns:
            RSI values (0-100)
        """
        prices = np.asarray(prices)
        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Simple moving average
        avg_gain = np.zeros(len(prices))
        avg_loss = np.zeros(len(prices))

        for i in range(period, len(prices)):
            avg_gain[i] = np.mean(gains[i-period:i])
            avg_loss[i] = np.mean(losses[i-period:i])

        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def realized_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Calculate realized volatility (rolling std of returns)

        Args:
            returns: Return array
            window: Rolling window size

        Returns:
            Volatility values
        """
        returns = np.asarray(returns)
        volatility = np.zeros_like(returns)

        for i in range(window, len(returns)):
            volatility[i] = np.std(returns[i-window:i])

        return volatility

    @staticmethod
    def volume_change(volumes: np.ndarray) -> np.ndarray:
        """Calculate volume change rate"""
        volumes = np.asarray(volumes)
        change = np.zeros_like(volumes)
        change[1:] = (volumes[1:] - volumes[:-1]) / (volumes[:-1] + 1e-8)
        return change

    @staticmethod
    def correlation_matrix(prices_dict: Dict[str, np.ndarray], window: int = 30) -> np.ndarray:
        """
        Calculate rolling correlation matrix between assets

        Args:
            prices_dict: Dict of {symbol: prices}
            window: Rolling window

        Returns:
            Correlation matrix [num_assets, num_assets]
        """
        symbols = list(prices_dict.keys())
        n = len(symbols)

        # Use returns for correlation
        returns = {s: FeatureEngineering.log_returns(p) for s, p in prices_dict.items()}

        # Take last `window` returns for correlation
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                r1 = returns[symbols[i]][-window:]
                r2 = returns[symbols[j]][-window:]
                if len(r1) > 1 and len(r2) > 1:
                    c = np.corrcoef(r1, r2)[0, 1]
                    if not np.isnan(c):
                        corr[i, j] = c
                        corr[j, i] = c

        return corr

    @staticmethod
    def normalize_features(features: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Normalize features

        Args:
            features: Feature array [seq_len, num_features]
            method: "zscore" or "minmax"

        Returns:
            Normalized features
        """
        if method == "zscore":
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True) + 1e-8
            return (features - mean) / std
        elif method == "minmax":
            min_val = np.min(features, axis=0, keepdims=True)
            max_val = np.max(features, axis=0, keepdims=True)
            return (features - min_val) / (max_val - min_val + 1e-8)
        else:
            return features


class MultiAssetDataset:
    """
    Dataset for multi-asset time series

    Example:
        dataset = MultiAssetDataset(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            seq_len=96
        )
        dataset.load_from_bybit(client)
        X, y = dataset.get_batch(0)
    """

    def __init__(
        self,
        symbols: List[str],
        seq_len: int = 96,
        prediction_horizon: int = 1,
        features: List[str] = None
    ):
        """
        Initialize dataset

        Args:
            symbols: List of trading symbols
            seq_len: Input sequence length
            prediction_horizon: Steps ahead to predict
            features: List of feature names
        """
        self.symbols = symbols
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        self.features = features or ["returns", "normalized_price", "volume_change", "rsi", "volatility"]

        self.data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None

        self.fe = FeatureEngineering()

    def load_from_bybit(
        self,
        client: BybitClient,
        interval: str = "1h",
        limit: int = 500
    ) -> None:
        """
        Load data from Bybit API

        Args:
            client: BybitClient instance
            interval: Time interval
            limit: Number of candles per symbol
        """
        for symbol in self.symbols:
            klines = client.get_klines(symbol, interval, limit)

            if klines:
                df = pd.DataFrame([
                    {
                        "timestamp": k.timestamp,
                        "open": k.open,
                        "high": k.high,
                        "low": k.low,
                        "close": k.close,
                        "volume": k.volume
                    }
                    for k in klines
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                self.data[symbol] = df
                print(f"Loaded {len(df)} candles for {symbol}")

        self._process_data()

    def load_from_dataframe(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Load data from DataFrames

        Args:
            data: Dict of {symbol: DataFrame} with OHLCV columns
        """
        self.data = data
        self._process_data()

    def _process_data(self) -> None:
        """Process raw data into features"""
        if not self.data:
            return

        # Find common timestamps
        common_index = None
        for symbol, df in self.data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        if common_index is None or len(common_index) < self.seq_len + self.prediction_horizon:
            print("Insufficient data")
            return

        num_samples = len(common_index) - self.seq_len - self.prediction_horizon + 1
        num_assets = len(self.symbols)
        num_features = len(self.features)

        self.processed_data = np.zeros((num_samples, num_assets, self.seq_len, num_features))
        self.targets = np.zeros((num_samples, num_assets))

        for t, symbol in enumerate(self.symbols):
            df = self.data[symbol].loc[common_index]
            prices = df["close"].values
            volumes = df["volume"].values

            # Calculate features
            returns = self.fe.log_returns(prices)
            normalized = (prices - prices[0]) / prices[0]
            vol_change = self.fe.volume_change(volumes)
            rsi = self.fe.rsi(prices, 14) / 100.0  # Normalize to [0, 1]
            volatility = self.fe.realized_volatility(returns, 20)

            feature_map = {
                "returns": returns,
                "normalized_price": normalized,
                "volume_change": vol_change,
                "rsi": rsi,
                "volatility": volatility,
                "close": prices / prices[0]
            }

            for i in range(num_samples):
                for f, feat_name in enumerate(self.features):
                    if feat_name in feature_map:
                        feat_values = feature_map[feat_name]
                        self.processed_data[i, t, :, f] = feat_values[i:i+self.seq_len]

                # Target: next period return
                target_idx = i + self.seq_len + self.prediction_horizon - 1
                if target_idx < len(returns):
                    self.targets[i, t] = returns[target_idx]

        print(f"Processed data shape: {self.processed_data.shape}")
        print(f"Targets shape: {self.targets.shape}")

    def get_batch(self, batch_idx: int, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of data

        Args:
            batch_idx: Batch index
            batch_size: Batch size

        Returns:
            (X, y) tuple
        """
        start = batch_idx * batch_size
        end = min(start + batch_size, len(self.processed_data))

        return self.processed_data[start:end], self.targets[start:end]

    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.processed_data) if self.processed_data is not None else 0

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get single sample"""
        return self.processed_data[idx], self.targets[idx]


if __name__ == "__main__":
    # Example usage
    print("Testing Bybit client...")
    client = BybitClient()

    ticker = client.get_ticker("BTCUSDT")
    if ticker:
        print(f"BTC Price: ${float(ticker.get('lastPrice', 0)):.2f}")

    klines = client.get_klines("BTCUSDT", "1h", limit=10)
    print(f"Loaded {len(klines)} klines")

    print("\nTesting feature engineering...")
    fe = FeatureEngineering()
    prices = np.array([100, 101, 99, 102, 103, 101, 104, 105, 103, 106])
    print(f"Log returns: {fe.log_returns(prices)}")
    print(f"RSI: {fe.rsi(prices, 3)}")
