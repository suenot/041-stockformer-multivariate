"""
Trading Strategy Module for Stockformer

Provides:
- SignalGenerator: Generate trading signals from predictions
- Backtester: Backtest trading strategies
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SignalType(Enum):
    """Trading signal type"""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class Signal:
    """Trading signal for a single asset"""
    signal_type: SignalType
    ticker_idx: int
    strength: float  # [0, 1]
    predicted_return: float
    confidence: float
    position_size: float = 0.0

    def __post_init__(self):
        self.position_size = self.strength * self.confidence

    def is_actionable(self, min_strength: float = 0.0, min_confidence: float = 0.0) -> bool:
        """Check if signal meets minimum thresholds"""
        return (
            self.signal_type != SignalType.HOLD and
            self.strength >= min_strength and
            self.confidence >= min_confidence
        )


@dataclass
class PortfolioSignal:
    """Portfolio-level trading signals"""
    signals: List[Signal]
    weights: List[float]
    timestamp: int = 0

    @property
    def overall_confidence(self) -> float:
        if not self.signals:
            return 0.0
        return sum(s.confidence for s in self.signals) / len(self.signals)

    def longs(self) -> List[Signal]:
        return [s for s in self.signals if s.signal_type == SignalType.LONG]

    def shorts(self) -> List[Signal]:
        return [s for s in self.signals if s.signal_type == SignalType.SHORT]


class SignalGenerator:
    """
    Generate trading signals from model predictions

    Example:
        gen = SignalGenerator(long_threshold=0.005, short_threshold=-0.005)
        portfolio = gen.generate_from_regression(predictions, confidence)
    """

    def __init__(
        self,
        long_threshold: float = 0.005,
        short_threshold: float = -0.005,
        min_confidence: float = 0.3,
        use_kelly: bool = True,
        max_position_size: float = 0.25
    ):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.min_confidence = min_confidence
        self.use_kelly = use_kelly
        self.max_position_size = max_position_size

    def generate_from_regression(
        self,
        predictions: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        symbols: Optional[List[str]] = None
    ) -> PortfolioSignal:
        """
        Generate signals from regression predictions

        Args:
            predictions: Predicted returns [num_tickers]
            confidence: Confidence scores [num_tickers]
            symbols: List of symbol names

        Returns:
            PortfolioSignal with trading signals
        """
        num_tickers = len(predictions)

        if confidence is None:
            confidence = np.ones(num_tickers) * 0.5

        signals = []
        weights = np.zeros(num_tickers)

        for t in range(num_tickers):
            pred = predictions[t]
            conf = confidence[t]

            if pred > self.long_threshold:
                signal_type = SignalType.LONG
                strength = min(pred / self.long_threshold, 1.0)
            elif pred < self.short_threshold:
                signal_type = SignalType.SHORT
                strength = min(abs(pred / self.short_threshold), 1.0)
            else:
                signal_type = SignalType.HOLD
                strength = 0.0

            signal = Signal(
                signal_type=signal_type,
                ticker_idx=t,
                strength=strength,
                predicted_return=pred,
                confidence=conf
            )

            if self.use_kelly and signal.is_actionable(0.0, self.min_confidence):
                kelly = self._kelly_criterion(pred, conf)
                weights[t] = kelly * (1 if signal_type == SignalType.LONG else -1)

            signals.append(signal)

        # Normalize weights
        total_weight = np.sum(np.abs(weights))
        if total_weight > 1.0:
            weights /= total_weight

        return PortfolioSignal(signals=signals, weights=weights.tolist())

    def generate_from_direction(
        self,
        probabilities: np.ndarray,
        symbols: Optional[List[str]] = None
    ) -> PortfolioSignal:
        """
        Generate signals from direction classification

        Args:
            probabilities: [num_tickers, 3] with [down, hold, up] probs

        Returns:
            PortfolioSignal
        """
        num_tickers = probabilities.shape[0]
        signals = []
        weights = np.zeros(num_tickers)

        for t in range(num_tickers):
            p_down, p_hold, p_up = probabilities[t]

            if p_up > p_down and p_up > p_hold:
                signal_type = SignalType.LONG
                strength = p_up
                pred_return = p_up - p_down
            elif p_down > p_up and p_down > p_hold:
                signal_type = SignalType.SHORT
                strength = p_down
                pred_return = p_down - p_up
            else:
                signal_type = SignalType.HOLD
                strength = p_hold
                pred_return = 0.0

            confidence = max(p_up, p_down, p_hold)

            signal = Signal(
                signal_type=signal_type,
                ticker_idx=t,
                strength=strength,
                predicted_return=pred_return,
                confidence=confidence
            )

            if signal.is_actionable(0.5, self.min_confidence):
                weights[t] = signal.position_size * (1 if signal_type == SignalType.LONG else -1)

            signals.append(signal)

        # Normalize
        total = np.sum(np.abs(weights))
        if total > 1.0:
            weights /= total

        return PortfolioSignal(signals=signals, weights=weights.tolist())

    def generate_from_portfolio(
        self,
        weights: np.ndarray,
        symbols: Optional[List[str]] = None
    ) -> PortfolioSignal:
        """
        Generate signals from portfolio weights

        Args:
            weights: Portfolio weights [num_tickers] that sum to 1

        Returns:
            PortfolioSignal
        """
        num_tickers = len(weights)
        equal_weight = 1.0 / num_tickers
        signals = []

        for t in range(num_tickers):
            w = weights[t]
            deviation = w - equal_weight

            if deviation > 0.05:
                signal_type = SignalType.LONG
                strength = min(deviation / equal_weight, 1.0)
            elif deviation < -0.05:
                signal_type = SignalType.SHORT
                strength = min(abs(deviation / equal_weight), 1.0)
            else:
                signal_type = SignalType.HOLD
                strength = 0.0

            signal = Signal(
                signal_type=signal_type,
                ticker_idx=t,
                strength=strength,
                predicted_return=deviation,
                confidence=w
            )
            signals.append(signal)

        return PortfolioSignal(signals=signals, weights=weights.tolist())

    def _kelly_criterion(self, expected_return: float, win_probability: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        if abs(expected_return) < 1e-8:
            return 0.0

        p = win_probability
        b = abs(expected_return)

        if b > 0:
            kelly = (p * (1 + b) - 1) / b
        else:
            kelly = 0.0

        # Use half-Kelly for conservatism
        return max(0.0, min(kelly * 0.5, self.max_position_size))


@dataclass
class Trade:
    """Record of a single trade"""
    ticker_idx: int
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    position_size: float
    direction: int  # 1 = long, -1 = short
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Results from backtesting"""
    trades: List[Trade]
    equity_curve: List[float]
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int

    def summary(self) -> str:
        return f"""
Backtest Results Summary
========================
Total Return: {self.total_return * 100:.2f}%
Annual Return: {self.annual_return * 100:.2f}%
Volatility: {self.volatility * 100:.2f}%
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}
Max Drawdown: {self.max_drawdown * 100:.2f}%
Calmar Ratio: {self.calmar_ratio:.2f}
Win Rate: {self.win_rate * 100:.2f}%
Profit Factor: {self.profit_factor:.2f}
Number of Trades: {self.num_trades}
"""


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    max_leverage: float = 1.0
    allow_short: bool = False
    min_position_size: float = 0.01
    use_stop_loss: bool = True
    stop_loss_level: float = 0.05
    use_take_profit: bool = True
    take_profit_level: float = 0.10


class Backtester:
    """
    Backtest trading strategies

    Example:
        config = BacktestConfig(initial_capital=100000)
        backtester = Backtester(config)
        result = backtester.run(prices, signals)
        print(result.summary())
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.reset()

    def reset(self):
        """Reset backtester state"""
        self.capital = self.config.initial_capital
        self.positions: Dict[int, dict] = {}
        self.equity_curve: List[float] = []
        self.trades: List[Trade] = []

    def run(
        self,
        prices: np.ndarray,
        signals: List[PortfolioSignal],
        periods_per_year: int = 252
    ) -> BacktestResult:
        """
        Run backtest

        Args:
            prices: Price matrix [time, num_tickers]
            signals: List of portfolio signals for each time step
            periods_per_year: Number of periods in a year (for annualization)

        Returns:
            BacktestResult with performance metrics
        """
        self.reset()
        num_periods, num_tickers = prices.shape

        if len(signals) != num_periods:
            raise ValueError("Number of signals must match number of periods")

        self.equity_curve.append(self.capital)

        for t in range(num_periods):
            current_prices = prices[t]

            # Check stops
            if t > 0:
                self._check_exits(t, current_prices)

            # Process signals
            self._process_signals(t, signals[t], current_prices)

            # Update equity
            portfolio_value = self._calculate_portfolio_value(current_prices)
            self.equity_curve.append(portfolio_value)

        # Close all remaining positions
        final_prices = prices[-1]
        self._close_all_positions(num_periods - 1, final_prices, "End of backtest")

        return self._calculate_metrics(periods_per_year)

    def _check_exits(self, time: int, prices: np.ndarray):
        """Check and execute stop-loss/take-profit"""
        to_close = []

        for ticker_idx, pos in self.positions.items():
            current_price = prices[ticker_idx]
            return_pct = (current_price - pos['entry_price']) / pos['entry_price'] * pos['direction']

            if self.config.use_stop_loss and return_pct <= -self.config.stop_loss_level:
                to_close.append((ticker_idx, "Stop Loss"))
            elif self.config.use_take_profit and return_pct >= self.config.take_profit_level:
                to_close.append((ticker_idx, "Take Profit"))

        for ticker_idx, reason in to_close:
            self._close_position(ticker_idx, time, prices[ticker_idx], reason)

    def _process_signals(self, time: int, portfolio: PortfolioSignal, prices: np.ndarray):
        """Process trading signals"""
        for signal in portfolio.signals:
            ticker_idx = signal.ticker_idx
            price = prices[ticker_idx]

            if signal.signal_type == SignalType.LONG:
                if ticker_idx not in self.positions:
                    position_value = self.capital * signal.position_size
                    if position_value >= self.config.min_position_size * self.capital:
                        self._open_position(ticker_idx, time, price, position_value, 1)

            elif signal.signal_type == SignalType.SHORT and self.config.allow_short:
                if ticker_idx not in self.positions:
                    position_value = self.capital * signal.position_size
                    if position_value >= self.config.min_position_size * self.capital:
                        self._open_position(ticker_idx, time, price, position_value, -1)

            elif signal.signal_type == SignalType.CLOSE:
                if ticker_idx in self.positions:
                    self._close_position(ticker_idx, time, price, "Signal")

    def _open_position(self, ticker_idx: int, time: int, price: float, size: float, direction: int):
        """Open a new position"""
        adjusted_price = price * (1 + self.config.slippage * direction)
        commission = size * self.config.commission

        self.capital -= commission

        self.positions[ticker_idx] = {
            'entry_time': time,
            'entry_price': adjusted_price,
            'size': size,
            'direction': direction
        }

    def _close_position(self, ticker_idx: int, time: int, price: float, reason: str):
        """Close an existing position"""
        if ticker_idx not in self.positions:
            return

        pos = self.positions.pop(ticker_idx)
        adjusted_price = price * (1 - self.config.slippage * pos['direction'])

        price_change = (adjusted_price - pos['entry_price']) / pos['entry_price']
        pnl = pos['size'] * price_change * pos['direction']
        commission = pos['size'] * self.config.commission

        self.capital += pos['size'] + pnl - commission

        self.trades.append(Trade(
            ticker_idx=ticker_idx,
            entry_time=pos['entry_time'],
            exit_time=time,
            entry_price=pos['entry_price'],
            exit_price=adjusted_price,
            position_size=pos['size'],
            direction=pos['direction'],
            pnl=pnl,
            return_pct=price_change * pos['direction'],
            exit_reason=reason
        ))

    def _close_all_positions(self, time: int, prices: np.ndarray, reason: str):
        """Close all open positions"""
        tickers = list(self.positions.keys())
        for ticker_idx in tickers:
            self._close_position(ticker_idx, time, prices[ticker_idx], reason)

    def _calculate_portfolio_value(self, prices: np.ndarray) -> float:
        """Calculate current portfolio value"""
        value = self.capital

        for ticker_idx, pos in self.positions.items():
            current_price = prices[ticker_idx]
            price_change = (current_price - pos['entry_price']) / pos['entry_price']
            unrealized_pnl = pos['size'] * price_change * pos['direction']
            value += pos['size'] + unrealized_pnl

        return value

    def _calculate_metrics(self, periods_per_year: int) -> BacktestResult:
        """Calculate performance metrics"""
        initial = self.config.initial_capital
        final = self.equity_curve[-1] if self.equity_curve else initial

        # Returns
        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])

        total_return = (final - initial) / initial
        num_periods = len(self.equity_curve)
        annual_return = (1 + total_return) ** (periods_per_year / max(num_periods, 1)) - 1

        # Volatility
        volatility = np.std(returns) * np.sqrt(periods_per_year) if len(returns) > 0 else 0

        # Sharpe ratio
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0

        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = sum(abs(t.pnl) for t in losing_trades)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0

        return BacktestResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(self.trades)
        )


if __name__ == "__main__":
    # Test signal generator
    print("Testing SignalGenerator...")

    gen = SignalGenerator(long_threshold=0.005, short_threshold=-0.005)

    predictions = np.array([0.02, -0.01, 0.003, 0.015, -0.02])
    confidence = np.array([0.7, 0.6, 0.5, 0.8, 0.65])

    portfolio = gen.generate_from_regression(predictions, confidence)

    print(f"Generated {len(portfolio.signals)} signals:")
    for i, signal in enumerate(portfolio.signals):
        print(f"  Ticker {i}: {signal.signal_type.value}, strength={signal.strength:.2f}")

    print(f"\nPortfolio weights: {portfolio.weights}")

    # Test backtester
    print("\n\nTesting Backtester...")

    np.random.seed(42)
    num_periods = 100
    num_tickers = 3

    # Generate synthetic prices
    prices = np.zeros((num_periods, num_tickers))
    for t in range(num_tickers):
        price = 100 + t * 50
        for p in range(num_periods):
            price *= 1 + np.random.normal(0.0005, 0.02)
            prices[p, t] = price

    # Generate signals
    signals = []
    for p in range(num_periods):
        pred = np.random.normal(0, 0.01, num_tickers)
        conf = np.random.uniform(0.4, 0.8, num_tickers)
        signals.append(gen.generate_from_regression(pred, conf))

    # Run backtest
    config = BacktestConfig(initial_capital=100000, use_stop_loss=True)
    backtester = Backtester(config)
    result = backtester.run(prices, signals, periods_per_year=252)

    print(result.summary())
