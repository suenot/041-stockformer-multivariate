#!/usr/bin/env python3
"""
Stockformer Example Usage

This script demonstrates the complete workflow:
1. Loading data from Bybit
2. Preparing features
3. Training the model
4. Generating predictions and trading signals
5. Running backtest
"""

import numpy as np
import torch
from typing import List, Optional

# Import from local modules
from data import BybitClient, MultiAssetDataset, FeatureEngineering
from model import StockformerConfig, StockformerModel, OutputType
from strategy import SignalGenerator, Backtester, BacktestConfig, PortfolioSignal


def main():
    print("=" * 60)
    print("Stockformer: Multivariate Stock Prediction Example")
    print("=" * 60)

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    seq_len = 96

    print(f"\nSymbols: {symbols}")
    print(f"Sequence length: {seq_len}")

    # Step 1: Load Data
    print("\n" + "=" * 40)
    print("Step 1: Loading Data")
    print("=" * 40)

    try:
        client = BybitClient()
        dataset = MultiAssetDataset(symbols=symbols, seq_len=seq_len)
        dataset.load_from_bybit(client, interval="1h", limit=500)

        if len(dataset) > 0:
            print(f"Loaded {len(dataset)} samples")
        else:
            print("No data loaded from API, using synthetic data")
            raise Exception("Using synthetic data")

    except Exception as e:
        print(f"API loading failed: {e}")
        print("Generating synthetic data for demonstration...")
        dataset = generate_synthetic_dataset(symbols, seq_len)

    # Step 2: Create Model
    print("\n" + "=" * 40)
    print("Step 2: Creating Model")
    print("=" * 40)

    config = StockformerConfig(
        num_tickers=len(symbols),
        seq_len=seq_len,
        input_features=5,
        d_model=32,
        num_heads=4,
        d_ff=128,
        num_encoder_layers=2,
        dropout=0.1,
        output_type=OutputType.REGRESSION,
        use_cross_ticker_attention=True
    )

    model = StockformerModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Step 3: Training Loop (Simplified)
    print("\n" + "=" * 40)
    print("Step 3: Training (Demonstration)")
    print("=" * 40)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(3):
        total_loss = 0
        num_batches = 0

        for i in range(0, len(dataset), 32):
            X, y = dataset.get_batch(i // 32, batch_size=32)

            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output['predictions'], y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/3: Loss = {avg_loss:.6f}")

    # Step 4: Generate Predictions
    print("\n" + "=" * 40)
    print("Step 4: Generating Predictions")
    print("=" * 40)

    model.eval()
    with torch.no_grad():
        X, y = dataset.get_batch(0, batch_size=1)
        X = torch.FloatTensor(X)

        output = model(X, return_attention=True)
        predictions = output['predictions'].numpy()[0]

        print("\nPredicted returns:")
        for i, symbol in enumerate(symbols):
            direction = "↑" if predictions[i] > 0 else "↓"
            print(f"  {symbol}: {predictions[i]*100:+.2f}% {direction}")

    # Step 5: Generate Trading Signals
    print("\n" + "=" * 40)
    print("Step 5: Generating Trading Signals")
    print("=" * 40)

    signal_gen = SignalGenerator(
        long_threshold=0.003,
        short_threshold=-0.003,
        min_confidence=0.3
    )

    portfolio = signal_gen.generate_from_regression(
        predictions,
        confidence=np.ones(len(symbols)) * 0.7
    )

    print("\nTrading Signals:")
    for signal in portfolio.signals:
        symbol = symbols[signal.ticker_idx]
        print(f"  {symbol}: {signal.signal_type.value.upper()}, "
              f"strength={signal.strength:.2f}, "
              f"position_size={signal.position_size:.2f}")

    print(f"\nPortfolio weights: {[f'{w:.2f}' for w in portfolio.weights]}")

    # Step 6: Run Backtest
    print("\n" + "=" * 40)
    print("Step 6: Running Backtest")
    print("=" * 40)

    # Generate signals for all periods
    num_periods = min(len(dataset), 200)
    signals: List[PortfolioSignal] = []
    prices = np.zeros((num_periods, len(symbols)))

    for i in range(num_periods):
        X, y = dataset[i]
        X = torch.FloatTensor(X).unsqueeze(0)

        with torch.no_grad():
            output = model(X)
            preds = output['predictions'].numpy()[0]

        signals.append(signal_gen.generate_from_regression(preds, np.ones(len(symbols)) * 0.7))

        # Use last close price from features
        for t in range(len(symbols)):
            prices[i, t] = 100 * (1 + X[0, t, -1, 1].item())  # Normalized price feature

    # Run backtest
    bt_config = BacktestConfig(
        initial_capital=100_000,
        commission=0.001,
        slippage=0.0005,
        use_stop_loss=True,
        stop_loss_level=0.05,
        use_take_profit=True,
        take_profit_level=0.10
    )

    backtester = Backtester(bt_config)
    result = backtester.run(prices, signals, periods_per_year=8760)

    print(result.summary())

    # Step 7: Analyze Attention
    print("\n" + "=" * 40)
    print("Step 7: Attention Analysis")
    print("=" * 40)

    if output.get('attention_weights'):
        print("\nCross-ticker attention patterns:")
        for layer_name, attn_dict in output['attention_weights'].items():
            if 'cross_ticker' in attn_dict:
                cross_attn = attn_dict['cross_ticker']
                if cross_attn is not None:
                    # Average over batch and heads
                    avg_attn = cross_attn.mean(dim=(0, 1)).numpy()
                    print(f"\n{layer_name} - Cross-ticker attention matrix:")
                    print("     " + "  ".join(f"{s:>6}" for s in symbols))
                    for i, row_symbol in enumerate(symbols):
                        row = "  ".join(f"{avg_attn[i, j]:.3f}" for j in range(len(symbols)))
                        print(f"{row_symbol}: {row}")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


def generate_synthetic_dataset(symbols: List[str], seq_len: int) -> MultiAssetDataset:
    """Generate synthetic dataset for demonstration"""
    import pandas as pd

    num_periods = 500
    data = {}

    for i, symbol in enumerate(symbols):
        base_price = 100 * (i + 1)
        prices = [base_price]
        volumes = [1000000]

        for _ in range(num_periods - 1):
            ret = np.random.normal(0.0001, 0.02)
            prices.append(prices[-1] * (1 + ret))
            volumes.append(volumes[-1] * (1 + np.random.normal(0, 0.1)))

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': volumes
        })
        df.index = pd.date_range(start='2024-01-01', periods=num_periods, freq='h')
        data[symbol] = df

    dataset = MultiAssetDataset(symbols=symbols, seq_len=seq_len)
    dataset.load_from_dataframe(data)

    return dataset


if __name__ == "__main__":
    main()
