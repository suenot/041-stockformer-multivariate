# Chapter 43: Stockformer — Multivariate Stock Prediction with Cross-Asset Attention

This chapter explores **Stockformer**, a transformer-based architecture designed for multivariate stock prediction. Unlike traditional univariate forecasting models, Stockformer leverages attention mechanisms to capture cross-asset relationships and temporal dependencies simultaneously.

<p align="center">
<img src="https://i.imgur.com/XqZ8k2P.png" width="70%">
</p>

## Contents

1. [Introduction to Stockformer](#introduction-to-stockformer)
    * [Why Multivariate Prediction?](#why-multivariate-prediction)
    * [Key Advantages](#key-advantages)
    * [Comparison with Other Models](#comparison-with-other-models)
2. [Stockformer Architecture](#stockformer-architecture)
    * [Token Embedding Layer](#token-embedding-layer)
    * [Cross-Ticker Attention](#cross-ticker-attention)
    * [ProbSparse Attention](#probsparse-attention)
    * [Prediction Head](#prediction-head)
3. [Multivariate Data Representation](#multivariate-data-representation)
    * [Log Percent Change](#log-percent-change)
    * [Multi-Ticker Inputs](#multi-ticker-inputs)
    * [Feature Engineering](#feature-engineering)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: Stockformer Architecture](#02-stockformer-architecture)
    * [03: Model Training](#03-model-training)
    * [04: Cross-Asset Prediction](#04-cross-asset-prediction)
    * [05: Portfolio Backtesting](#05-portfolio-backtesting)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to Stockformer

Stockformer is a modified Transformer architecture specifically designed for financial time-series forecasting. Instead of treating stock prediction as a simple univariate autoregression problem, Stockformer models the task as a **multivariate forecasting problem**, using attention mechanisms to discover relationships between multiple financial instruments.

### Why Multivariate Prediction?

Traditional models predict each asset independently:

```
BTCUSDT → Model → BTCUSDT_prediction
ETHUSDT → Model → ETHUSDT_prediction
```

Stockformer predicts using cross-asset information:

```
[BTCUSDT, ETHUSDT, SOLUSDT, ...] → Stockformer → BTCUSDT_prediction
                                                  (considering all relationships)
```

**Key insight**: Financial markets are interconnected. Bitcoin's movement affects Ethereum, oil prices affect airline stocks, and tech stocks often move together. Stockformer explicitly models these dependencies.

### Key Advantages

1. **Cross-Asset Relationship Modeling**
   - Captures correlations between different assets
   - Uses Granger causality to identify predictive relationships
   - Attention weights show which assets influence predictions

2. **Efficient Attention Mechanisms**
   - ProbSparse attention reduces complexity from O(L²) to O(L·log(L))
   - Self-attention distilling for memory efficiency
   - Handles long sequences efficiently

3. **Flexible Output Types**
   - Price regression (MSE/MAE loss)
   - Direction prediction (binary signals)
   - Portfolio allocation (tanh-bounded positions)

4. **Interpretable Predictions**
   - Attention weights reveal cross-asset dependencies
   - Clear visualization of which assets matter for each prediction

### Comparison with Other Models

| Feature | LSTM | Transformer | TFT | Stockformer |
|---------|------|-------------|-----|-------------|
| Cross-asset modeling | ✗ | ✗ | Limited | ✓ |
| ProbSparse attention | ✗ | ✗ | ✗ | ✓ |
| Multivariate input | ✗ | ✗ | ✓ | ✓ |
| Correlation-aware | ✗ | ✗ | ✗ | ✓ |
| Portfolio allocation | ✗ | ✗ | ✗ | ✓ |

## Stockformer Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         STOCKFORMER                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ Asset 1 │  │ Asset 2 │  │ Asset 3 │  │ Asset N │                 │
│  │ (BTC)   │  │ (ETH)   │  │ (SOL)   │  │  (...)  │                 │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                 │
│       │            │            │            │                        │
│       ▼            ▼            ▼            ▼                        │
│  ┌──────────────────────────────────────────────────┐                │
│  │           Token Embedding (1D-CNN)                │                │
│  │    Extracts temporal patterns per asset           │                │
│  └───────────────────────┬──────────────────────────┘                │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────┐                │
│  │         Cross-Ticker Self-Attention               │                │
│  │    Models relationships between all assets        │                │
│  │    (which assets predict which?)                  │                │
│  └───────────────────────┬──────────────────────────┘                │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────┐                │
│  │    Encoder Stack (ProbSparse or Full Attention)   │                │
│  │         + Self-Attention Distilling               │                │
│  └───────────────────────┬──────────────────────────┘                │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────┐                │
│  │              Prediction Head                      │                │
│  │    • Price regression (MSE/MAE)                   │                │
│  │    • Direction signal (binary)                    │                │
│  │    • Portfolio allocation (tanh)                  │                │
│  └──────────────────────────────────────────────────┘                │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Token Embedding Layer

Stockformer uses 1D-CNN based embeddings instead of simple linear projections:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3):
        super().__init__()
        # Separate kernel for each input channel (asset)
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x):
        # x: [batch, seq_len, n_assets]
        x = x.permute(0, 2, 1)  # [batch, n_assets, seq_len]
        x = self.conv(x)        # [batch, d_model, seq_len]
        return x.permute(0, 2, 1)  # [batch, seq_len, d_model]
```

**Why 1D-CNN?**
- Preserves temporal relationships while extracting local patterns
- Learns different kernels for each asset
- More efficient than position-wise fully connected layers

### Cross-Ticker Attention

The key innovation: attention across both time AND assets:

```python
class CrossTickerAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_assets):
        super().__init__()
        self.n_assets = n_assets
        self.mha = nn.MultiheadAttention(d_model, n_heads)

    def forward(self, x):
        # x: [batch, seq_len, n_assets, d_model]
        batch, seq_len, n_assets, d_model = x.shape

        # Reshape for cross-asset attention
        # Treat (seq_len * batch) as batch, n_assets as sequence
        x = x.view(batch * seq_len, n_assets, d_model)

        # Self-attention across assets at each time step
        attn_out, attn_weights = self.mha(x, x, x)

        # attn_weights shows which assets affect which
        return attn_out.view(batch, seq_len, n_assets, d_model), attn_weights
```

The attention weights reveal **which assets influence the prediction**:
- High attention from ETH to BTC → ETH movements help predict BTC
- Useful for understanding market dynamics and building portfolios

### ProbSparse Attention

Standard self-attention has O(L²) complexity. ProbSparse attention reduces this to O(L·log(L)):

```python
class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.factor = factor  # Controls sparsity
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B, L, D = queries.shape

        # Sample top-u keys based on query "spikiness"
        u = int(self.factor * np.log(L))
        u = min(u, L)

        # Calculate attention scores for sampled queries
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.value_proj(values)

        # Measure query spikiness: max(QK^T) - mean(QK^T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(D)
        M = scores.max(-1)[0] - scores.mean(-1)

        # Select top-u queries with highest spikiness
        M_top = M.topk(u, sorted=False)[1]

        # Compute sparse attention only for selected queries
        Q_sampled = Q[torch.arange(B)[:, None], M_top]
        attn = torch.softmax(
            torch.matmul(Q_sampled, K.transpose(-2, -1)) / np.sqrt(D),
            dim=-1
        )

        # Aggregate values
        context = torch.matmul(attn, V)

        return context
```

**Intuition**: Not all queries need full attention computation. "Spiky" queries (those with dominant attention to specific keys) matter most.

### Prediction Head

Stockformer supports multiple output types:

```python
class PredictionHead(nn.Module):
    def __init__(self, d_model, output_type='regression'):
        super().__init__()
        self.output_type = output_type

        if output_type == 'regression':
            # Direct price prediction
            self.head = nn.Linear(d_model, 1)
            self.loss_fn = nn.MSELoss()

        elif output_type == 'direction':
            # Binary up/down classification
            self.head = nn.Linear(d_model, 1)
            self.loss_fn = nn.BCEWithLogitsLoss()

        elif output_type == 'allocation':
            # Portfolio weights via tanh
            self.head = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Tanh()  # Bounds output to [-1, 1]
            )
            self.loss_fn = lambda pred, ret: -torch.mean(pred * ret)  # Maximize returns

    def forward(self, x):
        return self.head(x)
```

## Multivariate Data Representation

### Log Percent Change

Raw prices are transformed for stable training:

```python
def log_percent_change(close, open_price):
    """
    Transform prices to log percent change.

    Advantages:
    - Variance stabilization across different price scales
    - BTC at $40,000 and ETH at $2,000 become comparable
    - Stationary series (important for neural networks)
    """
    return np.log(close / open_price + 1)
```

### Multi-Ticker Inputs

Structure data for multivariate prediction:

```python
# Data structure for Stockformer
# Shape: [batch, seq_len, n_assets, features]

data = {
    'prices': torch.tensor([
        # Time step 1
        [[45000, 2500, 100],   # [BTC, ETH, SOL] close prices
         [44800, 2480, 99]],   # Time step 2
         # ...
    ]),
    'volumes': torch.tensor([...]),
    'returns': torch.tensor([...]),
}
```

### Feature Engineering

Recommended features for each asset:

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `log_return` | Log price change | ln(close/prev_close) |
| `volume_change` | Relative volume | vol/vol_ma_20 |
| `volatility` | Rolling volatility | std(returns, 20) |
| `rsi` | Relative Strength | Standard RSI calculation |
| `correlation` | Pairwise correlation | rolling_corr(asset_i, asset_j) |
| `funding_rate` | Crypto funding | From exchange API |
| `open_interest` | Derivatives OI | From exchange API |

## Practical Examples

### 01: Data Preparation

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Dict

def prepare_multivariate_data(
    symbols: List[str],
    lookback: int = 168,  # 7 days of hourly data
    horizon: int = 24     # Predict 24 hours ahead
) -> Dict:
    """
    Prepare data for Stockformer training.

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        lookback: Number of historical time steps
        horizon: Prediction horizon

    Returns:
        Dictionary with X (features) and y (targets)
    """

    all_data = []

    for symbol in symbols:
        # Load data from Bybit or other source
        df = load_bybit_data(symbol)

        # Calculate features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_change'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility'] = df['log_return'].rolling(20).std()

        all_data.append(df)

    # Align all dataframes on timestamp
    aligned_data = pd.concat(all_data, axis=1, keys=symbols)

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(aligned_data) - horizon):
        X.append(aligned_data.iloc[i-lookback:i].values)
        y.append(aligned_data.iloc[i+horizon]['log_return'].values)

    return {
        'X': np.array(X),  # [n_samples, lookback, n_assets * n_features]
        'y': np.array(y),  # [n_samples, n_assets]
        'symbols': symbols
    }
```

### 02: Stockformer Architecture

See [python/stockformer.py](python/stockformer.py) for complete implementation.

### 03: Model Training

```python
# python/03_train_model.py

import torch
from stockformer import Stockformer

# Model configuration
config = {
    'n_assets': 5,
    'd_model': 128,
    'n_heads': 8,
    'n_encoder_layers': 3,
    'dropout': 0.1,
    'attention_type': 'probsparse',
    'output_type': 'allocation'
}

# Initialize model
model = Stockformer(**config)

# Training loop with learning rate scheduling
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

for epoch in range(100):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_x)

        # Calculate loss (depends on output_type)
        loss = model.compute_loss(predictions, batch_y)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Validation
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)
```

### 04: Cross-Asset Prediction

```python
# python/04_cross_asset_prediction.py

def predict_with_attention(model, X):
    """
    Make predictions and return attention weights.
    """
    model.eval()
    with torch.no_grad():
        predictions, attention_weights = model(X, return_attention=True)

    # attention_weights: [batch, n_heads, n_assets, n_assets]
    # Shows which assets influence which predictions

    return predictions, attention_weights

def visualize_attention(attention_weights, symbols):
    """
    Create heatmap of cross-asset attention.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Average over heads and batch
    avg_attention = attention_weights.mean(dim=[0, 1]).numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        avg_attention,
        xticklabels=symbols,
        yticklabels=symbols,
        annot=True,
        cmap='Blues'
    )
    plt.title('Cross-Asset Attention Weights')
    plt.xlabel('Key (Source)')
    plt.ylabel('Query (Target)')
    plt.savefig('attention_heatmap.png')
```

### 05: Portfolio Backtesting

```python
# python/05_backtest.py

def backtest_stockformer_strategy(
    model,
    test_data,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001
):
    """
    Backtest Stockformer portfolio allocation strategy.
    """
    capital = initial_capital
    positions = np.zeros(model.n_assets)

    results = []

    for i, (X, returns) in enumerate(test_data):
        # Get model allocation weights
        weights = model(X).numpy().flatten()  # [-1, 1] per asset

        # Normalize to sum to 1 (long-only) or allow shorting
        weights = weights / np.abs(weights).sum()

        # Calculate position changes and costs
        position_change = weights - positions
        costs = np.abs(position_change).sum() * transaction_cost * capital

        # Update positions
        positions = weights

        # Calculate returns
        portfolio_return = np.sum(positions * returns)
        capital = capital * (1 + portfolio_return) - costs

        results.append({
            'capital': capital,
            'weights': weights.copy(),
            'return': portfolio_return
        })

    return pd.DataFrame(results)
```

## Rust Implementation

See [rust_stockformer](rust_stockformer/) for complete Rust implementation using Bybit data.

```
rust_stockformer/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP client for Bybit
│   │   └── types.rs        # API response types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading utilities
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset for training
│   ├── model/              # Stockformer architecture
│   │   ├── mod.rs
│   │   ├── embedding.rs    # Token embedding layer
│   │   ├── attention.rs    # Cross-ticker & ProbSparse attention
│   │   ├── encoder.rs      # Encoder stack
│   │   └── stockformer.rs  # Complete model
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── fetch_data.rs       # Download Bybit data
    ├── train.rs            # Train model
    └── backtest.rs         # Run backtest
```

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd rust_stockformer

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Train model
cargo run --example train -- --epochs 100 --batch-size 32

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── stockformer.py          # Main model implementation
├── data_loader.py          # Bybit data loading
├── features.py             # Feature engineering
├── train.py                # Training script
├── backtest.py             # Backtesting utilities
├── requirements.txt        # Dependencies
└── examples/
    ├── 01_data_preparation.ipynb
    ├── 02_model_architecture.ipynb
    ├── 03_training.ipynb
    ├── 04_prediction.ipynb
    └── 05_backtesting.ipynb
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch data
python data_loader.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Train model
python train.py --config configs/default.yaml

# Run backtest
python backtest.py --model checkpoints/best_model.pt
```

## Best Practices

### When to Use Stockformer

**Good use cases:**
- Trading correlated assets (crypto, sector ETFs)
- Portfolio allocation across multiple assets
- Discovering cross-asset dependencies
- Long-term predictions (hours to days)

**Not ideal for:**
- High-frequency trading (inference latency)
- Single asset prediction (use simpler models)
- Very small datasets (<1000 samples)

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `d_model` | 128-256 | Match with n_heads |
| `n_heads` | 8 | Should divide d_model |
| `n_assets` | 5-20 | More requires more data |
| `lookback` | 168 (7d hourly) | Depends on data frequency |
| `dropout` | 0.1-0.2 | Higher for small datasets |

### Common Pitfalls

1. **Gradient instability**: Use gradient clipping and learning rate scheduling
2. **Overfitting**: Apply dropout, use early stopping
3. **Data leakage**: Ensure proper train/val/test splits
4. **Correlation collapse**: Monitor attention weights for diversity

## Resources

### Papers

- [Transformer Based Time-Series Forecasting For Stock](https://arxiv.org/abs/2502.09625) — Original Stockformer paper
- [Stockformer: A Price-Volume Factor Stock Selection Model](https://arxiv.org/abs/2401.06139) — Advanced variant with wavelet transform
- [MASTER: Market-Guided Stock Transformer](https://arxiv.org/abs/2312.15235) — Related architecture
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer

### Implementations

- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — Time series library
- [Informer](https://github.com/zhouhaoyi/Informer2020) — ProbSparse attention implementation
- [Autoformer](https://github.com/thuml/Autoformer) — Related architecture

### Related Chapters

- [Chapter 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Multi-horizon forecasting
- [Chapter 41: Higher Order Transformers](../41_higher_order_transformers) — Advanced attention mechanisms
- [Chapter 47: Cross-Attention Multi-Asset](../47_cross_attention_multi_asset) — Cross-asset modeling

---

## Difficulty Level

**Advanced**

Prerequisites:
- Transformer architecture and attention mechanisms
- Time series forecasting basics
- Portfolio theory fundamentals
- PyTorch/Rust ML libraries
