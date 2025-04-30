# Stockformer Rust Implementation

Rust implementation of the Stockformer multivariate stock prediction model with Bybit exchange integration.

## Features

- **Multi-asset prediction**: Predict multiple assets simultaneously using cross-ticker attention
- **ProbSparse Attention**: Efficient O(L·log(L)) attention mechanism
- **Token Embedding**: 1D-CNN based embedding for time series
- **Trading Signals**: Automatic signal generation from predictions
- **Backtesting**: Full backtesting framework with risk management

## Project Structure

```
rust_stockformer/
├── src/
│   ├── lib.rs              # Library entry point
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP client
│   │   └── types.rs        # API response types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── features.rs     # Feature engineering
│   │   ├── loader.rs       # Multi-asset data loader
│   │   └── dataset.rs      # Dataset structures
│   ├── model/              # Stockformer architecture
│   │   ├── mod.rs
│   │   ├── config.rs       # Model configuration
│   │   ├── attention.rs    # Attention mechanisms
│   │   ├── embedding.rs    # Token embedding
│   │   └── stockformer.rs  # Main model
│   └── strategy/           # Trading strategies
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting framework
└── examples/
    ├── fetch_data.rs       # Data fetching example
    ├── train.rs            # Training demonstration
    ├── predict.rs          # Prediction example
    └── backtest.rs         # Backtesting example
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rust_stockformer = { path = "path/to/rust_stockformer" }
```

Or build from source:

```bash
cd rust_stockformer
cargo build --release
```

## Quick Start

### Fetching Data

```rust
use rust_stockformer::api::{BybitClient, Interval};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = BybitClient::new();

    // Get current ticker
    let ticker = client.get_ticker("BTCUSDT").await?;
    println!("BTC Price: ${:.2}", ticker.last_price);

    // Get historical klines
    let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(100)).await?;
    println!("Loaded {} candles", klines.len());

    Ok(())
}
```

### Creating a Model

```rust
use rust_stockformer::model::{StockformerConfig, StockformerModel, OutputType};

let config = StockformerConfig {
    num_tickers: 5,
    seq_len: 96,
    input_features: 6,
    d_model: 64,
    num_heads: 4,
    d_ff: 256,
    num_encoder_layers: 2,
    output_type: OutputType::Regression,
    use_cross_ticker_attention: true,
    ..Default::default()
};

let model = StockformerModel::new(config)?;
```

### Making Predictions

```rust
use ndarray::Array4;

// Input: [batch, num_tickers, seq_len, features]
let input = Array4::zeros((1, 5, 96, 6));
let result = model.forward(&input);

println!("Predictions: {:?}", result.predictions);
println!("Cross-ticker attention available: {}",
    result.attention_weights.cross_ticker_weights.is_some());
```

### Generating Trading Signals

```rust
use rust_stockformer::strategy::{SignalGenerator, SignalType};

let signal_gen = SignalGenerator::new()
    .with_thresholds(0.005, -0.005)
    .with_min_confidence(0.3);

let portfolio = signal_gen.generate(
    &result,
    OutputType::Regression,
    &["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>(),
);

for signal in &portfolio.signals {
    match signal.signal_type {
        SignalType::Long => println!("BUY signal for ticker {}", signal.ticker_idx),
        SignalType::Short => println!("SELL signal for ticker {}", signal.ticker_idx),
        _ => {}
    }
}
```

### Running Backtest

```rust
use rust_stockformer::strategy::{BacktestConfig, Backtester};

let config = BacktestConfig {
    initial_capital: 100_000.0,
    commission: 0.001,
    slippage: 0.0005,
    use_stop_loss: true,
    stop_loss_level: 0.05,
    ..Default::default()
};

let mut backtester = Backtester::new(config);
let result = backtester.run(&prices, &signals, 8760);

println!("{}", result.summary());
println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
println!("Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
```

## Examples

Run the examples:

```bash
# Fetch data from Bybit
cargo run --example fetch_data

# Demonstrate model training
cargo run --example train

# Make predictions and generate signals
cargo run --example predict

# Run backtest
cargo run --example backtest
```

## Configuration Options

### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_tickers` | Number of assets | 5 |
| `seq_len` | Input sequence length | 96 |
| `input_features` | Features per asset | 6 |
| `d_model` | Model dimension | 64 |
| `num_heads` | Attention heads | 4 |
| `d_ff` | Feed-forward dimension | 256 |
| `num_encoder_layers` | Encoder layers | 2 |
| `dropout` | Dropout probability | 0.1 |
| `attention_type` | `ProbSparse`, `Full`, `Local` | `ProbSparse` |
| `output_type` | `Regression`, `Direction`, `Portfolio`, `Quantile` | `Regression` |

### Output Types

- **Regression**: Predict continuous returns
- **Direction**: Classify up/down/hold
- **Portfolio**: Output portfolio weights (sum to 1)
- **Quantile**: Predict confidence intervals

### Backtest Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `initial_capital` | Starting capital | 100,000 |
| `commission` | Trading fee (fraction) | 0.001 |
| `slippage` | Price slippage | 0.0005 |
| `max_leverage` | Maximum leverage | 1.0 |
| `allow_short` | Enable short selling | false |
| `stop_loss_level` | Stop loss threshold | 0.05 |
| `take_profit_level` | Take profit threshold | 0.10 |

## Feature Engineering

Available features:

```rust
use rust_stockformer::data::FeatureEngineering;

let fe = FeatureEngineering::new();

// Log returns
let returns = fe.log_returns(&prices);

// RSI indicator
let rsi = fe.rsi(&prices, 14);

// Realized volatility
let volatility = fe.realized_volatility(&returns, 20);

// Volume change
let vol_change = fe.volume_change(&volumes);

// Correlation matrix
let corr = fe.correlation_matrix(&asset_prices);
```

## API Reference

### Bybit Client

```rust
// Available intervals
pub enum Interval {
    Min1, Min3, Min5, Min15, Min30,
    Hour1, Hour2, Hour4, Hour6, Hour12,
    Day1, Week1, Month1,
}

// Methods
client.get_ticker(symbol).await?;
client.get_klines(symbol, interval, limit).await?;
client.get_orderbook(symbol, depth).await?;
```

## Testing

Run tests:

```bash
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_model_forward
```

## Performance Notes

- **ProbSparse Attention** reduces complexity from O(L²) to O(L·log(L))
- **Cross-ticker attention** adds O(T²) complexity per timestep
- For large portfolios, consider using `StockformerConfig::small()` for faster inference

## License

MIT License - see the main project LICENSE file.

## References

- [Stockformer: A Swing Trading Strategy Based on STL Decomposition and Self-Attention Networks](https://arxiv.org/abs/2502.09625)
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)
