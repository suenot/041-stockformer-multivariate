//! Модуль торговых стратегий
//!
//! Содержит реализации:
//! - Генерация торговых сигналов
//! - Бэктестинг стратегий
//! - Управление портфелем

mod signals;
mod backtest;

pub use signals::{Signal, SignalType, SignalGenerator, PortfolioSignal};
pub use backtest::{BacktestConfig, BacktestResult, Backtester, Trade};
