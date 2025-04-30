//! Бэктестинг торговых стратегий
//!
//! Симуляция торговли на исторических данных

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::strategy::{PortfolioSignal, Signal, SignalType};

/// Конфигурация бэктестинга
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Начальный капитал
    pub initial_capital: f64,
    /// Комиссия за сделку (в долях, например 0.001 = 0.1%)
    pub commission: f64,
    /// Проскальзывание (в долях)
    pub slippage: f64,
    /// Максимальное плечо
    pub max_leverage: f64,
    /// Разрешить короткие позиции
    pub allow_short: bool,
    /// Минимальный размер позиции
    pub min_position_size: f64,
    /// Использовать стоп-лосс
    pub use_stop_loss: bool,
    /// Уровень стоп-лосса (в долях)
    pub stop_loss_level: f64,
    /// Использовать тейк-профит
    pub use_take_profit: bool,
    /// Уровень тейк-профита (в долях)
    pub take_profit_level: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            commission: 0.001,
            slippage: 0.0005,
            max_leverage: 1.0,
            allow_short: false,
            min_position_size: 0.01,
            use_stop_loss: true,
            stop_loss_level: 0.05,
            use_take_profit: true,
            take_profit_level: 0.10,
        }
    }
}

/// Информация о сделке
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Индекс тикера
    pub ticker_idx: usize,
    /// Время входа (индекс)
    pub entry_time: usize,
    /// Время выхода (индекс)
    pub exit_time: usize,
    /// Цена входа
    pub entry_price: f64,
    /// Цена выхода
    pub exit_price: f64,
    /// Размер позиции (в единицах капитала)
    pub position_size: f64,
    /// Направление (1 = long, -1 = short)
    pub direction: i32,
    /// Прибыль/убыток
    pub pnl: f64,
    /// Доходность в процентах
    pub return_pct: f64,
    /// Причина закрытия
    pub exit_reason: String,
}

/// Результаты бэктестинга
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Список всех сделок
    pub trades: Vec<Trade>,
    /// Кривая капитала
    pub equity_curve: Vec<f64>,
    /// Общая доходность
    pub total_return: f64,
    /// Годовая доходность
    pub annual_return: f64,
    /// Волатильность (годовая)
    pub volatility: f64,
    /// Коэффициент Шарпа
    pub sharpe_ratio: f64,
    /// Коэффициент Сортино
    pub sortino_ratio: f64,
    /// Максимальная просадка
    pub max_drawdown: f64,
    /// Коэффициент Кальмара
    pub calmar_ratio: f64,
    /// Процент выигрышных сделок
    pub win_rate: f64,
    /// Средняя прибыль на сделку
    pub avg_profit: f64,
    /// Средний убыток на сделку
    pub avg_loss: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Количество сделок
    pub num_trades: usize,
    /// Средняя длительность сделки
    pub avg_trade_duration: f64,
}

impl BacktestResult {
    /// Создает пустой результат
    pub fn empty(initial_capital: f64) -> Self {
        Self {
            trades: Vec::new(),
            equity_curve: vec![initial_capital],
            total_return: 0.0,
            annual_return: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            win_rate: 0.0,
            avg_profit: 0.0,
            avg_loss: 0.0,
            profit_factor: 0.0,
            num_trades: 0,
            avg_trade_duration: 0.0,
        }
    }

    /// Выводит сводку результатов
    pub fn summary(&self) -> String {
        format!(
            r#"Backtest Results Summary
========================
Total Return: {:.2}%
Annual Return: {:.2}%
Volatility: {:.2}%
Sharpe Ratio: {:.2}
Sortino Ratio: {:.2}
Max Drawdown: {:.2}%
Calmar Ratio: {:.2}
Win Rate: {:.2}%
Profit Factor: {:.2}
Number of Trades: {}
Avg Trade Duration: {:.1} periods
"#,
            self.total_return * 100.0,
            self.annual_return * 100.0,
            self.volatility * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.calmar_ratio,
            self.win_rate * 100.0,
            self.profit_factor,
            self.num_trades,
            self.avg_trade_duration,
        )
    }
}

/// Открытая позиция
#[derive(Debug, Clone)]
struct Position {
    ticker_idx: usize,
    entry_time: usize,
    entry_price: f64,
    size: f64,
    direction: i32,
}

/// Бэктестер
#[derive(Debug)]
pub struct Backtester {
    config: BacktestConfig,
    positions: HashMap<usize, Position>,
    capital: f64,
    equity_curve: Vec<f64>,
    trades: Vec<Trade>,
}

impl Backtester {
    /// Создает новый бэктестер
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            capital: config.initial_capital,
            config,
            positions: HashMap::new(),
            equity_curve: Vec::new(),
            trades: Vec::new(),
        }
    }

    /// Запускает бэктест
    ///
    /// # Arguments
    /// * `prices` - Матрица цен [time, num_tickers]
    /// * `signals` - Сигналы для каждого временного шага
    /// * `periods_per_year` - Количество периодов в году (для расчета годовой доходности)
    pub fn run(
        &mut self,
        prices: &Array2<f64>,
        signals: &[PortfolioSignal],
        periods_per_year: usize,
    ) -> BacktestResult {
        let (num_periods, num_tickers) = prices.dim();

        if signals.len() != num_periods {
            return BacktestResult::empty(self.config.initial_capital);
        }

        self.equity_curve.push(self.capital);

        for t in 0..num_periods {
            let current_prices: Vec<f64> = (0..num_tickers).map(|i| prices[[t, i]]).collect();

            // Проверяем стоп-лоссы и тейк-профиты
            if t > 0 {
                self.check_exits(t, &current_prices);
            }

            // Обрабатываем сигналы
            self.process_signals(t, &signals[t], &current_prices);

            // Обновляем equity
            let portfolio_value = self.calculate_portfolio_value(&current_prices);
            self.equity_curve.push(portfolio_value);
        }

        // Закрываем все оставшиеся позиции
        let final_prices: Vec<f64> = (0..num_tickers)
            .map(|i| prices[[num_periods - 1, i]])
            .collect();
        self.close_all_positions(num_periods - 1, &final_prices, "End of backtest");

        // Вычисляем метрики
        self.calculate_metrics(periods_per_year)
    }

    /// Проверяет и исполняет стоп-лоссы/тейк-профиты
    fn check_exits(&mut self, time: usize, prices: &[f64]) {
        let positions_to_close: Vec<(usize, String)> = self
            .positions
            .iter()
            .filter_map(|(&ticker_idx, pos)| {
                let current_price = prices[ticker_idx];
                let return_pct = (current_price - pos.entry_price) / pos.entry_price
                    * pos.direction as f64;

                if self.config.use_stop_loss && return_pct <= -self.config.stop_loss_level {
                    Some((ticker_idx, "Stop Loss".to_string()))
                } else if self.config.use_take_profit && return_pct >= self.config.take_profit_level
                {
                    Some((ticker_idx, "Take Profit".to_string()))
                } else {
                    None
                }
            })
            .collect();

        for (ticker_idx, reason) in positions_to_close {
            self.close_position(ticker_idx, time, prices[ticker_idx], &reason);
        }
    }

    /// Обрабатывает сигналы
    fn process_signals(&mut self, time: usize, portfolio_signal: &PortfolioSignal, prices: &[f64]) {
        for signal in &portfolio_signal.signals {
            let ticker_idx = signal.ticker_idx;
            let current_price = prices[ticker_idx];

            match signal.signal_type {
                SignalType::Long => {
                    if !self.positions.contains_key(&ticker_idx) {
                        let position_value = self.capital * signal.position_size;
                        if position_value >= self.config.min_position_size * self.capital {
                            self.open_position(ticker_idx, time, current_price, position_value, 1);
                        }
                    }
                }
                SignalType::Short if self.config.allow_short => {
                    if !self.positions.contains_key(&ticker_idx) {
                        let position_value = self.capital * signal.position_size;
                        if position_value >= self.config.min_position_size * self.capital {
                            self.open_position(ticker_idx, time, current_price, position_value, -1);
                        }
                    }
                }
                SignalType::Close => {
                    if self.positions.contains_key(&ticker_idx) {
                        self.close_position(ticker_idx, time, current_price, "Signal");
                    }
                }
                _ => {}
            }
        }
    }

    /// Открывает позицию
    fn open_position(
        &mut self,
        ticker_idx: usize,
        time: usize,
        price: f64,
        size: f64,
        direction: i32,
    ) {
        // Учитываем комиссию и проскальзывание
        let adjusted_price = price * (1.0 + self.config.slippage * direction as f64);
        let commission = size * self.config.commission;

        self.capital -= commission;

        let position = Position {
            ticker_idx,
            entry_time: time,
            entry_price: adjusted_price,
            size,
            direction,
        };

        self.positions.insert(ticker_idx, position);
    }

    /// Закрывает позицию
    fn close_position(&mut self, ticker_idx: usize, time: usize, price: f64, reason: &str) {
        if let Some(position) = self.positions.remove(&ticker_idx) {
            // Учитываем проскальзывание
            let adjusted_price = price * (1.0 - self.config.slippage * position.direction as f64);

            // Рассчитываем PnL
            let price_change = (adjusted_price - position.entry_price) / position.entry_price;
            let pnl = position.size * price_change * position.direction as f64;

            // Комиссия на выход
            let commission = position.size * self.config.commission;

            self.capital += position.size + pnl - commission;

            let trade = Trade {
                ticker_idx,
                entry_time: position.entry_time,
                exit_time: time,
                entry_price: position.entry_price,
                exit_price: adjusted_price,
                position_size: position.size,
                direction: position.direction,
                pnl,
                return_pct: price_change * position.direction as f64,
                exit_reason: reason.to_string(),
            };

            self.trades.push(trade);
        }
    }

    /// Закрывает все позиции
    fn close_all_positions(&mut self, time: usize, prices: &[f64], reason: &str) {
        let tickers: Vec<usize> = self.positions.keys().cloned().collect();
        for ticker_idx in tickers {
            self.close_position(ticker_idx, time, prices[ticker_idx], reason);
        }
    }

    /// Рассчитывает текущую стоимость портфеля
    fn calculate_portfolio_value(&self, prices: &[f64]) -> f64 {
        let mut value = self.capital;

        for (ticker_idx, position) in &self.positions {
            let current_price = prices[*ticker_idx];
            let price_change = (current_price - position.entry_price) / position.entry_price;
            let unrealized_pnl = position.size * price_change * position.direction as f64;
            value += position.size + unrealized_pnl;
        }

        value
    }

    /// Рассчитывает метрики бэктеста
    fn calculate_metrics(&self, periods_per_year: usize) -> BacktestResult {
        let initial = self.config.initial_capital;
        let final_value = *self.equity_curve.last().unwrap_or(&initial);

        // Доходности
        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let total_return = (final_value - initial) / initial;
        let num_periods = self.equity_curve.len() as f64;
        let annual_return =
            (1.0 + total_return).powf(periods_per_year as f64 / num_periods) - 1.0;

        // Волатильность
        let mean_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len().max(1) as f64;
        let volatility = variance.sqrt() * (periods_per_year as f64).sqrt();

        // Sharpe Ratio (assuming risk-free rate = 0)
        let sharpe_ratio = if volatility > 0.0 {
            annual_return / volatility
        } else {
            0.0
        };

        // Sortino Ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_std = downside_variance.sqrt() * (periods_per_year as f64).sqrt();
        let sortino_ratio = if downside_std > 0.0 {
            annual_return / downside_std
        } else {
            0.0
        };

        // Max Drawdown
        let mut max_drawdown = 0.0;
        let mut peak = self.equity_curve[0];
        for &value in &self.equity_curve {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calmar Ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annual_return / max_drawdown
        } else {
            0.0
        };

        // Trade statistics
        let winning_trades: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = if !self.trades.is_empty() {
            winning_trades.len() as f64 / self.trades.len() as f64
        } else {
            0.0
        };

        let avg_profit = if !winning_trades.is_empty() {
            winning_trades.iter().map(|t| t.pnl).sum::<f64>() / winning_trades.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing_trades.is_empty() {
            losing_trades.iter().map(|t| t.pnl.abs()).sum::<f64>() / losing_trades.len() as f64
        } else {
            0.0
        };

        let total_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let total_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if total_loss > 0.0 {
            total_profit / total_loss
        } else if total_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_duration = if !self.trades.is_empty() {
            self.trades
                .iter()
                .map(|t| (t.exit_time - t.entry_time) as f64)
                .sum::<f64>()
                / self.trades.len() as f64
        } else {
            0.0
        };

        BacktestResult {
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
            total_return,
            annual_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            avg_profit,
            avg_loss,
            profit_factor,
            num_trades: self.trades.len(),
            avg_trade_duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_prices(num_periods: usize, num_tickers: usize, trend: f64) -> Array2<f64> {
        let mut prices = Array2::zeros((num_periods, num_tickers));

        for t in 0..num_tickers {
            let base_price = 100.0 + t as f64 * 10.0;
            for p in 0..num_periods {
                // Тренд + небольшой шум
                let noise = (rand::random::<f64>() - 0.5) * 2.0;
                prices[[p, t]] = base_price * (1.0 + trend * p as f64 / 100.0 + noise / 100.0);
            }
        }

        prices
    }

    fn create_test_signals(
        num_periods: usize,
        num_tickers: usize,
        long_tickers: Vec<usize>,
    ) -> Vec<PortfolioSignal> {
        (0..num_periods)
            .map(|_| {
                let signals: Vec<Signal> = (0..num_tickers)
                    .map(|t| {
                        let signal_type = if long_tickers.contains(&t) {
                            SignalType::Long
                        } else {
                            SignalType::Hold
                        };
                        Signal::new(signal_type, t, 0.5, 0.01, 0.7)
                    })
                    .collect();

                let weights = vec![1.0 / num_tickers as f64; num_tickers];
                PortfolioSignal::new(signals, weights)
            })
            .collect()
    }

    #[test]
    fn test_backtest_basic() {
        let config = BacktestConfig::default();
        let mut backtester = Backtester::new(config);

        let prices = create_test_prices(100, 3, 0.5); // Positive trend
        let signals = create_test_signals(100, 3, vec![0]); // Long first ticker

        let result = backtester.run(&prices, &signals, 252);

        assert!(result.num_trades > 0);
        assert!(!result.equity_curve.is_empty());
    }

    #[test]
    fn test_backtest_with_stop_loss() {
        let config = BacktestConfig {
            use_stop_loss: true,
            stop_loss_level: 0.02, // 2% stop loss
            ..Default::default()
        };

        let mut backtester = Backtester::new(config);

        // Создаем падающий тренд
        let prices = create_test_prices(100, 2, -1.0);
        let signals = create_test_signals(100, 2, vec![0]);

        let result = backtester.run(&prices, &signals, 252);

        // Должны быть сделки с выходом по стоп-лоссу
        let stop_loss_trades: Vec<&Trade> = result
            .trades
            .iter()
            .filter(|t| t.exit_reason == "Stop Loss")
            .collect();

        // В падающем рынке стоп-лоссы должны срабатывать
        assert!(stop_loss_trades.len() > 0 || result.trades.is_empty());
    }

    #[test]
    fn test_backtest_metrics() {
        let config = BacktestConfig::default();
        let mut backtester = Backtester::new(config.clone());

        let prices = create_test_prices(252, 3, 0.3);
        let signals = create_test_signals(252, 3, vec![0, 1]);

        let result = backtester.run(&prices, &signals, 252);

        // Проверяем, что метрики вычислены
        assert!(result.volatility >= 0.0);
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
    }

    #[test]
    fn test_backtest_empty_signals() {
        let config = BacktestConfig::default();
        let mut backtester = Backtester::new(config.clone());

        let prices = create_test_prices(100, 2, 0.0);
        let signals = create_test_signals(100, 2, vec![]); // No long signals

        let result = backtester.run(&prices, &signals, 252);

        // Без сигналов не должно быть сделок
        assert_eq!(result.num_trades, 0);
        assert!((result.total_return).abs() < 0.01); // Почти нулевой возврат
    }

    #[test]
    fn test_trade_structure() {
        let trade = Trade {
            ticker_idx: 0,
            entry_time: 10,
            exit_time: 20,
            entry_price: 100.0,
            exit_price: 110.0,
            position_size: 10000.0,
            direction: 1,
            pnl: 1000.0,
            return_pct: 0.10,
            exit_reason: "Take Profit".to_string(),
        };

        assert_eq!(trade.return_pct, 0.10);
        assert_eq!(trade.exit_time - trade.entry_time, 10);
    }

    #[test]
    fn test_backtest_result_summary() {
        let result = BacktestResult {
            trades: vec![],
            equity_curve: vec![100000.0, 110000.0],
            total_return: 0.10,
            annual_return: 0.12,
            volatility: 0.15,
            sharpe_ratio: 0.80,
            sortino_ratio: 1.20,
            max_drawdown: 0.05,
            calmar_ratio: 2.40,
            win_rate: 0.60,
            avg_profit: 500.0,
            avg_loss: 300.0,
            profit_factor: 1.67,
            num_trades: 50,
            avg_trade_duration: 5.0,
        };

        let summary = result.summary();
        assert!(summary.contains("Total Return: 10.00%"));
        assert!(summary.contains("Sharpe Ratio: 0.80"));
    }
}
