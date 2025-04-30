//! Пример бэктестинга стратегии на основе Stockformer
//!
//! Демонстрирует полный цикл: загрузка данных → предсказание → торговля → анализ

use ndarray::{Array2, Array4};
use rust_stockformer::api::{BybitClient, Interval};
use rust_stockformer::data::{FeatureEngineering, MultiAssetLoader};
use rust_stockformer::model::{StockformerConfig, StockformerModel, OutputType};
use rust_stockformer::strategy::{
    BacktestConfig, Backtester, SignalGenerator, PortfolioSignal, Signal, SignalType,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Stockformer: Бэктестинг стратегии ===\n");

    // Параметры
    let symbols = vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "SOLUSDT".to_string(),
    ];

    let lookback = 96;  // Окно для модели
    let total_periods = 500; // Общее количество периодов для бэктеста

    println!("Параметры бэктеста:");
    println!("  Тикеры: {:?}", symbols);
    println!("  Lookback: {} периодов", lookback);
    println!("  Всего периодов: {}", total_periods);
    println!();

    // Загружаем или генерируем данные
    println!("Подготовка данных...");
    let (prices, features) = prepare_backtest_data(&symbols, lookback, total_periods).await;

    println!("  Размер цен: {:?}", prices.dim());
    println!();

    // Создаем модель
    let config = StockformerConfig {
        num_tickers: symbols.len(),
        seq_len: lookback,
        input_features: 6,
        d_model: 32,
        num_heads: 4,
        d_ff: 128,
        num_encoder_layers: 2,
        output_type: OutputType::Regression,
        use_cross_ticker_attention: true,
        ..Default::default()
    };

    let model = StockformerModel::new(config.clone())?;
    let signal_gen = SignalGenerator::new()
        .with_thresholds(0.005, -0.005)
        .with_min_confidence(0.3);

    // Генерируем сигналы для каждого периода
    println!("Генерация сигналов...");

    let mut signals: Vec<PortfolioSignal> = Vec::with_capacity(total_periods);

    for t in 0..total_periods {
        if t < lookback {
            // Недостаточно данных - держим
            let hold_signals: Vec<Signal> = (0..symbols.len())
                .map(|i| Signal::new(SignalType::Hold, i, 0.0, 0.0, 0.5))
                .collect();
            signals.push(PortfolioSignal::new(hold_signals, vec![0.0; symbols.len()]));
        } else {
            // Формируем входные данные для модели
            let window = features.slice(ndarray::s![.., .., (t-lookback)..t, ..]).to_owned();

            // Делаем предсказание
            let result = model.forward(&window);

            // Генерируем сигналы
            let portfolio_signal = signal_gen.generate(
                &result,
                OutputType::Regression,
                &symbols,
            );

            signals.push(portfolio_signal);
        }

        if (t + 1) % 100 == 0 {
            println!("  Обработано: {}/{}", t + 1, total_periods);
        }
    }

    println!("  Сигналы сгенерированы: {}", signals.len());
    println!();

    // Запускаем бэктест с разными конфигурациями
    println!("=== Бэктест: Базовая стратегия ===\n");

    let base_config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: 0.001,     // 0.1% комиссия
        slippage: 0.0005,      // 0.05% проскальзывание
        max_leverage: 1.0,     // Без плеча
        allow_short: false,    // Только long
        use_stop_loss: true,
        stop_loss_level: 0.05, // 5% стоп-лосс
        use_take_profit: true,
        take_profit_level: 0.10, // 10% тейк-профит
        ..Default::default()
    };

    let mut backtester = Backtester::new(base_config.clone());
    let result = backtester.run(&prices, &signals, 8760); // 8760 часов в году

    println!("{}", result.summary());

    // Статистика по тикерам
    println!("Статистика по тикерам:");
    for (i, symbol) in symbols.iter().enumerate() {
        let ticker_trades: Vec<_> = result.trades.iter()
            .filter(|t| t.ticker_idx == i)
            .collect();

        let wins = ticker_trades.iter().filter(|t| t.pnl > 0.0).count();
        let total = ticker_trades.len();
        let total_pnl: f64 = ticker_trades.iter().map(|t| t.pnl).sum();

        println!(
            "  {}: {} сделок, Win Rate = {:.1}%, PnL = ${:.2}",
            symbol,
            total,
            if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 },
            total_pnl
        );
    }
    println!();

    // Бэктест с короткими позициями
    println!("=== Бэктест: Long/Short стратегия ===\n");

    let ls_config = BacktestConfig {
        allow_short: true,
        ..base_config.clone()
    };

    let mut ls_backtester = Backtester::new(ls_config);
    let ls_result = ls_backtester.run(&prices, &signals, 8760);

    println!("{}", ls_result.summary());

    // Бэктест без стоп-лоссов
    println!("=== Бэктест: Без стоп-лоссов ===\n");

    let no_sl_config = BacktestConfig {
        use_stop_loss: false,
        use_take_profit: false,
        ..base_config.clone()
    };

    let mut no_sl_backtester = Backtester::new(no_sl_config);
    let no_sl_result = no_sl_backtester.run(&prices, &signals, 8760);

    println!("{}", no_sl_result.summary());

    // Сравнение стратегий
    println!("=== Сравнение стратегий ===\n");

    println!(
        "{:<25} {:>12} {:>12} {:>12} {:>12}",
        "Стратегия", "Доходность", "Sharpe", "MaxDD", "Сделки"
    );
    println!("{}", "-".repeat(75));

    println!(
        "{:<25} {:>11.2}% {:>12.2} {:>11.2}% {:>12}",
        "Базовая (Long only)",
        result.total_return * 100.0,
        result.sharpe_ratio,
        result.max_drawdown * 100.0,
        result.num_trades
    );

    println!(
        "{:<25} {:>11.2}% {:>12.2} {:>11.2}% {:>12}",
        "Long/Short",
        ls_result.total_return * 100.0,
        ls_result.sharpe_ratio,
        ls_result.max_drawdown * 100.0,
        ls_result.num_trades
    );

    println!(
        "{:<25} {:>11.2}% {:>12.2} {:>11.2}% {:>12}",
        "Без стоп-лоссов",
        no_sl_result.total_return * 100.0,
        no_sl_result.sharpe_ratio,
        no_sl_result.max_drawdown * 100.0,
        no_sl_result.num_trades
    );

    // Анализ кривой капитала
    println!("\n=== Анализ кривой капитала ===\n");

    let equity = &result.equity_curve;
    if !equity.is_empty() {
        let min_equity = equity.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_equity = equity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("Начальный капитал: ${:.2}", equity[0]);
        println!("Конечный капитал:  ${:.2}", equity.last().unwrap_or(&0.0));
        println!("Минимум:           ${:.2}", min_equity);
        println!("Максимум:          ${:.2}", max_equity);

        // Простая визуализация
        println!("\nКривая капитала (упрощенно):");

        let samples = 20;
        let step = equity.len() / samples;

        for i in 0..samples {
            let idx = i * step;
            let value = equity.get(idx).unwrap_or(&0.0);
            let normalized = (value - min_equity) / (max_equity - min_equity).max(1.0);
            let bar_len = (normalized * 30.0).round() as usize;
            let bar: String = "█".repeat(bar_len);

            println!("{:>4}: ${:>10.0} {}", idx, value, bar);
        }
    }

    // Анализ лучших и худших сделок
    println!("\n=== Лучшие и худшие сделки ===\n");

    let mut sorted_trades = result.trades.clone();
    sorted_trades.sort_by(|a, b| b.pnl.partial_cmp(&a.pnl).unwrap());

    println!("Топ-5 лучших сделок:");
    for trade in sorted_trades.iter().take(5) {
        println!(
            "  {}: PnL = ${:.2} ({:.2}%), Длительность = {} периодов, {}",
            symbols[trade.ticker_idx],
            trade.pnl,
            trade.return_pct * 100.0,
            trade.exit_time - trade.entry_time,
            trade.exit_reason
        );
    }

    println!("\nТоп-5 худших сделок:");
    for trade in sorted_trades.iter().rev().take(5) {
        println!(
            "  {}: PnL = ${:.2} ({:.2}%), Длительность = {} периодов, {}",
            symbols[trade.ticker_idx],
            trade.pnl,
            trade.return_pct * 100.0,
            trade.exit_time - trade.entry_time,
            trade.exit_reason
        );
    }

    println!("\n=== Готово ===");

    Ok(())
}

/// Подготавливает данные для бэктеста
async fn prepare_backtest_data(
    symbols: &[String],
    lookback: usize,
    total_periods: usize,
) -> (Array2<f64>, Array4<f64>) {
    // Пробуем загрузить реальные данные
    let client = BybitClient::new();

    // Генерируем синтетические данные для демонстрации
    // В реальном использовании здесь будет загрузка исторических данных
    println!("Генерация синтетических данных для демонстрации...");

    let num_tickers = symbols.len();
    let num_features = 6;

    // Матрица цен [time, tickers]
    let mut prices = Array2::zeros((total_periods, num_tickers));

    // Признаки [batch=1, tickers, time, features]
    let mut features = Array4::zeros((1, num_tickers, total_periods, num_features));

    // Параметры генерации
    let base_prices: Vec<f64> = vec![45000.0, 3000.0, 100.0]; // BTC, ETH, SOL примерные цены
    let volatilities: Vec<f64> = vec![0.02, 0.025, 0.04]; // Волатильность
    let correlations: Vec<f64> = vec![1.0, 0.85, 0.75]; // Корреляция с BTC

    // Генерируем коррелированные цены
    let mut btc_shocks = Vec::with_capacity(total_periods);
    for _ in 0..total_periods {
        btc_shocks.push(rand_normal());
    }

    for t in 0..num_tickers {
        let base = base_prices.get(t).unwrap_or(&100.0);
        let vol = volatilities.get(t).unwrap_or(&0.02);
        let corr = correlations.get(t).unwrap_or(&0.5);

        let mut price = *base;
        let mut returns = Vec::with_capacity(total_periods);

        for p in 0..total_periods {
            // Коррелированный шок
            let idio_shock = rand_normal();
            let common_shock = btc_shocks[p];
            let shock = corr * common_shock + (1.0 - corr * corr).sqrt() * idio_shock;

            // Добавляем небольшой тренд и mean reversion
            let trend = 0.0001 * (t as f64 + 1.0);
            let mean_rev = -0.001 * (price / base - 1.0);

            let return_val = trend + mean_rev + vol * shock;
            price *= 1.0 + return_val;

            prices[[p, t]] = price;
            returns.push(return_val);

            // Заполняем признаки
            features[[0, t, p, 0]] = return_val;
            features[[0, t, p, 1]] = (price / base) - 1.0; // Normalized price
            features[[0, t, p, 2]] = rand_normal() * 0.1; // Volume change (synthetic)
            features[[0, t, p, 3]] = 0.5 + rand_normal() * 0.1; // RSI-like
            features[[0, t, p, 4]] = vol; // Volatility
            features[[0, t, p, 5]] = price / base; // Relative price
        }
    }

    (prices, features)
}

fn rand_normal() -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
