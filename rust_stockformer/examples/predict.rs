//! Пример предсказания с использованием Stockformer
//!
//! Демонстрирует загрузку реальных данных и генерацию торговых сигналов

use ndarray::Array4;
use rust_stockformer::api::{BybitClient, Interval};
use rust_stockformer::data::{FeatureEngineering, MultiAssetLoader};
use rust_stockformer::model::{StockformerConfig, StockformerModel, OutputType};
use rust_stockformer::strategy::{SignalGenerator, SignalType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Stockformer: Генерация торговых сигналов ===\n");

    // Конфигурация
    let symbols = vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "SOLUSDT".to_string(),
        "BNBUSDT".to_string(),
        "XRPUSDT".to_string(),
    ];

    let seq_len = 96; // 96 часов = 4 дня

    // Создаем клиент и загрузчик данных
    let client = BybitClient::new();
    let loader = MultiAssetLoader::new(symbols.clone(), seq_len);

    println!("Загрузка данных с Bybit...");
    println!("Тикеры: {:?}", symbols);
    println!("Длина последовательности: {} часов\n", seq_len);

    // Загружаем реальные данные
    let raw_data = match loader.load_from_bybit(&client, Interval::Hour1).await {
        Ok(data) => {
            println!("Данные загружены: {:?}", data.dim());
            data
        }
        Err(e) => {
            println!("Ошибка загрузки реальных данных: {}", e);
            println!("Используем синтетические данные для демонстрации...\n");
            generate_demo_data(symbols.len(), seq_len)
        }
    };

    // Подготавливаем признаки
    println!("\nПодготовка признаков...");
    let fe = FeatureEngineering::new();
    let features = prepare_features(&raw_data, &fe);
    println!("Размер признаков: {:?}", features.dim());

    // Создаем модель
    let config = StockformerConfig {
        num_tickers: symbols.len(),
        seq_len,
        input_features: 6,
        d_model: 32,
        num_heads: 4,
        d_ff: 128,
        num_encoder_layers: 2,
        output_type: OutputType::Regression,
        use_cross_ticker_attention: true,
        ..Default::default()
    };

    println!("\nСоздание модели Stockformer...");
    let model = StockformerModel::new(config.clone())?;

    // Делаем предсказание
    println!("Генерация предсказаний...\n");
    let result = model.forward(&features);

    // Выводим предсказания
    println!("=== Предсказания доходности ===\n");

    for (i, symbol) in symbols.iter().enumerate() {
        let pred = result.predictions[[0, i]];
        let direction = if pred > 0.005 {
            "↑ РОСТ"
        } else if pred < -0.005 {
            "↓ ПАДЕНИЕ"
        } else {
            "→ БОКОВИК"
        };

        println!(
            "{}: Ожидаемая доходность = {:.2}% ({})",
            symbol,
            pred * 100.0,
            direction
        );
    }

    // Анализ весов внимания
    println!("\n=== Анализ кросс-тикерного внимания ===\n");

    if let Some(ref _weights) = result.attention_weights.cross_ticker_weights {
        let top_relationships = result.attention_weights.top_k_cross_ticker(10);

        println!("Топ-10 влияний между активами:");
        for (from, to, weight) in top_relationships {
            let from_symbol = symbols.get(from).map(|s| s.as_str()).unwrap_or("?");
            let to_symbol = symbols.get(to).map(|s| s.as_str()).unwrap_or("?");

            let bar_len = (weight * 20.0).round() as usize;
            let bar: String = "█".repeat(bar_len);

            println!(
                "  {} → {}: {:.4} {}",
                from_symbol, to_symbol, weight, bar
            );
        }
    }

    // Генерация торговых сигналов
    println!("\n=== Торговые сигналы ===\n");

    let signal_gen = SignalGenerator::new()
        .with_thresholds(0.003, -0.003)  // 0.3% пороги
        .with_min_confidence(0.3);

    let portfolio_signal = signal_gen.generate(
        &result,
        OutputType::Regression,
        &symbols,
    );

    println!("Рекомендации:");
    for signal in &portfolio_signal.signals {
        let symbol = &symbols[signal.ticker_idx];

        let action = match signal.signal_type {
            SignalType::Long => "🟢 ПОКУПКА",
            SignalType::Short => "🔴 ПРОДАЖА",
            SignalType::Hold => "⚪ ДЕРЖАТЬ",
            SignalType::Close => "🟡 ЗАКРЫТЬ",
        };

        let confidence_bar = "●".repeat((signal.confidence * 5.0).round() as usize);

        println!(
            "  {}: {} | Сила: {:.0}% | Уверенность: {} ({:.0}%)",
            symbol,
            action,
            signal.strength * 100.0,
            confidence_bar,
            signal.confidence * 100.0
        );
    }

    // Портфельное распределение
    println!("\n=== Рекомендуемое распределение портфеля ===\n");

    let total_weight: f64 = portfolio_signal.weights.iter().map(|w| w.abs()).sum();
    let cash_weight = (1.0 - total_weight).max(0.0);

    for (i, &weight) in portfolio_signal.weights.iter().enumerate() {
        if weight.abs() > 0.01 {
            let bar_len = (weight.abs() * 40.0).round() as usize;
            let bar: String = if weight > 0.0 {
                "█".repeat(bar_len)
            } else {
                "░".repeat(bar_len)
            };

            println!(
                "  {}: {:>6.1}% {}",
                symbols[i],
                weight * 100.0,
                bar
            );
        }
    }

    if cash_weight > 0.01 {
        println!("  CASH:  {:>6.1}%", cash_weight * 100.0);
    }

    // Квантильная регрессия для оценки риска
    println!("\n=== Оценка риска (квантильная регрессия) ===\n");

    let quantile_config = StockformerConfig {
        output_type: OutputType::Quantile,
        quantiles: vec![0.1, 0.5, 0.9],
        ..config.clone()
    };

    let quantile_model = StockformerModel::new(quantile_config)?;
    let quantile_result = quantile_model.forward(&features);

    println!("Интервалы предсказаний (10% - 50% - 90%):");
    for (i, symbol) in symbols.iter().enumerate() {
        let q10 = quantile_result.predictions[[0, i * 3]] * 100.0;
        let q50 = quantile_result.predictions[[0, i * 3 + 1]] * 100.0;
        let q90 = quantile_result.predictions[[0, i * 3 + 2]] * 100.0;

        let range = q90 - q10;
        let risk_level = if range > 5.0 {
            "⚠️  ВЫСОКИЙ"
        } else if range > 2.0 {
            "⚡ СРЕДНИЙ"
        } else {
            "✅ НИЗКИЙ"
        };

        println!(
            "  {}: [{:>+6.2}% | {:>+6.2}% | {:>+6.2}%] Риск: {}",
            symbol, q10, q50, q90, risk_level
        );
    }

    if let Some(ref confidence) = quantile_result.confidence {
        println!("\nОбщая уверенность модели:");
        let avg_confidence: f64 = confidence.row(0).sum() / symbols.len() as f64;
        println!("  Средняя: {:.1}%", avg_confidence * 100.0);
    }

    println!("\n=== Готово ===");

    Ok(())
}

/// Подготавливает признаки из сырых данных
fn prepare_features(raw_data: &Array4<f64>, fe: &FeatureEngineering) -> Array4<f64> {
    let (batch, num_tickers, seq_len, raw_features) = raw_data.dim();
    let num_features = 6; // returns, normalized price, volume_change, rsi, volatility, close

    let mut features = Array4::zeros((batch, num_tickers, seq_len, num_features));

    for b in 0..batch {
        for t in 0..num_tickers {
            // Извлекаем цены закрытия
            let closes: Vec<f64> = (0..seq_len)
                .map(|s| raw_data[[b, t, s, 3]]) // close - индекс 3
                .collect();

            // Извлекаем объемы
            let volumes: Vec<f64> = (0..seq_len)
                .map(|s| raw_data[[b, t, s, 4]]) // volume - индекс 4
                .collect();

            // Вычисляем признаки
            let returns = fe.log_returns(&closes);
            let rsi = fe.rsi(&closes, 14);
            let volatility = fe.realized_volatility(&returns, 20);
            let volume_changes = fe.volume_change(&volumes);

            // Нормализуем цены
            let first_close = closes[0];
            let normalized_prices: Vec<f64> = closes
                .iter()
                .map(|&p| (p - first_close) / first_close)
                .collect();

            // Заполняем признаки
            for s in 0..seq_len {
                features[[b, t, s, 0]] = *returns.get(s).unwrap_or(&0.0);
                features[[b, t, s, 1]] = normalized_prices[s];
                features[[b, t, s, 2]] = *volume_changes.get(s).unwrap_or(&0.0);
                features[[b, t, s, 3]] = *rsi.get(s).unwrap_or(&50.0) / 100.0; // Normalize to [0, 1]
                features[[b, t, s, 4]] = *volatility.get(s).unwrap_or(&0.0);
                features[[b, t, s, 5]] = closes[s] / first_close; // Relative close
            }
        }
    }

    features
}

/// Генерирует демонстрационные данные
fn generate_demo_data(num_tickers: usize, seq_len: usize) -> Array4<f64> {
    use std::f64::consts::PI;

    let mut data = Array4::zeros((1, num_tickers, seq_len, 5)); // OHLCV

    for t in 0..num_tickers {
        let base_price = 100.0 + t as f64 * 1000.0;
        let trend = (rand::random::<f64>() - 0.5) * 0.001;

        for s in 0..seq_len {
            let noise = rand_normal() * 0.02;
            let seasonal = (2.0 * PI * s as f64 / 24.0).sin() * 0.01;

            let close = base_price * (1.0 + trend * s as f64 + noise + seasonal);
            let open = close * (1.0 + rand_normal() * 0.005);
            let high = close.max(open) * (1.0 + rand::random::<f64>() * 0.01);
            let low = close.min(open) * (1.0 - rand::random::<f64>() * 0.01);
            let volume = 1000000.0 * (1.0 + rand_normal() * 0.3).max(0.1);

            data[[0, t, s, 0]] = open;
            data[[0, t, s, 1]] = high;
            data[[0, t, s, 2]] = low;
            data[[0, t, s, 3]] = close;
            data[[0, t, s, 4]] = volume;
        }
    }

    data
}

fn rand_normal() -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
