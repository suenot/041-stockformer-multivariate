//! Пример обучения модели Stockformer
//!
//! Демонстрирует процесс создания и обучения модели на синтетических данных

use ndarray::{Array4, Axis};
use rust_stockformer::model::{
    StockformerConfig, StockformerModel, OutputType, AttentionType,
};

fn main() {
    println!("=== Stockformer: Демонстрация обучения модели ===\n");

    // Конфигурация модели
    let config = StockformerConfig {
        num_tickers: 5,
        seq_len: 48,
        input_features: 6,  // OHLCV + returns
        d_model: 32,
        num_heads: 4,
        d_ff: 128,
        num_encoder_layers: 2,
        dropout: 0.1,
        attention_type: AttentionType::ProbSparse,
        output_type: OutputType::Regression,
        prediction_horizon: 1,
        use_cross_ticker_attention: true,
        use_positional_encoding: true,
        ..Default::default()
    };

    println!("Конфигурация модели:");
    println!("  - Количество тикеров: {}", config.num_tickers);
    println!("  - Длина последовательности: {}", config.seq_len);
    println!("  - Размерность модели: {}", config.d_model);
    println!("  - Количество голов внимания: {}", config.num_heads);
    println!("  - Количество слоев энкодера: {}", config.num_encoder_layers);
    println!("  - Тип внимания: {:?}", config.attention_type);
    println!("  - Тип выхода: {:?}", config.output_type);
    println!();

    // Создаем модель
    println!("Создание модели...");
    let model = match StockformerModel::new(config.clone()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Ошибка создания модели: {}", e);
            return;
        }
    };

    println!("Модель создана. Количество параметров: ~{}", model.num_parameters());
    println!();

    // Генерируем синтетические данные для демонстрации
    println!("Генерация синтетических данных...");
    let batch_size = 4;
    let (train_data, train_targets) = generate_synthetic_data(
        batch_size,
        config.num_tickers,
        config.seq_len,
        config.input_features,
    );

    println!("  - Размер обучающих данных: {:?}", train_data.dim());
    println!("  - Размер целевых значений: {:?}", train_targets.dim());
    println!();

    // Демонстрация forward pass
    println!("Выполнение forward pass...");
    let result = model.forward(&train_data);

    println!("Результаты:");
    println!("  - Размер предсказаний: {:?}", result.predictions.dim());
    println!("  - Temporal attention: {}",
        result.attention_weights.temporal_weights.is_some());
    println!("  - Cross-ticker attention: {}",
        result.attention_weights.cross_ticker_weights.is_some());
    println!();

    // Анализ весов внимания
    if let Some(ref cross_weights) = result.attention_weights.cross_ticker_weights {
        println!("Анализ кросс-тикерного внимания:");
        let top_k = result.attention_weights.top_k_cross_ticker(5);

        let ticker_names = vec!["BTC", "ETH", "SOL", "BNB", "XRP"];
        for (i, j, weight) in top_k {
            println!(
                "  {} -> {}: {:.4}",
                ticker_names.get(i).unwrap_or(&"?"),
                ticker_names.get(j).unwrap_or(&"?"),
                weight
            );
        }
        println!();
    }

    // Демонстрация обучающего цикла (упрощенный)
    println!("=== Демонстрация обучающего цикла ===\n");

    let learning_rate = 0.001;
    let num_epochs = 3;

    for epoch in 0..num_epochs {
        let result = model.forward(&train_data);

        // Вычисляем MSE loss
        let mut total_loss = 0.0;
        for b in 0..batch_size {
            for t in 0..config.num_tickers {
                let pred = result.predictions[[b, t]];
                let target = train_targets[[b, t]];
                total_loss += (pred - target).powi(2);
            }
        }
        let mse = total_loss / (batch_size * config.num_tickers) as f64;

        println!(
            "Эпоха {}/{}: MSE Loss = {:.6}",
            epoch + 1,
            num_epochs,
            mse
        );

        // В реальном коде здесь был бы backward pass и обновление весов
        // Это упрощенная демонстрация
    }

    println!();

    // Тестирование разных конфигураций
    println!("=== Тестирование разных конфигураций ===\n");

    // Направленное предсказание
    let direction_config = StockformerConfig {
        output_type: OutputType::Direction,
        ..config.clone()
    };

    let direction_model = StockformerModel::new(direction_config).unwrap();
    let direction_result = direction_model.forward(&train_data);

    println!("Направленное предсказание (Direction):");
    println!("  - Размер выхода: {:?}", direction_result.predictions.dim());

    // Показываем вероятности для первого тикера
    let ticker_names = vec!["BTC", "ETH", "SOL", "BNB", "XRP"];
    for t in 0..config.num_tickers.min(3) {
        let p_down = direction_result.predictions[[0, t * 3]];
        let p_hold = direction_result.predictions[[0, t * 3 + 1]];
        let p_up = direction_result.predictions[[0, t * 3 + 2]];

        println!(
            "  {}: Down={:.2}%, Hold={:.2}%, Up={:.2}%",
            ticker_names[t],
            p_down * 100.0,
            p_hold * 100.0,
            p_up * 100.0
        );
    }
    println!();

    // Портфельное распределение
    let portfolio_config = StockformerConfig {
        output_type: OutputType::Portfolio,
        ..config.clone()
    };

    let portfolio_model = StockformerModel::new(portfolio_config).unwrap();
    let portfolio_result = portfolio_model.forward(&train_data);

    println!("Портфельное распределение (Portfolio):");
    let weights_sum: f64 = (0..config.num_tickers)
        .map(|t| portfolio_result.predictions[[0, t]])
        .sum();

    for t in 0..config.num_tickers {
        let weight = portfolio_result.predictions[[0, t]];
        println!("  {}: {:.2}%", ticker_names[t], weight * 100.0);
    }
    println!("  Сумма весов: {:.4}", weights_sum);
    println!();

    // Квантильная регрессия
    let quantile_config = StockformerConfig {
        output_type: OutputType::Quantile,
        quantiles: vec![0.1, 0.5, 0.9],
        ..config.clone()
    };

    let quantile_model = StockformerModel::new(quantile_config.clone()).unwrap();
    let quantile_result = quantile_model.forward(&train_data);

    println!("Квантильная регрессия (Quantile):");
    println!("  - Квантили: {:?}", quantile_config.quantiles);

    for t in 0..config.num_tickers.min(3) {
        let q10 = quantile_result.predictions[[0, t * 3]];
        let q50 = quantile_result.predictions[[0, t * 3 + 1]];
        let q90 = quantile_result.predictions[[0, t * 3 + 2]];

        println!(
            "  {}: Q10={:.4}, Q50={:.4}, Q90={:.4}",
            ticker_names[t], q10, q50, q90
        );
    }

    if let Some(ref confidence) = quantile_result.confidence {
        println!("\n  Уверенность предсказания:");
        for t in 0..config.num_tickers {
            println!("    {}: {:.2}%", ticker_names[t], confidence[[0, t]] * 100.0);
        }
    }

    println!("\n=== Готово ===");
}

/// Генерирует синтетические данные для демонстрации
fn generate_synthetic_data(
    batch_size: usize,
    num_tickers: usize,
    seq_len: usize,
    num_features: usize,
) -> (Array4<f64>, ndarray::Array2<f64>) {
    use std::f64::consts::PI;

    // Генерируем входные данные с некоторой структурой
    let data = Array4::from_shape_fn(
        (batch_size, num_tickers, seq_len, num_features),
        |(b, t, s, f)| {
            // Базовая синусоида с разными частотами для разных тикеров
            let freq = 0.1 + t as f64 * 0.05;
            let phase = b as f64 * 0.5;
            let base = (2.0 * PI * freq * s as f64 + phase).sin();

            // Добавляем корреляцию между тикерами
            let correlation = if t > 0 { 0.3 * (t - 1) as f64 } else { 0.0 };

            // Разные признаки имеют разный масштаб
            let scale = match f {
                0 => 0.01,  // returns
                1 => 1.0,   // normalized price
                2 => 0.1,   // volume change
                3 => 50.0,  // RSI-like (0-100)
                4 => 0.02,  // volatility
                _ => 0.1,
            };

            let noise = rand_normal() * 0.1;
            (base + correlation + noise) * scale
        },
    );

    // Генерируем целевые значения (коррелирующие с входными данными)
    let targets = ndarray::Array2::from_shape_fn(
        (batch_size, num_tickers),
        |(b, t)| {
            // Целевое значение зависит от последних значений входных данных
            let last_value = data[[b, t, seq_len - 1, 0]];
            last_value + rand_normal() * 0.01
        },
    );

    (data, targets)
}

/// Генерирует случайное число из нормального распределения
fn rand_normal() -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
