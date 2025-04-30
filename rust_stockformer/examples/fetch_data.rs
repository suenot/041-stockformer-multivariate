//! Пример загрузки данных с Bybit
//!
//! Демонстрирует использование API клиента для получения криптовалютных данных

use rust_stockformer::api::{BybitClient, Interval};
use rust_stockformer::data::{FeatureEngineering, MultiAssetLoader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Stockformer: Загрузка данных с Bybit ===\n");

    // Создаем клиент API
    let client = BybitClient::new();

    // Список криптовалют для анализа
    let symbols = vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "SOLUSDT".to_string(),
    ];

    println!("Загружаем данные для: {:?}\n", symbols);

    // Загружаем текущие тикеры
    println!("--- Текущие цены ---");
    for symbol in &symbols {
        match client.get_ticker(symbol).await {
            Ok(ticker) => {
                println!(
                    "{}: Цена = ${:.2}, Изменение 24ч = {:.2}%, Объем = ${:.0}",
                    ticker.symbol,
                    ticker.last_price,
                    ticker.price_24h_pcnt * 100.0,
                    ticker.volume_24h
                );
            }
            Err(e) => {
                println!("{}: Ошибка загрузки - {}", symbol, e);
            }
        }
    }

    println!("\n--- Исторические данные (последние 100 свечей, 1 час) ---");

    // Загружаем исторические данные
    for symbol in &symbols {
        match client.get_klines(symbol, Interval::Hour1, Some(100)).await {
            Ok(klines) => {
                if let (Some(first), Some(last)) = (klines.first(), klines.last()) {
                    let price_change =
                        (last.close - first.close) / first.close * 100.0;
                    let total_volume: f64 = klines.iter().map(|k| k.volume).sum();

                    println!(
                        "{}: {} свечей, Начало = ${:.2}, Конец = ${:.2}, Изменение = {:.2}%, Объем = {:.0}",
                        symbol,
                        klines.len(),
                        first.close,
                        last.close,
                        price_change,
                        total_volume
                    );
                }
            }
            Err(e) => {
                println!("{}: Ошибка загрузки - {}", symbol, e);
            }
        }
    }

    println!("\n--- Стакан заявок (глубина 5) ---");

    // Загружаем стакан заявок
    for symbol in &symbols {
        match client.get_orderbook(symbol, Some(5)).await {
            Ok(orderbook) => {
                if let (Some(best_bid), Some(best_ask)) = (
                    orderbook.bids.first(),
                    orderbook.asks.first(),
                ) {
                    let spread = (best_ask.0 - best_bid.0) / best_bid.0 * 100.0;
                    println!(
                        "{}: Лучший бид = ${:.2}, Лучший аск = ${:.2}, Спред = {:.4}%",
                        symbol, best_bid.0, best_ask.0, spread
                    );
                }
            }
            Err(e) => {
                println!("{}: Ошибка загрузки - {}", symbol, e);
            }
        }
    }

    println!("\n--- Подготовка данных для модели ---");

    // Создаем загрузчик данных
    let loader = MultiAssetLoader::new(symbols.clone(), 96);

    // Загружаем данные через API
    match loader.load_from_bybit(&client, Interval::Hour1).await {
        Ok(raw_data) => {
            println!("Загружено данных: {:?}", raw_data.dim());

            // Подготавливаем признаки
            let fe = FeatureEngineering::new();

            // Вычисляем корреляционную матрицу
            let (batch, num_assets, seq_len, _) = raw_data.dim();
            println!("\n--- Корреляция между активами ---");

            // Извлекаем цены закрытия для расчета корреляции
            let mut close_prices = vec![vec![]; num_assets];
            for t in 0..num_assets {
                for s in 0..seq_len {
                    close_prices[t].push(raw_data[[0, t, s, 3]]); // close - индекс 3
                }
            }

            // Вычисляем попарные корреляции
            for i in 0..num_assets {
                for j in (i + 1)..num_assets {
                    let corr = pearson_correlation(&close_prices[i], &close_prices[j]);
                    println!("{} <-> {}: {:.4}", symbols[i], symbols[j], corr);
                }
            }

            // Демонстрируем технические индикаторы
            println!("\n--- Технические индикаторы (последние значения) ---");

            for (t, symbol) in symbols.iter().enumerate() {
                let prices: Vec<f64> = (0..seq_len)
                    .map(|s| raw_data[[0, t, s, 3]])
                    .collect();

                let returns = fe.log_returns(&prices);
                let rsi = fe.rsi(&prices, 14);
                let volatility = fe.realized_volatility(&returns, 20);

                if let (Some(&last_return), Some(&last_rsi), Some(&last_vol)) =
                    (returns.last(), rsi.last(), volatility.last())
                {
                    println!(
                        "{}: Возврат = {:.4}%, RSI = {:.2}, Волатильность = {:.4}%",
                        symbol,
                        last_return * 100.0,
                        last_rsi,
                        last_vol * 100.0
                    );
                }
            }
        }
        Err(e) => {
            println!("Ошибка загрузки данных: {}", e);
        }
    }

    println!("\n=== Готово ===");

    Ok(())
}

/// Вычисляет коэффициент корреляции Пирсона
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n as usize {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}
