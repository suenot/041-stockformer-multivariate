//! # Stockformer
//!
//! Библиотека для мультивариантного прогнозирования криптовалют с использованием
//! Stockformer архитектуры и данных с биржи Bybit.
//!
//! ## Особенности
//!
//! - Кросс-активное внимание для моделирования связей между активами
//! - ProbSparse attention для эффективной обработки длинных последовательностей
//! - Поддержка нескольких типов выхода: регрессия, направление, аллокация портфеля
//!
//! ## Модули
//!
//! - `api` - Клиент для работы с Bybit API
//! - `data` - Загрузка и предобработка мультивариантных данных
//! - `model` - Реализация архитектуры Stockformer
//! - `strategy` - Торговая стратегия и бэктестинг
//!
//! ## Пример использования
//!
//! ```no_run
//! use stockformer::{BybitClient, MultiAssetLoader, StockformerModel, StockformerConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Получаем данные для нескольких активов
//!     let client = BybitClient::new();
//!     let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
//!
//!     let mut all_klines = Vec::new();
//!     for symbol in &symbols {
//!         let klines = client.get_klines(symbol, "1h", 1000).await.unwrap();
//!         all_klines.push(klines);
//!     }
//!
//!     // Подготавливаем мультивариантные данные
//!     let loader = MultiAssetLoader::new(symbols.clone());
//!     let dataset = loader.prepare_dataset(&all_klines, 168, 24).unwrap();
//!
//!     // Создаём модель Stockformer
//!     let config = StockformerConfig {
//!         n_assets: symbols.len(),
//!         d_model: 128,
//!         n_heads: 8,
//!         n_encoder_layers: 3,
//!         attention_type: AttentionType::ProbSparse,
//!         output_type: OutputType::Allocation,
//!         ..Default::default()
//!     };
//!     let model = StockformerModel::new(config);
//!
//!     // Делаем прогноз с весами внимания
//!     let (predictions, attention_weights) = model.predict_with_attention(&dataset);
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

// Re-exports для удобства
pub use api::{BybitClient, BybitError, Kline, OrderBook, Ticker};
pub use data::{MultiAssetLoader, MultiAssetDataset, Features, CorrelationMatrix};
pub use model::{
    CrossTickerAttention, ProbSparseAttention, TokenEmbedding,
    StockformerConfig, StockformerModel, AttentionType, OutputType,
};
pub use strategy::{BacktestResult, PortfolioSignal, SignalGenerator, PortfolioStrategy};

/// Версия библиотеки
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Настройки по умолчанию
pub mod defaults {
    /// Размер скрытого слоя
    pub const D_MODEL: usize = 128;

    /// Количество голов внимания
    pub const N_HEADS: usize = 8;

    /// Количество активов по умолчанию
    pub const N_ASSETS: usize = 5;

    /// Количество encoder layers
    pub const N_ENCODER_LAYERS: usize = 3;

    /// Dropout rate
    pub const DROPOUT: f64 = 0.1;

    /// Длина encoder context
    pub const ENCODER_LENGTH: usize = 168; // 7 дней часовых данных

    /// Длина prediction horizon
    pub const PREDICTION_LENGTH: usize = 24; // 24 часа

    /// Скорость обучения
    pub const LEARNING_RATE: f64 = 0.001;

    /// Размер батча
    pub const BATCH_SIZE: usize = 32;

    /// Количество эпох
    pub const EPOCHS: usize = 100;

    /// Фактор разреженности для ProbSparse attention
    pub const SPARSITY_FACTOR: usize = 5;
}
