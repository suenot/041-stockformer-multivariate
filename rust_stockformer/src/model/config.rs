//! Конфигурация модели Stockformer
//!
//! Содержит настройки архитектуры и гиперпараметры

use serde::{Deserialize, Serialize};

/// Тип механизма внимания
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionType {
    /// Полное внимание O(L²)
    Full,
    /// ProbSparse внимание O(L·log(L))
    ProbSparse,
    /// Локальное внимание с фиксированным окном
    Local,
}

impl Default for AttentionType {
    fn default() -> Self {
        Self::ProbSparse
    }
}

/// Тип выходного слоя
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// Регрессия - предсказание непрерывных значений
    Regression,
    /// Классификация направления (вверх/вниз/без изменений)
    Direction,
    /// Распределение портфеля (веса активов)
    Portfolio,
    /// Квантильная регрессия для интервалов неопределенности
    Quantile,
}

impl Default for OutputType {
    fn default() -> Self {
        Self::Regression
    }
}

/// Конфигурация модели Stockformer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockformerConfig {
    /// Количество активов (тикеров)
    pub num_tickers: usize,

    /// Длина входной последовательности
    pub seq_len: usize,

    /// Количество входных признаков на актив
    pub input_features: usize,

    /// Размерность модели (d_model)
    pub d_model: usize,

    /// Количество голов внимания
    pub num_heads: usize,

    /// Размерность feed-forward слоя
    pub d_ff: usize,

    /// Количество слоев энкодера
    pub num_encoder_layers: usize,

    /// Dropout вероятность
    pub dropout: f64,

    /// Тип механизма внимания
    pub attention_type: AttentionType,

    /// Тип выходного слоя
    pub output_type: OutputType,

    /// Горизонт предсказания
    pub prediction_horizon: usize,

    /// Размер ядра для 1D-CNN embedding
    pub kernel_size: usize,

    /// Фактор разреженности для ProbSparse (c в формуле)
    pub sparsity_factor: f64,

    /// Квантили для квантильной регрессии
    pub quantiles: Vec<f64>,

    /// Использовать кросс-тикерное внимание
    pub use_cross_ticker_attention: bool,

    /// Использовать позиционное кодирование
    pub use_positional_encoding: bool,
}

impl Default for StockformerConfig {
    fn default() -> Self {
        Self {
            num_tickers: 5,
            seq_len: 96,
            input_features: 6,
            d_model: 64,
            num_heads: 4,
            d_ff: 256,
            num_encoder_layers: 2,
            dropout: 0.1,
            attention_type: AttentionType::ProbSparse,
            output_type: OutputType::Regression,
            prediction_horizon: 1,
            kernel_size: 3,
            sparsity_factor: 5.0,
            quantiles: vec![0.1, 0.5, 0.9],
            use_cross_ticker_attention: true,
            use_positional_encoding: true,
        }
    }
}

impl StockformerConfig {
    /// Создает конфигурацию для небольшой модели
    pub fn small() -> Self {
        Self {
            d_model: 32,
            num_heads: 2,
            d_ff: 128,
            num_encoder_layers: 1,
            ..Default::default()
        }
    }

    /// Создает конфигурацию для средней модели
    pub fn medium() -> Self {
        Self::default()
    }

    /// Создает конфигурацию для большой модели
    pub fn large() -> Self {
        Self {
            d_model: 128,
            num_heads: 8,
            d_ff: 512,
            num_encoder_layers: 4,
            ..Default::default()
        }
    }

    /// Создает конфигурацию для криптовалют
    pub fn crypto(num_tickers: usize) -> Self {
        Self {
            num_tickers,
            seq_len: 168, // 1 неделя часовых данных
            input_features: 8, // OHLCV + технические индикаторы
            use_cross_ticker_attention: true,
            ..Default::default()
        }
    }

    /// Проверяет валидность конфигурации
    pub fn validate(&self) -> Result<(), String> {
        if self.num_tickers == 0 {
            return Err("num_tickers must be > 0".to_string());
        }
        if self.seq_len == 0 {
            return Err("seq_len must be > 0".to_string());
        }
        if self.d_model == 0 {
            return Err("d_model must be > 0".to_string());
        }
        if self.d_model % self.num_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                self.d_model, self.num_heads
            ));
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err("dropout must be in [0, 1]".to_string());
        }
        if self.kernel_size == 0 || self.kernel_size % 2 == 0 {
            return Err("kernel_size must be odd and > 0".to_string());
        }
        if self.sparsity_factor <= 0.0 {
            return Err("sparsity_factor must be > 0".to_string());
        }

        // Проверка квантилей
        for q in &self.quantiles {
            if *q <= 0.0 || *q >= 1.0 {
                return Err("quantiles must be in (0, 1)".to_string());
            }
        }

        Ok(())
    }

    /// Возвращает размерность головы внимания
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Возвращает количество выходов модели
    pub fn output_dim(&self) -> usize {
        match self.output_type {
            OutputType::Regression => self.num_tickers,
            OutputType::Direction => self.num_tickers * 3, // up/down/neutral
            OutputType::Portfolio => self.num_tickers,
            OutputType::Quantile => self.num_tickers * self.quantiles.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = StockformerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_small_config() {
        let config = StockformerConfig::small();
        assert!(config.validate().is_ok());
        assert_eq!(config.d_model, 32);
    }

    #[test]
    fn test_large_config() {
        let config = StockformerConfig::large();
        assert!(config.validate().is_ok());
        assert_eq!(config.d_model, 128);
    }

    #[test]
    fn test_crypto_config() {
        let config = StockformerConfig::crypto(10);
        assert!(config.validate().is_ok());
        assert_eq!(config.num_tickers, 10);
        assert_eq!(config.seq_len, 168);
    }

    #[test]
    fn test_invalid_config() {
        let mut config = StockformerConfig::default();
        config.num_tickers = 0;
        assert!(config.validate().is_err());

        config = StockformerConfig::default();
        config.d_model = 65; // not divisible by num_heads=4
        assert!(config.validate().is_err());

        config = StockformerConfig::default();
        config.kernel_size = 4; // even
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_head_dim() {
        let config = StockformerConfig::default();
        assert_eq!(config.head_dim(), 64 / 4);
    }

    #[test]
    fn test_output_dim() {
        let mut config = StockformerConfig::default();
        config.num_tickers = 5;

        config.output_type = OutputType::Regression;
        assert_eq!(config.output_dim(), 5);

        config.output_type = OutputType::Direction;
        assert_eq!(config.output_dim(), 15);

        config.output_type = OutputType::Portfolio;
        assert_eq!(config.output_dim(), 5);

        config.output_type = OutputType::Quantile;
        config.quantiles = vec![0.1, 0.5, 0.9];
        assert_eq!(config.output_dim(), 15);
    }

    #[test]
    fn test_serialization() {
        let config = StockformerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: StockformerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.d_model, deserialized.d_model);
        assert_eq!(config.num_heads, deserialized.num_heads);
    }
}
