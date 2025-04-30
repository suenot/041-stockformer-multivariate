//! Token Embedding модуль
//!
//! Реализует 1D-CNN embedding для преобразования временных рядов
//! в токены фиксированной размерности

use ndarray::{Array1, Array2, Array3, Axis, s};
use crate::model::config::StockformerConfig;

/// Token Embedding с использованием 1D свертки
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    /// Веса свертки: [out_channels, in_channels, kernel_size]
    weights: Array3<f64>,
    /// Смещения: [out_channels]
    bias: Array1<f64>,
    /// Размер ядра свертки
    kernel_size: usize,
    /// Размер паддинга
    padding: usize,
}

impl TokenEmbedding {
    /// Создает новый TokenEmbedding
    pub fn new(config: &StockformerConfig) -> Self {
        let in_channels = config.input_features;
        let out_channels = config.d_model;
        let kernel_size = config.kernel_size;
        let padding = kernel_size / 2;

        // Инициализация весов Xavier/Glorot
        let scale = (2.0 / (in_channels + out_channels) as f64).sqrt();
        let weights = Array3::from_shape_fn(
            (out_channels, in_channels, kernel_size),
            |_| rand_normal() * scale
        );

        let bias = Array1::zeros(out_channels);

        Self {
            weights,
            bias,
            kernel_size,
            padding,
        }
    }

    /// Создает TokenEmbedding с заданными весами
    pub fn with_weights(weights: Array3<f64>, bias: Array1<f64>, kernel_size: usize) -> Self {
        let padding = kernel_size / 2;
        Self {
            weights,
            bias,
            kernel_size,
            padding,
        }
    }

    /// Применяет 1D свертку к входным данным
    ///
    /// # Arguments
    /// * `input` - [batch, seq_len, features]
    ///
    /// # Returns
    /// * `output` - [batch, seq_len, d_model]
    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, _) = input.dim();
        let out_channels = self.weights.dim().0;

        // Паддинг входных данных
        let padded = self.pad_input(input);

        let mut output = Array3::zeros((batch_size, seq_len, out_channels));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for c_out in 0..out_channels {
                    let mut sum = self.bias[c_out];

                    for k in 0..self.kernel_size {
                        let t_idx = t + k;
                        for c_in in 0..self.weights.dim().1 {
                            sum += padded[[b, t_idx, c_in]] * self.weights[[c_out, c_in, k]];
                        }
                    }

                    // ReLU активация
                    output[[b, t, c_out]] = sum.max(0.0);
                }
            }
        }

        output
    }

    /// Паддинг входных данных
    fn pad_input(&self, input: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, features) = input.dim();
        let padded_len = seq_len + 2 * self.padding;

        let mut padded = Array3::zeros((batch_size, padded_len, features));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for f in 0..features {
                    padded[[b, t + self.padding, f]] = input[[b, t, f]];
                }
            }
        }

        padded
    }

    /// Возвращает количество параметров
    pub fn num_parameters(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

/// Позиционное кодирование
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Матрица позиционного кодирования [max_len, d_model]
    encoding: Array2<f64>,
    /// Максимальная длина последовательности
    max_len: usize,
}

impl PositionalEncoding {
    /// Создает синусоидальное позиционное кодирование
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut encoding = Array2::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f64 / (10000.0_f64).powf((2 * (i / 2)) as f64 / d_model as f64);

                if i % 2 == 0 {
                    encoding[[pos, i]] = angle.sin();
                } else {
                    encoding[[pos, i]] = angle.cos();
                }
            }
        }

        Self { encoding, max_len }
    }

    /// Добавляет позиционное кодирование к входным данным
    ///
    /// # Arguments
    /// * `input` - [batch, seq_len, d_model]
    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = input.dim();
        assert!(seq_len <= self.max_len, "Sequence length exceeds max_len");

        let mut output = input.clone();

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..d_model {
                    output[[b, t, d]] += self.encoding[[t, d]];
                }
            }
        }

        output
    }

    /// Получает срез позиционного кодирования
    pub fn get_encoding(&self, seq_len: usize) -> Array2<f64> {
        self.encoding.slice(s![..seq_len, ..]).to_owned()
    }
}

/// Тикер-специфичное кодирование
#[derive(Debug, Clone)]
pub struct TickerEncoding {
    /// Эмбеддинги тикеров [num_tickers, d_model]
    embeddings: Array2<f64>,
}

impl TickerEncoding {
    /// Создает кодирование для тикеров
    pub fn new(num_tickers: usize, d_model: usize) -> Self {
        let scale = (1.0 / d_model as f64).sqrt();
        let embeddings = Array2::from_shape_fn(
            (num_tickers, d_model),
            |_| rand_normal() * scale
        );

        Self { embeddings }
    }

    /// Добавляет тикер-специфичное кодирование
    ///
    /// # Arguments
    /// * `input` - [batch, num_tickers, seq_len, d_model]
    pub fn forward(&self, input: &ndarray::Array4<f64>) -> ndarray::Array4<f64> {
        let (batch_size, num_tickers, seq_len, d_model) = input.dim();
        let mut output = input.clone();

        for b in 0..batch_size {
            for ticker in 0..num_tickers {
                for t in 0..seq_len {
                    for d in 0..d_model {
                        output[[b, ticker, t, d]] += self.embeddings[[ticker, d]];
                    }
                }
            }
        }

        output
    }

    /// Получает эмбеддинг конкретного тикера
    pub fn get_embedding(&self, ticker_idx: usize) -> Array1<f64> {
        self.embeddings.row(ticker_idx).to_owned()
    }
}

/// Генерирует случайное число из нормального распределения (Box-Muller)
fn rand_normal() -> f64 {
    use std::f64::consts::PI;

    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();

    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_embedding() {
        let config = StockformerConfig {
            input_features: 6,
            d_model: 32,
            kernel_size: 3,
            ..Default::default()
        };

        let embedding = TokenEmbedding::new(&config);

        // [batch=2, seq_len=10, features=6]
        let input = Array3::from_shape_fn((2, 10, 6), |_| rand_normal());
        let output = embedding.forward(&input);

        assert_eq!(output.dim(), (2, 10, 32));
    }

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::new(64, 100);

        let input = Array3::zeros((2, 50, 64));
        let output = pe.forward(&input);

        assert_eq!(output.dim(), (2, 50, 64));

        // Проверяем, что позиционное кодирование добавлено
        assert!(output[[0, 0, 0]].abs() > 0.0);
    }

    #[test]
    fn test_ticker_encoding() {
        let te = TickerEncoding::new(5, 64);

        let input = ndarray::Array4::zeros((2, 5, 10, 64));
        let output = te.forward(&input);

        assert_eq!(output.dim(), (2, 5, 10, 64));

        // Эмбеддинги разных тикеров должны отличаться
        let emb0 = te.get_embedding(0);
        let emb1 = te.get_embedding(1);
        assert!((emb0.clone() - emb1.clone()).mapv(|x| x.abs()).sum() > 0.0);
    }

    #[test]
    fn test_positional_encoding_values() {
        let pe = PositionalEncoding::new(4, 10);
        let encoding = pe.get_encoding(10);

        // Проверяем размерность
        assert_eq!(encoding.dim(), (10, 4));

        // Проверяем, что значения в разумном диапазоне [-1, 1]
        for val in encoding.iter() {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_embedding_parameters() {
        let config = StockformerConfig {
            input_features: 6,
            d_model: 32,
            kernel_size: 3,
            ..Default::default()
        };

        let embedding = TokenEmbedding::new(&config);
        let num_params = embedding.num_parameters();

        // weights: 32 * 6 * 3 + bias: 32 = 576 + 32 = 608
        assert_eq!(num_params, 32 * 6 * 3 + 32);
    }
}
