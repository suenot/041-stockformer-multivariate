//! Основная модель Stockformer
//!
//! Объединяет все компоненты в единую архитектуру

use ndarray::{Array1, Array2, Array3, Array4, Axis, s};

use crate::model::{
    AttentionWeights,
    CrossTickerAttention,
    ProbSparseAttention,
    TokenEmbedding,
    StockformerConfig,
    AttentionType,
    OutputType,
};
use crate::model::embedding::{PositionalEncoding, TickerEncoding};

/// Результат предсказания модели
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Предсказанные значения [batch, num_tickers] или [batch, num_tickers, num_quantiles]
    pub predictions: Array2<f64>,
    /// Веса внимания для интерпретации
    pub attention_weights: AttentionWeights,
    /// Уверенность предсказания (для квантильной регрессии)
    pub confidence: Option<Array2<f64>>,
}

/// Feed-Forward Network слой
#[derive(Debug, Clone)]
struct FeedForward {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}

impl FeedForward {
    fn new(d_model: usize, d_ff: usize) -> Self {
        let scale1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let scale2 = (2.0 / (d_ff + d_model) as f64).sqrt();

        Self {
            w1: Array2::from_shape_fn((d_model, d_ff), |_| rand_normal() * scale1),
            b1: Array1::zeros(d_ff),
            w2: Array2::from_shape_fn((d_ff, d_model), |_| rand_normal() * scale2),
            b2: Array1::zeros(d_model),
        }
    }

    fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, d_model) = x.dim();
        let d_ff = self.w1.dim().1;

        let mut hidden = Array3::zeros((batch, seq_len, d_ff));
        let mut output = Array3::zeros((batch, seq_len, d_model));

        // First linear + GELU activation
        for b in 0..batch {
            for t in 0..seq_len {
                for f in 0..d_ff {
                    let mut sum = self.b1[f];
                    for d in 0..d_model {
                        sum += x[[b, t, d]] * self.w1[[d, f]];
                    }
                    // GELU approximation
                    hidden[[b, t, f]] = gelu(sum);
                }
            }
        }

        // Second linear
        for b in 0..batch {
            for t in 0..seq_len {
                for d in 0..d_model {
                    let mut sum = self.b2[d];
                    for f in 0..d_ff {
                        sum += hidden[[b, t, f]] * self.w2[[f, d]];
                    }
                    output[[b, t, d]] = sum;
                }
            }
        }

        output
    }
}

/// Layer Normalization
#[derive(Debug, Clone)]
struct LayerNorm {
    gamma: Array1<f64>,
    beta: Array1<f64>,
    eps: f64,
}

impl LayerNorm {
    fn new(d_model: usize) -> Self {
        Self {
            gamma: Array1::ones(d_model),
            beta: Array1::zeros(d_model),
            eps: 1e-5,
        }
    }

    fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, d_model) = x.dim();
        let mut output = Array3::zeros((batch, seq_len, d_model));

        for b in 0..batch {
            for t in 0..seq_len {
                // Compute mean and variance
                let mut mean = 0.0;
                for d in 0..d_model {
                    mean += x[[b, t, d]];
                }
                mean /= d_model as f64;

                let mut var = 0.0;
                for d in 0..d_model {
                    var += (x[[b, t, d]] - mean).powi(2);
                }
                var /= d_model as f64;

                // Normalize
                for d in 0..d_model {
                    output[[b, t, d]] =
                        self.gamma[d] * (x[[b, t, d]] - mean) / (var + self.eps).sqrt() + self.beta[d];
                }
            }
        }

        output
    }
}

/// Encoder Layer
#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attention: ProbSparseAttention,
    cross_ticker_attention: Option<CrossTickerAttention>,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: Option<LayerNorm>,
    dropout: f64,
}

impl EncoderLayer {
    fn new(config: &StockformerConfig) -> Self {
        let cross_ticker = if config.use_cross_ticker_attention {
            Some(CrossTickerAttention::new(config))
        } else {
            None
        };

        let norm3 = if config.use_cross_ticker_attention {
            Some(LayerNorm::new(config.d_model))
        } else {
            None
        };

        Self {
            self_attention: ProbSparseAttention::new(config),
            cross_ticker_attention: cross_ticker,
            feed_forward: FeedForward::new(config.d_model, config.d_ff),
            norm1: LayerNorm::new(config.d_model),
            norm2: LayerNorm::new(config.d_model),
            norm3,
            dropout: config.dropout,
        }
    }
}

/// Основная модель Stockformer
#[derive(Debug)]
pub struct StockformerModel {
    /// Конфигурация
    config: StockformerConfig,
    /// Token embedding
    token_embedding: TokenEmbedding,
    /// Позиционное кодирование
    positional_encoding: Option<PositionalEncoding>,
    /// Тикер-специфичное кодирование
    ticker_encoding: TickerEncoding,
    /// Слои энкодера
    encoder_layers: Vec<EncoderLayer>,
    /// Выходной слой
    output_projection: Array2<f64>,
    /// Смещение выходного слоя
    output_bias: Array1<f64>,
}

impl StockformerModel {
    /// Создает новую модель Stockformer
    pub fn new(config: StockformerConfig) -> Result<Self, String> {
        config.validate()?;

        let token_embedding = TokenEmbedding::new(&config);

        let positional_encoding = if config.use_positional_encoding {
            Some(PositionalEncoding::new(config.d_model, config.seq_len * 2))
        } else {
            None
        };

        let ticker_encoding = TickerEncoding::new(config.num_tickers, config.d_model);

        let encoder_layers: Vec<EncoderLayer> = (0..config.num_encoder_layers)
            .map(|_| EncoderLayer::new(&config))
            .collect();

        let output_dim = config.output_dim();
        let scale = (2.0 / (config.d_model + output_dim) as f64).sqrt();
        let output_projection = Array2::from_shape_fn(
            (config.d_model, output_dim),
            |_| rand_normal() * scale
        );
        let output_bias = Array1::zeros(output_dim);

        Ok(Self {
            config,
            token_embedding,
            positional_encoding,
            ticker_encoding,
            encoder_layers,
            output_projection,
            output_bias,
        })
    }

    /// Прямой проход модели
    ///
    /// # Arguments
    /// * `x` - Входные данные [batch, num_tickers, seq_len, features]
    ///
    /// # Returns
    /// * `PredictionResult` - Предсказания и веса внимания
    pub fn forward(&self, x: &Array4<f64>) -> PredictionResult {
        let (batch_size, num_tickers, seq_len, _features) = x.dim();

        // 1. Token Embedding для каждого тикера
        let mut embedded = Array4::zeros((batch_size, num_tickers, seq_len, self.config.d_model));

        for b in 0..batch_size {
            for t in 0..num_tickers {
                let ticker_data = x.slice(s![b..b+1, t, .., ..]).to_owned();
                let ticker_3d = ticker_data.into_shape((1, seq_len, x.dim().3)).unwrap();
                let emb = self.token_embedding.forward(&ticker_3d);

                for s in 0..seq_len {
                    for d in 0..self.config.d_model {
                        embedded[[b, t, s, d]] = emb[[0, s, d]];
                    }
                }
            }
        }

        // 2. Позиционное кодирование
        if let Some(ref pe) = self.positional_encoding {
            for b in 0..batch_size {
                for t in 0..num_tickers {
                    let slice = embedded.slice(s![b..b+1, t, .., ..]).to_owned();
                    let slice_3d = slice.into_shape((1, seq_len, self.config.d_model)).unwrap();
                    let pe_applied = pe.forward(&slice_3d);

                    for s in 0..seq_len {
                        for d in 0..self.config.d_model {
                            embedded[[b, t, s, d]] = pe_applied[[0, s, d]];
                        }
                    }
                }
            }
        }

        // 3. Тикер-специфичное кодирование
        let embedded = self.ticker_encoding.forward(&embedded);

        // 4. Проход через энкодер
        let mut all_attention_weights = AttentionWeights::new();
        let mut current = embedded;

        for layer in &self.encoder_layers {
            // Self-attention для каждого тикера
            let mut after_self_attn = Array4::zeros(current.dim());

            for t in 0..num_tickers {
                let ticker_slice = current.slice(s![.., t, .., ..]).to_owned();
                let (attn_out, weights) = layer.self_attention.forward(&ticker_slice);

                // Residual + Norm
                for b in 0..batch_size {
                    for s in 0..seq_len {
                        for d in 0..self.config.d_model {
                            after_self_attn[[b, t, s, d]] = current[[b, t, s, d]] + attn_out[[b, s, d]];
                        }
                    }
                }

                // Сохраняем веса внимания первого слоя
                if all_attention_weights.temporal_weights.is_none() {
                    all_attention_weights.temporal_weights = weights.temporal_weights;
                }
            }

            // Layer Norm после self-attention
            let normed = self.apply_layer_norm_4d(&after_self_attn, &layer.norm1);

            // Cross-ticker attention
            let after_cross = if let Some(ref cross_attn) = layer.cross_ticker_attention {
                let (cross_out, cross_weights) = cross_attn.forward(&normed);

                if all_attention_weights.cross_ticker_weights.is_none() {
                    all_attention_weights.cross_ticker_weights = cross_weights.cross_ticker_weights;
                }

                // Residual
                let mut result = normed.clone();
                for b in 0..batch_size {
                    for t in 0..num_tickers {
                        for s in 0..seq_len {
                            for d in 0..self.config.d_model {
                                result[[b, t, s, d]] += cross_out[[b, t, s, d]];
                            }
                        }
                    }
                }

                if let Some(ref norm3) = layer.norm3 {
                    self.apply_layer_norm_4d(&result, norm3)
                } else {
                    result
                }
            } else {
                normed
            };

            // Feed-forward для каждого тикера
            let mut after_ff = Array4::zeros(after_cross.dim());

            for t in 0..num_tickers {
                let ticker_slice = after_cross.slice(s![.., t, .., ..]).to_owned();
                let ff_out = layer.feed_forward.forward(&ticker_slice);

                // Residual
                for b in 0..batch_size {
                    for s in 0..seq_len {
                        for d in 0..self.config.d_model {
                            after_ff[[b, t, s, d]] = after_cross[[b, t, s, d]] + ff_out[[b, s, d]];
                        }
                    }
                }
            }

            current = self.apply_layer_norm_4d(&after_ff, &layer.norm2);
        }

        // 5. Pooling - берем последний timestep
        let mut pooled = Array3::zeros((batch_size, num_tickers, self.config.d_model));
        for b in 0..batch_size {
            for t in 0..num_tickers {
                for d in 0..self.config.d_model {
                    pooled[[b, t, d]] = current[[b, t, seq_len - 1, d]];
                }
            }
        }

        // 6. Выходная проекция
        let predictions = self.output_head(&pooled);

        // 7. Вычисляем уверенность для квантильной регрессии
        let confidence = if self.config.output_type == OutputType::Quantile {
            Some(self.compute_confidence(&predictions))
        } else {
            None
        };

        PredictionResult {
            predictions,
            attention_weights: all_attention_weights,
            confidence,
        }
    }

    /// Применяет Layer Norm к 4D тензору
    fn apply_layer_norm_4d(&self, x: &Array4<f64>, norm: &LayerNorm) -> Array4<f64> {
        let (batch, num_tickers, seq_len, d_model) = x.dim();
        let mut result = Array4::zeros((batch, num_tickers, seq_len, d_model));

        for t in 0..num_tickers {
            let slice = x.slice(s![.., t, .., ..]).to_owned();
            let normed = norm.forward(&slice);

            for b in 0..batch {
                for s in 0..seq_len {
                    for d in 0..d_model {
                        result[[b, t, s, d]] = normed[[b, s, d]];
                    }
                }
            }
        }

        result
    }

    /// Выходной слой
    fn output_head(&self, x: &Array3<f64>) -> Array2<f64> {
        let (batch_size, num_tickers, d_model) = x.dim();
        let output_dim = self.config.output_dim();

        // Flatten тикеры и применяем проекцию
        let mut output = Array2::zeros((batch_size, output_dim));

        match self.config.output_type {
            OutputType::Regression | OutputType::Quantile => {
                // Каждый тикер получает свое предсказание
                for b in 0..batch_size {
                    for t in 0..num_tickers {
                        for o in 0..output_dim / num_tickers {
                            let out_idx = t * (output_dim / num_tickers) + o;
                            let mut sum = self.output_bias[out_idx];
                            for d in 0..d_model {
                                sum += x[[b, t, d]] * self.output_projection[[d, out_idx]];
                            }
                            output[[b, out_idx]] = sum;
                        }
                    }
                }
            }
            OutputType::Direction => {
                // 3 класса на тикер
                for b in 0..batch_size {
                    for t in 0..num_tickers {
                        for c in 0..3 {
                            let out_idx = t * 3 + c;
                            let mut sum = self.output_bias[out_idx];
                            for d in 0..d_model {
                                sum += x[[b, t, d]] * self.output_projection[[d, out_idx]];
                            }
                            output[[b, out_idx]] = sum;
                        }
                    }

                    // Softmax для каждого тикера
                    for t in 0..num_tickers {
                        let start = t * 3;
                        let max_val = (0..3)
                            .map(|c| output[[b, start + c]])
                            .fold(f64::NEG_INFINITY, f64::max);

                        let exp_sum: f64 = (0..3)
                            .map(|c| (output[[b, start + c]] - max_val).exp())
                            .sum();

                        for c in 0..3 {
                            output[[b, start + c]] =
                                (output[[b, start + c]] - max_val).exp() / exp_sum;
                        }
                    }
                }
            }
            OutputType::Portfolio => {
                // Softmax для распределения портфеля
                for b in 0..batch_size {
                    for t in 0..num_tickers {
                        let mut sum = self.output_bias[t];
                        for d in 0..d_model {
                            sum += x[[b, t, d]] * self.output_projection[[d, t]];
                        }
                        output[[b, t]] = sum;
                    }

                    // Softmax
                    let max_val = (0..num_tickers)
                        .map(|t| output[[b, t]])
                        .fold(f64::NEG_INFINITY, f64::max);

                    let exp_sum: f64 = (0..num_tickers)
                        .map(|t| (output[[b, t]] - max_val).exp())
                        .sum();

                    for t in 0..num_tickers {
                        output[[b, t]] = (output[[b, t]] - max_val).exp() / exp_sum;
                    }
                }
            }
        }

        output
    }

    /// Вычисляет уверенность из квантилей
    fn compute_confidence(&self, predictions: &Array2<f64>) -> Array2<f64> {
        let (batch_size, output_dim) = predictions.dim();
        let num_tickers = self.config.num_tickers;
        let num_quantiles = self.config.quantiles.len();

        let mut confidence = Array2::zeros((batch_size, num_tickers));

        for b in 0..batch_size {
            for t in 0..num_tickers {
                // Ширина интервала между крайними квантилями
                let low_idx = t * num_quantiles;
                let high_idx = t * num_quantiles + num_quantiles - 1;

                let interval_width = (predictions[[b, high_idx]] - predictions[[b, low_idx]]).abs();

                // Уверенность обратно пропорциональна ширине интервала
                confidence[[b, t]] = 1.0 / (1.0 + interval_width);
            }
        }

        confidence
    }

    /// Получает конфигурацию модели
    pub fn config(&self) -> &StockformerConfig {
        &self.config
    }

    /// Возвращает количество параметров модели
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        count += self.token_embedding.num_parameters();
        count += self.output_projection.len() + self.output_bias.len();

        // Encoder layers (примерная оценка)
        let layer_params = self.config.d_model * self.config.d_model * 4 // Q, K, V, O
            + self.config.d_model * self.config.d_ff * 2 // FFN
            + self.config.d_model * 4; // Layer norms

        count += layer_params * self.config.num_encoder_layers;

        if self.config.use_cross_ticker_attention {
            count += self.config.d_model * self.config.d_model * 4 * self.config.num_encoder_layers;
        }

        count
    }
}

/// GELU activation function
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Генерирует случайное число из нормального распределения
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
    fn test_model_creation() {
        let config = StockformerConfig::small();
        let model = StockformerModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_model_forward() {
        let config = StockformerConfig {
            num_tickers: 3,
            seq_len: 16,
            input_features: 6,
            d_model: 16,
            num_heads: 2,
            d_ff: 32,
            num_encoder_layers: 1,
            use_cross_ticker_attention: true,
            ..Default::default()
        };

        let model = StockformerModel::new(config.clone()).unwrap();

        // [batch=2, num_tickers=3, seq_len=16, features=6]
        let x = Array4::from_shape_fn((2, 3, 16, 6), |_| rand_normal() * 0.1);

        let result = model.forward(&x);

        assert_eq!(result.predictions.dim(), (2, 3)); // batch x num_tickers
        assert!(result.attention_weights.temporal_weights.is_some());
        assert!(result.attention_weights.cross_ticker_weights.is_some());
    }

    #[test]
    fn test_direction_output() {
        let config = StockformerConfig {
            num_tickers: 2,
            seq_len: 8,
            input_features: 4,
            d_model: 8,
            num_heads: 2,
            d_ff: 16,
            num_encoder_layers: 1,
            output_type: OutputType::Direction,
            use_cross_ticker_attention: false,
            ..Default::default()
        };

        let model = StockformerModel::new(config).unwrap();
        let x = Array4::from_shape_fn((1, 2, 8, 4), |_| rand_normal() * 0.1);

        let result = model.forward(&x);

        // 2 tickers * 3 classes = 6
        assert_eq!(result.predictions.dim(), (1, 6));

        // Check softmax sums to 1 for each ticker
        for t in 0..2 {
            let sum: f64 = (0..3).map(|c| result.predictions[[0, t * 3 + c]]).sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_portfolio_output() {
        let config = StockformerConfig {
            num_tickers: 4,
            seq_len: 8,
            input_features: 4,
            d_model: 8,
            num_heads: 2,
            d_ff: 16,
            num_encoder_layers: 1,
            output_type: OutputType::Portfolio,
            use_cross_ticker_attention: false,
            ..Default::default()
        };

        let model = StockformerModel::new(config).unwrap();
        let x = Array4::from_shape_fn((1, 4, 8, 4), |_| rand_normal() * 0.1);

        let result = model.forward(&x);

        // Check weights sum to 1
        let sum: f64 = result.predictions.row(0).sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check all weights are positive
        for &w in result.predictions.iter() {
            assert!(w >= 0.0);
        }
    }

    #[test]
    fn test_quantile_output() {
        let config = StockformerConfig {
            num_tickers: 2,
            seq_len: 8,
            input_features: 4,
            d_model: 8,
            num_heads: 2,
            d_ff: 16,
            num_encoder_layers: 1,
            output_type: OutputType::Quantile,
            quantiles: vec![0.1, 0.5, 0.9],
            use_cross_ticker_attention: false,
            ..Default::default()
        };

        let model = StockformerModel::new(config).unwrap();
        let x = Array4::from_shape_fn((1, 2, 8, 4), |_| rand_normal() * 0.1);

        let result = model.forward(&x);

        // 2 tickers * 3 quantiles = 6
        assert_eq!(result.predictions.dim(), (1, 6));
        assert!(result.confidence.is_some());

        let confidence = result.confidence.unwrap();
        assert_eq!(confidence.dim(), (1, 2));
    }

    #[test]
    fn test_gelu() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!(gelu(1.0) > 0.8);
        assert!(gelu(-1.0) < 0.0);
    }

    #[test]
    fn test_num_parameters() {
        let config = StockformerConfig::small();
        let model = StockformerModel::new(config).unwrap();

        let params = model.num_parameters();
        assert!(params > 0);
    }
}
