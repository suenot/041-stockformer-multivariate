//! Механизмы внимания для Stockformer
//!
//! Реализует:
//! - Cross-Ticker Attention (внимание между активами)
//! - ProbSparse Attention (эффективное разреженное внимание)

use ndarray::{Array1, Array2, Array3, Array4, Axis, s};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::model::config::StockformerConfig;

/// Веса внимания для интерпретации
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Матрица внимания [batch, num_heads, seq_len, seq_len]
    pub temporal_weights: Option<Array4<f64>>,
    /// Кросс-тикерные веса [batch, num_heads, num_tickers, num_tickers]
    pub cross_ticker_weights: Option<Array4<f64>>,
}

impl AttentionWeights {
    pub fn new() -> Self {
        Self {
            temporal_weights: None,
            cross_ticker_weights: None,
        }
    }

    /// Получает топ-k важных связей между тикерами
    pub fn top_k_cross_ticker(&self, k: usize) -> Vec<(usize, usize, f64)> {
        let mut results = Vec::new();

        if let Some(ref weights) = self.cross_ticker_weights {
            let (_, _, num_tickers, _) = weights.dim();

            // Усредняем по batch и heads
            let mean_weights = weights.mean_axis(Axis(0)).unwrap()
                .mean_axis(Axis(0)).unwrap();

            for i in 0..num_tickers {
                for j in 0..num_tickers {
                    if i != j {
                        results.push((i, j, mean_weights[[i, j]]));
                    }
                }
            }

            results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
            results.truncate(k);
        }

        results
    }
}

impl Default for AttentionWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-Ticker Attention
///
/// Моделирует зависимости между различными активами
#[derive(Debug, Clone)]
pub struct CrossTickerAttention {
    /// Query проекция [d_model, d_model]
    w_q: Array2<f64>,
    /// Key проекция [d_model, d_model]
    w_k: Array2<f64>,
    /// Value проекция [d_model, d_model]
    w_v: Array2<f64>,
    /// Output проекция [d_model, d_model]
    w_o: Array2<f64>,
    /// Количество голов
    num_heads: usize,
    /// Размерность головы
    head_dim: usize,
    /// Масштабирующий коэффициент
    scale: f64,
}

impl CrossTickerAttention {
    /// Создает новый Cross-Ticker Attention слой
    pub fn new(config: &StockformerConfig) -> Self {
        let d_model = config.d_model;
        let num_heads = config.num_heads;
        let head_dim = d_model / num_heads;

        // Xavier инициализация
        let scale_init = (2.0 / (d_model * 2) as f64).sqrt();

        let w_q = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);

        let scale = (head_dim as f64).sqrt();

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
            head_dim,
            scale,
        }
    }

    /// Прямой проход Cross-Ticker Attention
    ///
    /// # Arguments
    /// * `x` - [batch, num_tickers, seq_len, d_model]
    ///
    /// # Returns
    /// * `output` - [batch, num_tickers, seq_len, d_model]
    /// * `weights` - Веса внимания для интерпретации
    pub fn forward(&self, x: &Array4<f64>) -> (Array4<f64>, AttentionWeights) {
        let (batch_size, num_tickers, seq_len, d_model) = x.dim();

        // Для каждой временной позиции применяем внимание между тикерами
        let mut output = Array4::zeros((batch_size, num_tickers, seq_len, d_model));
        let mut all_weights = Array4::zeros((batch_size, self.num_heads, num_tickers, num_tickers));

        for t in 0..seq_len {
            // Извлекаем срез для временной позиции t: [batch, num_tickers, d_model]
            let x_t = x.slice(s![.., .., t, ..]).to_owned();

            // Вычисляем Q, K, V
            let q = self.linear_transform(&x_t, &self.w_q);
            let k = self.linear_transform(&x_t, &self.w_k);
            let v = self.linear_transform(&x_t, &self.w_v);

            // Multi-head attention между тикерами
            let (attn_output, attn_weights) = self.multi_head_attention(&q, &k, &v);

            // Записываем результат
            for b in 0..batch_size {
                for ticker in 0..num_tickers {
                    for d in 0..d_model {
                        output[[b, ticker, t, d]] = attn_output[[b, ticker, d]];
                    }
                }
            }

            // Накапливаем веса (усредняем по времени)
            for b in 0..batch_size {
                for h in 0..self.num_heads {
                    for i in 0..num_tickers {
                        for j in 0..num_tickers {
                            all_weights[[b, h, i, j]] += attn_weights[[b, h, i, j]] / seq_len as f64;
                        }
                    }
                }
            }
        }

        let weights = AttentionWeights {
            temporal_weights: None,
            cross_ticker_weights: Some(all_weights),
        };

        (output, weights)
    }

    /// Линейное преобразование
    fn linear_transform(&self, x: &Array3<f64>, w: &Array2<f64>) -> Array3<f64> {
        let (batch_size, num_tickers, d_model) = x.dim();
        let mut output = Array3::zeros((batch_size, num_tickers, d_model));

        for b in 0..batch_size {
            for t in 0..num_tickers {
                for d_out in 0..d_model {
                    let mut sum = 0.0;
                    for d_in in 0..d_model {
                        sum += x[[b, t, d_in]] * w[[d_in, d_out]];
                    }
                    output[[b, t, d_out]] = sum;
                }
            }
        }

        output
    }

    /// Multi-head attention
    fn multi_head_attention(
        &self,
        q: &Array3<f64>,
        k: &Array3<f64>,
        v: &Array3<f64>,
    ) -> (Array3<f64>, Array4<f64>) {
        let (batch_size, num_tickers, d_model) = q.dim();

        // Reshape для multi-head: [batch, num_tickers, num_heads, head_dim]
        // Затем транспонируем: [batch, num_heads, num_tickers, head_dim]

        let mut attn_weights = Array4::zeros((batch_size, self.num_heads, num_tickers, num_tickers));
        let mut output = Array3::zeros((batch_size, num_tickers, d_model));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                let h_start = h * self.head_dim;
                let h_end = (h + 1) * self.head_dim;

                // Вычисляем scores: Q @ K^T / sqrt(head_dim)
                for i in 0..num_tickers {
                    for j in 0..num_tickers {
                        let mut score = 0.0;
                        for d in h_start..h_end {
                            score += q[[b, i, d]] * k[[b, j, d]];
                        }
                        attn_weights[[b, h, i, j]] = score / self.scale;
                    }
                }

                // Softmax по последнему измерению
                for i in 0..num_tickers {
                    let max_val = (0..num_tickers)
                        .map(|j| attn_weights[[b, h, i, j]])
                        .fold(f64::NEG_INFINITY, f64::max);

                    let exp_sum: f64 = (0..num_tickers)
                        .map(|j| (attn_weights[[b, h, i, j]] - max_val).exp())
                        .sum();

                    for j in 0..num_tickers {
                        attn_weights[[b, h, i, j]] =
                            (attn_weights[[b, h, i, j]] - max_val).exp() / exp_sum;
                    }
                }

                // Применяем внимание: weights @ V
                for i in 0..num_tickers {
                    for d in h_start..h_end {
                        let mut sum = 0.0;
                        for j in 0..num_tickers {
                            sum += attn_weights[[b, h, i, j]] * v[[b, j, d]];
                        }
                        output[[b, i, d]] = sum;
                    }
                }
            }
        }

        // Применяем выходную проекцию
        let projected = self.linear_transform(&output, &self.w_o);

        (projected, attn_weights)
    }
}

/// ProbSparse Attention
///
/// Эффективный механизм внимания с вычислительной сложностью O(L·log(L))
#[derive(Debug, Clone)]
pub struct ProbSparseAttention {
    /// Query проекция
    w_q: Array2<f64>,
    /// Key проекция
    w_k: Array2<f64>,
    /// Value проекция
    w_v: Array2<f64>,
    /// Output проекция
    w_o: Array2<f64>,
    /// Количество голов
    num_heads: usize,
    /// Размерность головы
    head_dim: usize,
    /// Масштабирующий коэффициент
    scale: f64,
    /// Фактор разреженности
    sparsity_factor: f64,
}

impl ProbSparseAttention {
    /// Создает новый ProbSparse Attention слой
    pub fn new(config: &StockformerConfig) -> Self {
        let d_model = config.d_model;
        let num_heads = config.num_heads;
        let head_dim = d_model / num_heads;

        let scale_init = (2.0 / (d_model * 2) as f64).sqrt();

        let w_q = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
            head_dim,
            scale: (head_dim as f64).sqrt(),
            sparsity_factor: config.sparsity_factor,
        }
    }

    /// Прямой проход ProbSparse Attention
    ///
    /// # Arguments
    /// * `x` - [batch, seq_len, d_model]
    ///
    /// # Returns
    /// * `output` - [batch, seq_len, d_model]
    /// * `weights` - Веса внимания
    pub fn forward(&self, x: &Array3<f64>) -> (Array3<f64>, AttentionWeights) {
        let (batch_size, seq_len, d_model) = x.dim();

        // Линейные проекции
        let q = self.linear_transform_3d(x, &self.w_q);
        let k = self.linear_transform_3d(x, &self.w_k);
        let v = self.linear_transform_3d(x, &self.w_v);

        // Вычисляем количество выбираемых запросов
        let u = ((self.sparsity_factor * (seq_len as f64).ln()).ceil() as usize).max(1).min(seq_len);

        let mut output = Array3::zeros((batch_size, seq_len, d_model));
        let mut all_weights = Array4::zeros((batch_size, self.num_heads, seq_len, seq_len));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                let h_start = h * self.head_dim;
                let h_end = (h + 1) * self.head_dim;

                // Вычисляем sparsity measurement M для выбора top-u запросов
                let top_indices = self.select_top_queries(
                    &q.slice(s![b, .., h_start..h_end]),
                    &k.slice(s![b, .., h_start..h_end]),
                    u,
                );

                // Полное внимание только для выбранных запросов
                for &i in &top_indices {
                    for j in 0..seq_len {
                        let mut score = 0.0;
                        for d in h_start..h_end {
                            score += q[[b, i, d]] * k[[b, j, d]];
                        }
                        all_weights[[b, h, i, j]] = score / self.scale;
                    }

                    // Softmax
                    let max_val = (0..seq_len)
                        .map(|j| all_weights[[b, h, i, j]])
                        .fold(f64::NEG_INFINITY, f64::max);

                    let exp_sum: f64 = (0..seq_len)
                        .map(|j| (all_weights[[b, h, i, j]] - max_val).exp())
                        .sum();

                    for j in 0..seq_len {
                        all_weights[[b, h, i, j]] =
                            (all_weights[[b, h, i, j]] - max_val).exp() / exp_sum;
                    }

                    // Применяем внимание
                    for d in h_start..h_end {
                        let mut sum = 0.0;
                        for j in 0..seq_len {
                            sum += all_weights[[b, h, i, j]] * v[[b, j, d]];
                        }
                        output[[b, i, d]] = sum;
                    }
                }

                // Для остальных позиций используем усредненное значение
                let mean_v: Array1<f64> = v.slice(s![b, .., h_start..h_end])
                    .mean_axis(Axis(0))
                    .unwrap();

                for i in 0..seq_len {
                    if !top_indices.contains(&i) {
                        for (d_idx, d) in (h_start..h_end).enumerate() {
                            output[[b, i, d]] = mean_v[d_idx];
                        }
                    }
                }
            }
        }

        // Выходная проекция
        let projected = self.linear_transform_3d(&output, &self.w_o);

        let weights = AttentionWeights {
            temporal_weights: Some(all_weights),
            cross_ticker_weights: None,
        };

        (projected, weights)
    }

    /// Линейное преобразование для 3D тензора
    fn linear_transform_3d(&self, x: &Array3<f64>, w: &Array2<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_in) = x.dim();
        let d_out = w.dim().1;
        let mut output = Array3::zeros((batch_size, seq_len, d_out));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d_o in 0..d_out {
                    let mut sum = 0.0;
                    for d_i in 0..d_in {
                        sum += x[[b, t, d_i]] * w[[d_i, d_o]];
                    }
                    output[[b, t, d_o]] = sum;
                }
            }
        }

        output
    }

    /// Выбирает top-u запросов по sparsity measurement
    fn select_top_queries(
        &self,
        q: &ndarray::ArrayView2<f64>,
        k: &ndarray::ArrayView2<f64>,
        u: usize,
    ) -> Vec<usize> {
        let seq_len = q.dim().0;

        // Вычисляем M(q_i) = max(Q_i @ K^T) - mean(Q_i @ K^T)
        // Это измеряет "остроту" распределения внимания
        let mut measurements: Vec<(usize, f64)> = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let mut scores: Vec<f64> = Vec::with_capacity(seq_len);

            for j in 0..seq_len {
                let mut score = 0.0;
                for d in 0..q.dim().1 {
                    score += q[[i, d]] * k[[j, d]];
                }
                scores.push(score / self.scale);
            }

            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean_score: f64 = scores.iter().sum::<f64>() / seq_len as f64;

            measurements.push((i, max_score - mean_score));
        }

        // Сортируем по M в убывающем порядке и выбираем top-u
        measurements.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        measurements.iter().take(u).map(|(idx, _)| *idx).collect()
    }
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
    fn test_cross_ticker_attention() {
        let config = StockformerConfig {
            d_model: 32,
            num_heads: 4,
            ..Default::default()
        };

        let attn = CrossTickerAttention::new(&config);

        // [batch=2, num_tickers=3, seq_len=10, d_model=32]
        let x = Array4::from_shape_fn((2, 3, 10, 32), |_| rand_normal());

        let (output, weights) = attn.forward(&x);

        assert_eq!(output.dim(), (2, 3, 10, 32));
        assert!(weights.cross_ticker_weights.is_some());

        let cw = weights.cross_ticker_weights.unwrap();
        assert_eq!(cw.dim(), (2, 4, 3, 3));
    }

    #[test]
    fn test_probsparse_attention() {
        let config = StockformerConfig {
            d_model: 32,
            num_heads: 4,
            sparsity_factor: 5.0,
            ..Default::default()
        };

        let attn = ProbSparseAttention::new(&config);

        // [batch=2, seq_len=16, d_model=32]
        let x = Array3::from_shape_fn((2, 16, 32), |_| rand_normal());

        let (output, weights) = attn.forward(&x);

        assert_eq!(output.dim(), (2, 16, 32));
        assert!(weights.temporal_weights.is_some());
    }

    #[test]
    fn test_attention_weights_top_k() {
        let mut weights = AttentionWeights::new();

        // Создаем тестовые веса
        let mut cw = Array4::zeros((1, 2, 3, 3));
        cw[[0, 0, 0, 1]] = 0.8;
        cw[[0, 0, 1, 2]] = 0.6;
        cw[[0, 1, 2, 0]] = 0.9;

        weights.cross_ticker_weights = Some(cw);

        let top = weights.top_k_cross_ticker(2);

        assert_eq!(top.len(), 2);
        // Первый должен быть с наибольшим весом
        assert!(top[0].2 > top[1].2);
    }

    #[test]
    fn test_softmax_normalization() {
        let config = StockformerConfig {
            d_model: 8,
            num_heads: 2,
            ..Default::default()
        };

        let attn = CrossTickerAttention::new(&config);
        let x = Array4::from_shape_fn((1, 4, 5, 8), |_| rand_normal());

        let (_, weights) = attn.forward(&x);
        let cw = weights.cross_ticker_weights.unwrap();

        // Проверяем, что строки суммируются в 1 (softmax)
        for b in 0..cw.dim().0 {
            for h in 0..cw.dim().1 {
                for i in 0..cw.dim().2 {
                    let sum: f64 = (0..cw.dim().3).map(|j| cw[[b, h, i, j]]).sum();
                    assert!((sum - 1.0).abs() < 1e-6, "Softmax sum should be 1, got {}", sum);
                }
            }
        }
    }
}
