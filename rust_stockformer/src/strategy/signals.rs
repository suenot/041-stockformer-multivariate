//! Генерация торговых сигналов
//!
//! Преобразует предсказания модели в торговые сигналы

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::model::{PredictionResult, OutputType};

/// Тип торгового сигнала
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Длинная позиция (покупка)
    Long,
    /// Короткая позиция (продажа)
    Short,
    /// Удержание позиции
    Hold,
    /// Закрытие позиции
    Close,
}

/// Торговый сигнал для одного актива
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Тип сигнала
    pub signal_type: SignalType,
    /// Индекс тикера
    pub ticker_idx: usize,
    /// Сила сигнала [0, 1]
    pub strength: f64,
    /// Предсказанное изменение цены
    pub predicted_return: f64,
    /// Уверенность предсказания
    pub confidence: f64,
    /// Рекомендуемый размер позиции [0, 1]
    pub position_size: f64,
}

impl Signal {
    /// Создает новый сигнал
    pub fn new(
        signal_type: SignalType,
        ticker_idx: usize,
        strength: f64,
        predicted_return: f64,
        confidence: f64,
    ) -> Self {
        let position_size = strength * confidence;

        Self {
            signal_type,
            ticker_idx,
            strength,
            predicted_return,
            confidence,
            position_size,
        }
    }

    /// Проверяет, является ли сигнал действенным
    pub fn is_actionable(&self, min_strength: f64, min_confidence: f64) -> bool {
        self.signal_type != SignalType::Hold
            && self.strength >= min_strength
            && self.confidence >= min_confidence
    }
}

/// Сигнал для портфеля (несколько активов)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSignal {
    /// Сигналы для каждого актива
    pub signals: Vec<Signal>,
    /// Рекомендуемые веса портфеля
    pub weights: Vec<f64>,
    /// Общая уверенность
    pub overall_confidence: f64,
    /// Временная метка
    pub timestamp: i64,
}

impl PortfolioSignal {
    /// Создает новый портфельный сигнал
    pub fn new(signals: Vec<Signal>, weights: Vec<f64>) -> Self {
        let overall_confidence = if signals.is_empty() {
            0.0
        } else {
            signals.iter().map(|s| s.confidence).sum::<f64>() / signals.len() as f64
        };

        Self {
            signals,
            weights,
            overall_confidence,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Возвращает активы для покупки
    pub fn longs(&self) -> Vec<&Signal> {
        self.signals
            .iter()
            .filter(|s| s.signal_type == SignalType::Long)
            .collect()
    }

    /// Возвращает активы для продажи
    pub fn shorts(&self) -> Vec<&Signal> {
        self.signals
            .iter()
            .filter(|s| s.signal_type == SignalType::Short)
            .collect()
    }
}

/// Генератор торговых сигналов
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Порог для длинного сигнала
    pub long_threshold: f64,
    /// Порог для короткого сигнала
    pub short_threshold: f64,
    /// Минимальная уверенность
    pub min_confidence: f64,
    /// Использовать Kelly criterion для размера позиции
    pub use_kelly: bool,
    /// Максимальный размер позиции
    pub max_position_size: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self {
            long_threshold: 0.005,   // 0.5% ожидаемый рост
            short_threshold: -0.005, // 0.5% ожидаемое падение
            min_confidence: 0.3,
            use_kelly: true,
            max_position_size: 0.25,
        }
    }
}

impl SignalGenerator {
    /// Создает новый генератор сигналов
    pub fn new() -> Self {
        Self::default()
    }

    /// Устанавливает пороги
    pub fn with_thresholds(mut self, long: f64, short: f64) -> Self {
        self.long_threshold = long;
        self.short_threshold = short;
        self
    }

    /// Устанавливает минимальную уверенность
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Генерирует сигналы из предсказаний модели
    pub fn generate(
        &self,
        predictions: &PredictionResult,
        output_type: OutputType,
        ticker_names: &[String],
    ) -> PortfolioSignal {
        let num_tickers = ticker_names.len();

        match output_type {
            OutputType::Regression => self.generate_from_regression(predictions, num_tickers),
            OutputType::Direction => self.generate_from_direction(predictions, num_tickers),
            OutputType::Portfolio => self.generate_from_portfolio(predictions, num_tickers),
            OutputType::Quantile => self.generate_from_quantile(predictions, num_tickers),
        }
    }

    /// Генерирует сигналы из регрессионных предсказаний
    fn generate_from_regression(
        &self,
        predictions: &PredictionResult,
        num_tickers: usize,
    ) -> PortfolioSignal {
        let mut signals = Vec::with_capacity(num_tickers);
        let mut weights = vec![0.0; num_tickers];

        let preds = &predictions.predictions;
        let batch_idx = 0; // Берем первый элемент батча

        for t in 0..num_tickers {
            let predicted_return = preds[[batch_idx, t]];

            let confidence = predictions
                .confidence
                .as_ref()
                .map(|c| c[[batch_idx, t]])
                .unwrap_or(0.5);

            let (signal_type, strength) = if predicted_return > self.long_threshold {
                (SignalType::Long, (predicted_return / self.long_threshold).min(1.0))
            } else if predicted_return < self.short_threshold {
                (SignalType::Short, (predicted_return / self.short_threshold).min(1.0))
            } else {
                (SignalType::Hold, 0.0)
            };

            let signal = Signal::new(signal_type, t, strength, predicted_return, confidence);

            // Kelly criterion для размера позиции
            if self.use_kelly && signal.is_actionable(0.0, self.min_confidence) {
                let kelly_fraction = self.kelly_criterion(predicted_return, confidence);
                weights[t] = kelly_fraction.min(self.max_position_size);
            }

            signals.push(signal);
        }

        // Нормализуем веса
        let total_weight: f64 = weights.iter().map(|w| w.abs()).sum();
        if total_weight > 1.0 {
            for w in &mut weights {
                *w /= total_weight;
            }
        }

        PortfolioSignal::new(signals, weights)
    }

    /// Генерирует сигналы из направленных предсказаний
    fn generate_from_direction(
        &self,
        predictions: &PredictionResult,
        num_tickers: usize,
    ) -> PortfolioSignal {
        let mut signals = Vec::with_capacity(num_tickers);
        let mut weights = vec![0.0; num_tickers];

        let preds = &predictions.predictions;
        let batch_idx = 0;

        for t in 0..num_tickers {
            // [down, hold, up] probabilities
            let p_down = preds[[batch_idx, t * 3]];
            let p_hold = preds[[batch_idx, t * 3 + 1]];
            let p_up = preds[[batch_idx, t * 3 + 2]];

            let (signal_type, strength, predicted_return) = if p_up > p_down && p_up > p_hold {
                (SignalType::Long, p_up, p_up - p_down)
            } else if p_down > p_up && p_down > p_hold {
                (SignalType::Short, p_down, p_down - p_up)
            } else {
                (SignalType::Hold, p_hold, 0.0)
            };

            // Confidence based on max probability
            let confidence = p_up.max(p_down).max(p_hold);

            let signal = Signal::new(signal_type, t, strength, predicted_return, confidence);

            if signal.is_actionable(0.5, self.min_confidence) {
                weights[t] = signal.position_size;
            }

            signals.push(signal);
        }

        // Нормализуем веса
        let total: f64 = weights.iter().map(|w| w.abs()).sum();
        if total > 1.0 {
            for w in &mut weights {
                *w /= total;
            }
        }

        PortfolioSignal::new(signals, weights)
    }

    /// Генерирует сигналы из портфельных весов
    fn generate_from_portfolio(
        &self,
        predictions: &PredictionResult,
        num_tickers: usize,
    ) -> PortfolioSignal {
        let mut signals = Vec::with_capacity(num_tickers);
        let batch_idx = 0;

        let weights: Vec<f64> = (0..num_tickers)
            .map(|t| predictions.predictions[[batch_idx, t]])
            .collect();

        let equal_weight = 1.0 / num_tickers as f64;

        for t in 0..num_tickers {
            let weight = weights[t];
            let deviation = weight - equal_weight;

            let (signal_type, strength) = if deviation > 0.05 {
                (SignalType::Long, (deviation / equal_weight).min(1.0))
            } else if deviation < -0.05 {
                (SignalType::Short, (-deviation / equal_weight).min(1.0))
            } else {
                (SignalType::Hold, 0.0)
            };

            let confidence = 1.0 - weight.abs().min(1.0); // Higher weight = more confidence
            let signal = Signal::new(signal_type, t, strength, deviation, confidence);

            signals.push(signal);
        }

        PortfolioSignal::new(signals, weights)
    }

    /// Генерирует сигналы из квантильных предсказаний
    fn generate_from_quantile(
        &self,
        predictions: &PredictionResult,
        num_tickers: usize,
    ) -> PortfolioSignal {
        let mut signals = Vec::with_capacity(num_tickers);
        let mut weights = vec![0.0; num_tickers];

        let preds = &predictions.predictions;
        let batch_idx = 0;

        // Предполагаем 3 квантиля: 0.1, 0.5, 0.9
        let num_quantiles = 3;

        for t in 0..num_tickers {
            let q_low = preds[[batch_idx, t * num_quantiles]];     // 10%
            let q_med = preds[[batch_idx, t * num_quantiles + 1]]; // 50% (медиана)
            let q_high = preds[[batch_idx, t * num_quantiles + 2]]; // 90%

            let predicted_return = q_med;
            let interval_width = (q_high - q_low).abs();

            // Уверенность обратно пропорциональна ширине интервала
            let confidence = 1.0 / (1.0 + interval_width);

            let (signal_type, strength) = if q_low > 0.0 {
                // Даже 10% квантиль положительный - сильный long
                (SignalType::Long, 1.0)
            } else if q_high < 0.0 {
                // Даже 90% квантиль отрицательный - сильный short
                (SignalType::Short, 1.0)
            } else if q_med > self.long_threshold {
                (SignalType::Long, (q_med / self.long_threshold).min(1.0) * confidence)
            } else if q_med < self.short_threshold {
                (SignalType::Short, (q_med / self.short_threshold).min(1.0) * confidence)
            } else {
                (SignalType::Hold, 0.0)
            };

            let signal = Signal::new(signal_type, t, strength, predicted_return, confidence);

            if signal.is_actionable(0.0, self.min_confidence) {
                weights[t] = signal.position_size;
            }

            signals.push(signal);
        }

        // Нормализуем веса
        let total: f64 = weights.iter().map(|w| w.abs()).sum();
        if total > 1.0 {
            for w in &mut weights {
                *w /= total;
            }
        }

        PortfolioSignal::new(signals, weights)
    }

    /// Kelly criterion для оптимального размера позиции
    fn kelly_criterion(&self, expected_return: f64, win_probability: f64) -> f64 {
        // Упрощенная формула Kelly: f = p - (1-p)/b
        // где p = вероятность выигрыша, b = отношение выигрыша к проигрышу

        if expected_return.abs() < 1e-8 {
            return 0.0;
        }

        let p = win_probability;
        let b = expected_return.abs(); // Предполагаем симметричный риск/награду

        let kelly = if b > 0.0 {
            (p * (1.0 + b) - 1.0) / b
        } else {
            0.0
        };

        // Используем половину Kelly для консервативности
        (kelly * 0.5).max(0.0).min(self.max_position_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn mock_regression_predictions(returns: Vec<f64>) -> PredictionResult {
        let num_tickers = returns.len();
        let predictions = Array2::from_shape_vec((1, num_tickers), returns).unwrap();

        PredictionResult {
            predictions,
            attention_weights: crate::model::AttentionWeights::new(),
            confidence: Some(Array2::from_elem((1, num_tickers), 0.7)),
        }
    }

    #[test]
    fn test_signal_generator_long() {
        let gen = SignalGenerator::new();
        let preds = mock_regression_predictions(vec![0.02, -0.01, 0.0]); // 2%, -1%, 0%

        let portfolio = gen.generate(&preds, OutputType::Regression, &["A".into(), "B".into(), "C".into()]);

        assert_eq!(portfolio.signals[0].signal_type, SignalType::Long);
        assert_eq!(portfolio.signals[1].signal_type, SignalType::Short);
        assert_eq!(portfolio.signals[2].signal_type, SignalType::Hold);
    }

    #[test]
    fn test_signal_actionable() {
        let signal = Signal::new(SignalType::Long, 0, 0.8, 0.02, 0.7);
        assert!(signal.is_actionable(0.5, 0.5));

        let weak_signal = Signal::new(SignalType::Long, 0, 0.3, 0.01, 0.7);
        assert!(!weak_signal.is_actionable(0.5, 0.5));
    }

    #[test]
    fn test_portfolio_weights_normalized() {
        let gen = SignalGenerator::new();
        let preds = mock_regression_predictions(vec![0.1, 0.1, 0.1, 0.1, 0.1]);

        let portfolio = gen.generate(
            &preds,
            OutputType::Regression,
            &["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
        );

        let total: f64 = portfolio.weights.iter().sum();
        assert!(total <= 1.0 + 1e-6);
    }

    #[test]
    fn test_direction_signals() {
        let num_tickers = 2;
        // [down, hold, up] for each ticker
        let predictions = Array2::from_shape_vec(
            (1, num_tickers * 3),
            vec![0.1, 0.2, 0.7, 0.6, 0.3, 0.1],
        )
        .unwrap();

        let result = PredictionResult {
            predictions,
            attention_weights: crate::model::AttentionWeights::new(),
            confidence: None,
        };

        let gen = SignalGenerator::new();
        let portfolio = gen.generate(&result, OutputType::Direction, &["A".into(), "B".into()]);

        assert_eq!(portfolio.signals[0].signal_type, SignalType::Long);  // up=0.7
        assert_eq!(portfolio.signals[1].signal_type, SignalType::Short); // down=0.6
    }

    #[test]
    fn test_kelly_criterion() {
        let gen = SignalGenerator::new();

        // Высокая вероятность, положительный возврат
        let kelly = gen.kelly_criterion(0.1, 0.7);
        assert!(kelly > 0.0);
        assert!(kelly <= gen.max_position_size);

        // Низкая вероятность
        let kelly_low = gen.kelly_criterion(0.1, 0.3);
        assert!(kelly_low < kelly);
    }

    #[test]
    fn test_portfolio_longs_shorts() {
        let signals = vec![
            Signal::new(SignalType::Long, 0, 0.8, 0.02, 0.7),
            Signal::new(SignalType::Short, 1, 0.6, -0.01, 0.6),
            Signal::new(SignalType::Hold, 2, 0.0, 0.0, 0.5),
        ];

        let portfolio = PortfolioSignal::new(signals, vec![0.4, -0.3, 0.0]);

        assert_eq!(portfolio.longs().len(), 1);
        assert_eq!(portfolio.shorts().len(), 1);
    }
}
