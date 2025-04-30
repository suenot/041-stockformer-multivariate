//! Типы данных для Bybit API

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Ошибки при работе с Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("API error: code={code}, message={message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),
}

/// Базовый ответ от Bybit API
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: Option<T>,
    pub time: Option<u64>,
}

/// Данные свечи (kline)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Время открытия свечи (Unix timestamp в миллисекундах)
    pub start_time: u64,

    /// Цена открытия
    pub open: f64,

    /// Максимальная цена
    pub high: f64,

    /// Минимальная цена
    pub low: f64,

    /// Цена закрытия
    pub close: f64,

    /// Объём в базовой валюте
    pub volume: f64,

    /// Оборот в котируемой валюте
    pub turnover: f64,
}

impl Kline {
    /// Создаёт Kline из массива строк (формат Bybit API)
    pub fn from_bybit_array(arr: &[serde_json::Value]) -> Result<Self, BybitError> {
        if arr.len() < 7 {
            return Err(BybitError::ParseError(
                "Invalid kline array length".to_string(),
            ));
        }

        let parse_f64 = |v: &serde_json::Value| -> Result<f64, BybitError> {
            v.as_str()
                .and_then(|s| s.parse::<f64>().ok())
                .or_else(|| v.as_f64())
                .ok_or_else(|| BybitError::ParseError("Failed to parse f64".to_string()))
        };

        let parse_u64 = |v: &serde_json::Value| -> Result<u64, BybitError> {
            v.as_str()
                .and_then(|s| s.parse::<u64>().ok())
                .or_else(|| v.as_u64())
                .ok_or_else(|| BybitError::ParseError("Failed to parse u64".to_string()))
        };

        Ok(Kline {
            start_time: parse_u64(&arr[0])?,
            open: parse_f64(&arr[1])?,
            high: parse_f64(&arr[2])?,
            low: parse_f64(&arr[3])?,
            close: parse_f64(&arr[4])?,
            volume: parse_f64(&arr[5])?,
            turnover: parse_f64(&arr[6])?,
        })
    }

    /// Возвращает время открытия как DateTime
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.start_time as i64)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap())
    }

    /// Рассчитывает логарифмическую доходность
    pub fn log_return(&self) -> f64 {
        (self.close / self.open).ln()
    }

    /// Рассчитывает диапазон свечи
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Рассчитывает относительный диапазон
    pub fn relative_range(&self) -> f64 {
        if self.open > 0.0 {
            self.range() / self.open
        } else {
            0.0
        }
    }
}

/// Уровень в книге ордеров
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Книга ордеров
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: u64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Возвращает лучшую цену покупки
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Возвращает лучшую цену продажи
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Рассчитывает спред
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Рассчитывает относительный спред
    pub fn spread_percent(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) if bid > 0.0 => Some((ask - bid) / bid * 100.0),
            _ => None,
        }
    }

    /// Рассчитывает дисбаланс книги ордеров
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_vol: f64 = self.bids.iter().take(depth).map(|l| l.quantity).sum();
        let ask_vol: f64 = self.asks.iter().take(depth).map(|l| l.quantity).sum();
        let total = bid_vol + ask_vol;

        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }
}

/// Информация о тикере
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub high_price_24h: f64,
    pub low_price_24h: f64,
    pub volume_24h: f64,
    pub turnover_24h: f64,
    pub price_change_percent_24h: f64,
    pub funding_rate: Option<f64>,
    pub next_funding_time: Option<u64>,
    pub open_interest: Option<f64>,
}

/// Результат запроса klines
#[derive(Debug, Deserialize)]
pub struct KlinesResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<serde_json::Value>>,
}

/// Результат запроса тикеров
#[derive(Debug, Deserialize)]
pub struct TickersResult {
    pub category: String,
    pub list: Vec<TickerData>,
}

#[derive(Debug, Deserialize)]
pub struct TickerData {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
    #[serde(rename = "fundingRate")]
    pub funding_rate: Option<String>,
    #[serde(rename = "nextFundingTime")]
    pub next_funding_time: Option<String>,
    #[serde(rename = "openInterest")]
    pub open_interest: Option<String>,
}

impl TickerData {
    pub fn to_ticker(&self) -> Result<Ticker, BybitError> {
        let parse = |s: &str| -> Result<f64, BybitError> {
            s.parse::<f64>()
                .map_err(|_| BybitError::ParseError(format!("Failed to parse: {}", s)))
        };

        Ok(Ticker {
            symbol: self.symbol.clone(),
            last_price: parse(&self.last_price)?,
            high_price_24h: parse(&self.high_price_24h)?,
            low_price_24h: parse(&self.low_price_24h)?,
            volume_24h: parse(&self.volume_24h)?,
            turnover_24h: parse(&self.turnover_24h)?,
            price_change_percent_24h: parse(&self.price_24h_pcnt)? * 100.0,
            funding_rate: self.funding_rate.as_ref().and_then(|s| s.parse().ok()),
            next_funding_time: self.next_funding_time.as_ref().and_then(|s| s.parse().ok()),
            open_interest: self.open_interest.as_ref().and_then(|s| s.parse().ok()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_log_return() {
        let kline = Kline {
            start_time: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        let expected = (105.0_f64 / 100.0).ln();
        assert!((kline.log_return() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_orderbook_spread() {
        let ob = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 0,
            bids: vec![
                OrderBookLevel { price: 99.0, quantity: 10.0 },
                OrderBookLevel { price: 98.0, quantity: 20.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, quantity: 10.0 },
                OrderBookLevel { price: 102.0, quantity: 20.0 },
            ],
        };

        assert_eq!(ob.best_bid(), Some(99.0));
        assert_eq!(ob.best_ask(), Some(101.0));
        assert_eq!(ob.spread(), Some(2.0));
    }
}
