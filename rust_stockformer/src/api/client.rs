//! HTTP клиент для Bybit API

use super::types::{ApiResponse, BybitError, Kline, KlinesResult, OrderBook, OrderBookLevel, Ticker, TickersResult};
use reqwest::Client;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Базовый URL для Bybit API v5
const BASE_URL: &str = "https://api.bybit.com";

/// Клиент для работы с Bybit API
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Создаёт новый клиент
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BASE_URL.to_string(),
        }
    }

    /// Создаёт клиент с кастомным URL (для тестирования)
    pub fn with_base_url(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    /// Получает исторические свечи
    ///
    /// # Arguments
    ///
    /// * `symbol` - Торговая пара (например, "BTCUSDT")
    /// * `interval` - Интервал свечей ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")
    /// * `limit` - Количество свечей (макс. 1000)
    /// * `start` - Начальное время (опционально, Unix timestamp в мс)
    /// * `end` - Конечное время (опционально, Unix timestamp в мс)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>, BybitError> {
        self.get_klines_with_time(symbol, interval, limit, None, None).await
    }

    /// Получает исторические свечи с указанием временного диапазона
    pub async fn get_klines_with_time(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
        start: Option<u64>,
        end: Option<u64>,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let mut params = vec![
            ("category", "linear".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
            ("limit", limit.min(1000).to_string()),
        ];

        if let Some(s) = start {
            params.push(("start", s.to_string()));
        }
        if let Some(e) = end {
            params.push(("end", e.to_string()));
        }

        debug!("Fetching klines for {} interval={} limit={}", symbol, interval, limit);

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let status = response.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(BybitError::RateLimitExceeded);
        }

        let api_response: ApiResponse<KlinesResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let result = api_response
            .result
            .ok_or_else(|| BybitError::ParseError("No result in response".to_string()))?;

        let mut klines: Vec<Kline> = result
            .list
            .iter()
            .map(|arr| Kline::from_bybit_array(arr))
            .collect::<Result<Vec<_>, _>>()?;

        // Bybit возвращает данные в обратном порядке (новые первые)
        klines.reverse();

        info!("Fetched {} klines for {}", klines.len(), symbol);

        Ok(klines)
    }

    /// Получает данные для нескольких символов параллельно
    pub async fn get_multi_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<Vec<(String, Vec<Kline>)>, BybitError> {
        let mut results = Vec::new();

        // Последовательно для избежания rate limits
        for symbol in symbols {
            let klines = self.get_klines(symbol, interval, limit).await?;
            results.push((symbol.to_string(), klines));

            // Небольшая задержка между запросами
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(results)
    }

    /// Получает текущую информацию о тикере
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let params = [
            ("category", "linear"),
            ("symbol", symbol),
        ];

        debug!("Fetching ticker for {}", symbol);

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let api_response: ApiResponse<TickersResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let result = api_response
            .result
            .ok_or_else(|| BybitError::ParseError("No result in response".to_string()))?;

        result
            .list
            .first()
            .ok_or_else(|| BybitError::InvalidSymbol(symbol.to_string()))?
            .to_ticker()
    }

    /// Получает тикеры для нескольких символов
    pub async fn get_tickers(&self, symbols: &[&str]) -> Result<Vec<Ticker>, BybitError> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let params = [("category", "linear")];

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let api_response: ApiResponse<TickersResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let result = api_response
            .result
            .ok_or_else(|| BybitError::ParseError("No result in response".to_string()))?;

        let symbol_set: std::collections::HashSet<_> = symbols.iter().collect();

        result
            .list
            .iter()
            .filter(|t| symbol_set.contains(&t.symbol.as_str()))
            .map(|t| t.to_ticker())
            .collect()
    }

    /// Получает книгу ордеров
    pub async fn get_orderbook(&self, symbol: &str, limit: usize) -> Result<OrderBook, BybitError> {
        let url = format!("{}/v5/market/orderbook", self.base_url);

        let params = [
            ("category", "linear"),
            ("symbol", symbol),
            ("limit", &limit.min(500).to_string()),
        ];

        debug!("Fetching orderbook for {}", symbol);

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let api_response: ApiResponse<serde_json::Value> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let result = api_response
            .result
            .ok_or_else(|| BybitError::ParseError("No result in response".to_string()))?;

        let parse_levels = |arr: &[serde_json::Value]| -> Vec<OrderBookLevel> {
            arr.iter()
                .filter_map(|level| {
                    let arr = level.as_array()?;
                    let price = arr.first()?.as_str()?.parse().ok()?;
                    let quantity = arr.get(1)?.as_str()?.parse().ok()?;
                    Some(OrderBookLevel { price, quantity })
                })
                .collect()
        };

        let bids = result["b"]
            .as_array()
            .map(|arr| parse_levels(arr))
            .unwrap_or_default();

        let asks = result["a"]
            .as_array()
            .map(|arr| parse_levels(arr))
            .unwrap_or_default();

        let timestamp = result["ts"]
            .as_u64()
            .or_else(|| result["ts"].as_str()?.parse().ok())
            .unwrap_or(0);

        Ok(OrderBook {
            symbol: symbol.to_string(),
            timestamp,
            bids,
            asks,
        })
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests - uncomment to run against real API
    /*
    #[tokio::test]
    async fn test_get_klines() {
        let client = BybitClient::new();
        let klines = client.get_klines("BTCUSDT", "60", 100).await.unwrap();

        assert!(!klines.is_empty());
        assert!(klines.len() <= 100);

        // Check data is valid
        for k in &klines {
            assert!(k.close > 0.0);
            assert!(k.volume >= 0.0);
        }
    }

    #[tokio::test]
    async fn test_get_ticker() {
        let client = BybitClient::new();
        let ticker = client.get_ticker("BTCUSDT").await.unwrap();

        assert_eq!(ticker.symbol, "BTCUSDT");
        assert!(ticker.last_price > 0.0);
    }
    */
}
