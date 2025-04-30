//! Модуль с архитектурой Stockformer
//!
//! Содержит реализацию основных компонентов:
//! - Token Embedding (1D-CNN)
//! - Cross-Ticker Attention
//! - ProbSparse Attention
//! - Encoder Stack
//! - Prediction Head

mod attention;
mod embedding;
mod stockformer;
mod config;

pub use attention::{CrossTickerAttention, ProbSparseAttention, AttentionWeights};
pub use embedding::TokenEmbedding;
pub use stockformer::StockformerModel;
pub use config::{StockformerConfig, AttentionType, OutputType};
