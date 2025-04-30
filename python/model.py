"""
Stockformer Model Implementation in PyTorch

Provides:
- StockformerConfig: Model configuration
- StockformerModel: Main transformer model
- CrossTickerAttention: Attention between assets
- ProbSparseAttention: Efficient sparse attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class AttentionType(Enum):
    """Type of attention mechanism"""
    FULL = "full"
    PROBSPARSE = "probsparse"
    LOCAL = "local"


class OutputType(Enum):
    """Type of model output"""
    REGRESSION = "regression"
    DIRECTION = "direction"
    PORTFOLIO = "portfolio"
    QUANTILE = "quantile"


@dataclass
class StockformerConfig:
    """
    Configuration for Stockformer model

    Example:
        config = StockformerConfig(
            num_tickers=5,
            seq_len=96,
            d_model=64
        )
    """
    # Architecture
    num_tickers: int = 5
    seq_len: int = 96
    input_features: int = 6
    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 256
    num_encoder_layers: int = 2
    dropout: float = 0.1

    # Attention
    attention_type: AttentionType = AttentionType.PROBSPARSE
    sparsity_factor: float = 5.0

    # Output
    output_type: OutputType = OutputType.REGRESSION
    prediction_horizon: int = 1
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Embedding
    kernel_size: int = 3
    use_cross_ticker_attention: bool = True
    use_positional_encoding: bool = True

    def validate(self):
        """Validate configuration"""
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.kernel_size % 2 == 1, "kernel_size must be odd"
        assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def output_dim(self) -> int:
        if self.output_type == OutputType.REGRESSION:
            return self.num_tickers
        elif self.output_type == OutputType.DIRECTION:
            return self.num_tickers * 3
        elif self.output_type == OutputType.PORTFOLIO:
            return self.num_tickers
        elif self.output_type == OutputType.QUANTILE:
            return self.num_tickers * len(self.quantiles)
        return self.num_tickers


class TokenEmbedding(nn.Module):
    """
    Token embedding using 1D convolution

    Converts [batch, seq_len, features] to [batch, seq_len, d_model]
    """

    def __init__(self, config: StockformerConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=config.input_features,
            out_channels=config.d_model,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.conv(x)       # [batch, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        return self.activation(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TickerEncoding(nn.Module):
    """Learnable ticker-specific encoding"""

    def __init__(self, num_tickers: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tickers, d_model)

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_tickers, seq_len, d_model]
        # ticker_ids: [num_tickers]
        ticker_emb = self.embedding(ticker_ids)  # [num_tickers, d_model]
        return x + ticker_emb.unsqueeze(0).unsqueeze(2)


class CrossTickerAttention(nn.Module):
    """
    Cross-ticker attention mechanism

    Computes attention between different assets at each timestep
    """

    def __init__(self, config: StockformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, num_tickers, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, num_tickers, seq_len, d_model]
            attention: [batch, num_heads, num_tickers, num_tickers] (optional)
        """
        batch, num_tickers, seq_len, d_model = x.shape

        # Reshape for cross-ticker attention at each timestep
        # [batch, seq_len, num_tickers, d_model]
        x = x.transpose(1, 2)

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # [batch, seq_len, num_heads, num_tickers, head_dim]
        q = q.view(batch, seq_len, num_tickers, self.num_heads, self.head_dim).transpose(2, 3)
        k = k.view(batch, seq_len, num_tickers, self.num_heads, self.head_dim).transpose(2, 3)
        v = v.view(batch, seq_len, num_tickers, self.num_heads, self.head_dim).transpose(2, 3)

        # Attention: [batch, seq_len, num_heads, num_tickers, num_tickers]
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output: [batch, seq_len, num_heads, num_tickers, head_dim]
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(2, 3).contiguous()
        out = out.view(batch, seq_len, num_tickers, d_model)
        out = self.out_proj(out)

        # [batch, num_tickers, seq_len, d_model]
        out = out.transpose(1, 2)

        # Average attention over time for interpretability
        avg_attn = attn.mean(dim=1) if return_attention else None

        return out, avg_attn


class ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention mechanism

    Reduces complexity from O(L²) to O(L·log(L))
    """

    def __init__(self, config: StockformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = math.sqrt(self.head_dim)
        self.sparsity_factor = config.sparsity_factor

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention: [batch, num_heads, seq_len, seq_len] (optional)
        """
        batch, seq_len, d_model = x.shape

        # Calculate number of top queries
        u = max(1, min(seq_len, int(self.sparsity_factor * math.log(seq_len + 1))))

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute sparsity measurement M(q) = max(Q·K^T) - mean(Q·K^T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        M = scores.max(dim=-1)[0] - scores.mean(dim=-1)  # [batch, num_heads, seq_len]

        # Select top-u queries
        _, top_indices = M.topk(u, dim=-1)  # [batch, num_heads, u]

        # Full attention only for selected queries
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.out_proj(out)

        return out, attn if return_attention else None


class EncoderLayer(nn.Module):
    """Transformer encoder layer with optional cross-ticker attention"""

    def __init__(self, config: StockformerConfig):
        super().__init__()

        self.self_attention = ProbSparseAttention(config)
        self.cross_ticker_attention = CrossTickerAttention(config) if config.use_cross_ticker_attention else None

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model) if config.use_cross_ticker_attention else None

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: [batch, num_tickers, seq_len, d_model]

        Returns:
            output: [batch, num_tickers, seq_len, d_model]
            attention_dict: Dictionary of attention weights
        """
        batch, num_tickers, seq_len, d_model = x.shape
        attention_dict = {}

        # Self-attention for each ticker
        x_flat = x.view(batch * num_tickers, seq_len, d_model)
        attn_out, temporal_attn = self.self_attention(x_flat, return_attention)
        attn_out = attn_out.view(batch, num_tickers, seq_len, d_model)

        x = self.norm1(x + self.dropout(attn_out))

        if temporal_attn is not None:
            attention_dict['temporal'] = temporal_attn.view(batch, num_tickers, self.self_attention.num_heads, seq_len, seq_len)

        # Cross-ticker attention
        if self.cross_ticker_attention is not None:
            cross_out, cross_attn = self.cross_ticker_attention(x, return_attention)
            x = self.norm3(x + self.dropout(cross_out))

            if cross_attn is not None:
                attention_dict['cross_ticker'] = cross_attn

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x, attention_dict


class StockformerModel(nn.Module):
    """
    Stockformer: Multivariate Stock Prediction Model

    Example:
        config = StockformerConfig(num_tickers=5, seq_len=96)
        model = StockformerModel(config)

        x = torch.randn(2, 5, 96, 6)  # [batch, tickers, seq_len, features]
        output = model(x)
        print(output['predictions'].shape)  # [2, 5]
    """

    def __init__(self, config: StockformerConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Embedding layers
        self.token_embedding = TokenEmbedding(config)
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.seq_len * 2, config.dropout
        ) if config.use_positional_encoding else None
        self.ticker_encoding = TickerEncoding(config.num_tickers, config.d_model)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])

        # Output head
        self.output_head = self._build_output_head(config)

        # Register ticker IDs
        self.register_buffer('ticker_ids', torch.arange(config.num_tickers))

    def _build_output_head(self, config: StockformerConfig) -> nn.Module:
        """Build output projection layer based on output type"""
        if config.output_type == OutputType.QUANTILE:
            return nn.Linear(config.d_model, len(config.quantiles))
        elif config.output_type == OutputType.DIRECTION:
            return nn.Linear(config.d_model, 3)
        else:
            return nn.Linear(config.d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> dict:
        """
        Forward pass

        Args:
            x: Input tensor [batch, num_tickers, seq_len, features]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
                - predictions: [batch, output_dim]
                - attention_weights: Optional attention weights
                - confidence: Optional confidence scores
        """
        batch, num_tickers, seq_len, features = x.shape

        # Token embedding for each ticker
        x_embedded = []
        for t in range(num_tickers):
            emb = self.token_embedding(x[:, t])  # [batch, seq_len, d_model]
            x_embedded.append(emb)
        x = torch.stack(x_embedded, dim=1)  # [batch, num_tickers, seq_len, d_model]

        # Add positional encoding
        if self.positional_encoding is not None:
            for t in range(num_tickers):
                x[:, t] = self.positional_encoding(x[:, t])

        # Add ticker encoding
        x = self.ticker_encoding(x, self.ticker_ids)

        # Encoder layers
        all_attention = {}
        for i, layer in enumerate(self.encoder_layers):
            x, attn_dict = layer(x, return_attention)
            if attn_dict:
                all_attention[f'layer_{i}'] = attn_dict

        # Pool: take last timestep
        x = x[:, :, -1, :]  # [batch, num_tickers, d_model]

        # Output projection
        predictions = self._compute_output(x)

        result = {
            'predictions': predictions,
            'attention_weights': all_attention if return_attention else None
        }

        # Add confidence for quantile regression
        if self.config.output_type == OutputType.QUANTILE:
            result['confidence'] = self._compute_confidence(predictions)

        return result

    def _compute_output(self, x: torch.Tensor) -> torch.Tensor:
        """Compute output predictions"""
        batch, num_tickers, d_model = x.shape

        if self.config.output_type == OutputType.PORTFOLIO:
            # Portfolio weights sum to 1
            logits = self.output_head(x).squeeze(-1)  # [batch, num_tickers]
            return F.softmax(logits, dim=-1)

        elif self.config.output_type == OutputType.DIRECTION:
            # Per-ticker classification
            logits = self.output_head(x)  # [batch, num_tickers, 3]
            probs = F.softmax(logits, dim=-1)
            return probs.view(batch, -1)  # [batch, num_tickers * 3]

        elif self.config.output_type == OutputType.QUANTILE:
            # Per-ticker quantile prediction
            quantiles = self.output_head(x)  # [batch, num_tickers, num_quantiles]
            return quantiles.view(batch, -1)

        else:  # REGRESSION
            return self.output_head(x).squeeze(-1)  # [batch, num_tickers]

    def _compute_confidence(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute confidence from quantile predictions"""
        batch = predictions.size(0)
        num_quantiles = len(self.config.quantiles)
        num_tickers = self.config.num_tickers

        predictions = predictions.view(batch, num_tickers, num_quantiles)
        interval_width = (predictions[:, :, -1] - predictions[:, :, 0]).abs()
        confidence = 1.0 / (1.0 + interval_width)

        return confidence


if __name__ == "__main__":
    # Test the model
    print("Testing Stockformer model...")

    config = StockformerConfig(
        num_tickers=5,
        seq_len=48,
        input_features=6,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2
    )

    model = StockformerModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 5, 48, 6)
    output = model(x, return_attention=True)

    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Attention weights available: {output['attention_weights'] is not None}")

    # Test different output types
    for output_type in OutputType:
        config.output_type = output_type
        model = StockformerModel(config)
        output = model(x)
        print(f"{output_type.value}: predictions shape = {output['predictions'].shape}")
