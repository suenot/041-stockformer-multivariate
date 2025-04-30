# Глава 43: Stockformer — Мультивариантное предсказание с кросс-активным вниманием

Эта глава посвящена **Stockformer** — архитектуре на основе Transformer, разработанной для мультивариантного прогнозирования цен. В отличие от традиционных моделей одномерного прогнозирования, Stockformer использует механизмы внимания для захвата связей между активами и временных зависимостей одновременно.

<p align="center">
<img src="https://i.imgur.com/XqZ8k2P.png" width="70%">
</p>

## Содержание

1. [Введение в Stockformer](#введение-в-stockformer)
    * [Почему мультивариантное предсказание?](#почему-мультивариантное-предсказание)
    * [Ключевые преимущества](#ключевые-преимущества)
    * [Сравнение с другими моделями](#сравнение-с-другими-моделями)
2. [Архитектура Stockformer](#архитектура-stockformer)
    * [Слой Token Embedding](#слой-token-embedding)
    * [Кросс-тикерное внимание](#кросс-тикерное-внимание)
    * [ProbSparse Attention](#probsparse-attention)
    * [Голова предсказания](#голова-предсказания)
3. [Представление мультивариантных данных](#представление-мультивариантных-данных)
    * [Логарифмическое процентное изменение](#логарифмическое-процентное-изменение)
    * [Мульти-тикерные входы](#мульти-тикерные-входы)
    * [Инженерия признаков](#инженерия-признаков)
4. [Практические примеры](#практические-примеры)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Лучшие практики](#лучшие-практики)
8. [Ресурсы](#ресурсы)

## Введение в Stockformer

Stockformer — это модифицированная архитектура Transformer, специально разработанная для прогнозирования финансовых временных рядов. Вместо того чтобы рассматривать предсказание цен как простую задачу одномерной авторегрессии, Stockformer моделирует задачу как **проблему мультивариантного прогнозирования**, используя механизмы внимания для обнаружения связей между различными финансовыми инструментами.

### Почему мультивариантное предсказание?

Традиционные модели предсказывают каждый актив независимо:

```
BTCUSDT → Модель → BTCUSDT_прогноз
ETHUSDT → Модель → ETHUSDT_прогноз
```

Stockformer использует кросс-активную информацию:

```
[BTCUSDT, ETHUSDT, SOLUSDT, ...] → Stockformer → BTCUSDT_прогноз
                                                  (учитывая все связи)
```

**Ключевая идея**: Финансовые рынки взаимосвязаны. Движение Bitcoin влияет на Ethereum, цены на нефть влияют на акции авиакомпаний, а технологические акции часто движутся вместе. Stockformer явно моделирует эти зависимости.

### Ключевые преимущества

1. **Моделирование кросс-активных связей**
   - Захватывает корреляции между различными активами
   - Использует причинность по Гренджеру для выявления предсказательных связей
   - Веса внимания показывают, какие активы влияют на прогнозы

2. **Эффективные механизмы внимания**
   - ProbSparse attention снижает сложность с O(L²) до O(L·log(L))
   - Self-attention distilling для эффективности памяти
   - Эффективная обработка длинных последовательностей

3. **Гибкие типы выхода**
   - Регрессия цены (MSE/MAE loss)
   - Предсказание направления (бинарные сигналы)
   - Аллокация портфеля (позиции ограниченные tanh)

4. **Интерпретируемые предсказания**
   - Веса внимания раскрывают кросс-активные зависимости
   - Понятная визуализация того, какие активы важны для каждого прогноза

### Сравнение с другими моделями

| Функция | LSTM | Transformer | TFT | Stockformer |
|---------|------|-------------|-----|-------------|
| Кросс-активное моделирование | ✗ | ✗ | Ограниченно | ✓ |
| ProbSparse attention | ✗ | ✗ | ✗ | ✓ |
| Мультивариантный вход | ✗ | ✗ | ✓ | ✓ |
| Учёт корреляций | ✗ | ✗ | ✗ | ✓ |
| Аллокация портфеля | ✗ | ✗ | ✗ | ✓ |

## Архитектура Stockformer

```
┌──────────────────────────────────────────────────────────────────────┐
│                         STOCKFORMER                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ Актив 1 │  │ Актив 2 │  │ Актив 3 │  │ Актив N │                 │
│  │ (BTC)   │  │ (ETH)   │  │ (SOL)   │  │  (...)  │                 │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                 │
│       │            │            │            │                        │
│       ▼            ▼            ▼            ▼                        │
│  ┌──────────────────────────────────────────────────┐                │
│  │           Token Embedding (1D-CNN)                │                │
│  │    Извлекает временные паттерны по активам        │                │
│  └───────────────────────┬──────────────────────────┘                │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────┐                │
│  │         Кросс-тикерное Self-Attention             │                │
│  │    Моделирует связи между всеми активами          │                │
│  │    (какие активы предсказывают какие?)            │                │
│  └───────────────────────┬──────────────────────────┘                │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────┐                │
│  │   Стек энкодеров (ProbSparse или Full Attention)  │                │
│  │         + Self-Attention Distilling               │                │
│  └───────────────────────┬──────────────────────────┘                │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────┐                │
│  │              Голова предсказания                  │                │
│  │    • Регрессия цены (MSE/MAE)                     │                │
│  │    • Сигнал направления (бинарный)                │                │
│  │    • Аллокация портфеля (tanh)                    │                │
│  └──────────────────────────────────────────────────┘                │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Слой Token Embedding

Stockformer использует эмбеддинги на основе 1D-CNN вместо простых линейных проекций:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3):
        super().__init__()
        # Отдельное ядро для каждого входного канала (актива)
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x):
        # x: [batch, seq_len, n_assets]
        x = x.permute(0, 2, 1)  # [batch, n_assets, seq_len]
        x = self.conv(x)        # [batch, d_model, seq_len]
        return x.permute(0, 2, 1)  # [batch, seq_len, d_model]
```

**Почему 1D-CNN?**
- Сохраняет временные связи при извлечении локальных паттернов
- Обучает разные ядра для каждого актива
- Более эффективен, чем позиционные полносвязные слои

### Кросс-тикерное внимание

Ключевая инновация: внимание одновременно по времени И активам:

```python
class CrossTickerAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_assets):
        super().__init__()
        self.n_assets = n_assets
        self.mha = nn.MultiheadAttention(d_model, n_heads)

    def forward(self, x):
        # x: [batch, seq_len, n_assets, d_model]
        batch, seq_len, n_assets, d_model = x.shape

        # Преобразуем для кросс-активного внимания
        # (seq_len * batch) как batch, n_assets как последовательность
        x = x.view(batch * seq_len, n_assets, d_model)

        # Self-attention между активами на каждом временном шаге
        attn_out, attn_weights = self.mha(x, x, x)

        # attn_weights показывает, какие активы влияют на какие
        return attn_out.view(batch, seq_len, n_assets, d_model), attn_weights
```

Веса внимания показывают **какие активы влияют на прогноз**:
- Высокое внимание от ETH к BTC → движения ETH помогают предсказать BTC
- Полезно для понимания рыночной динамики и построения портфелей

### ProbSparse Attention

Стандартный self-attention имеет сложность O(L²). ProbSparse attention снижает её до O(L·log(L)):

```python
class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.factor = factor  # Контролирует разреженность
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B, L, D = queries.shape

        # Выбираем top-u ключей на основе "остроты" запроса
        u = int(self.factor * np.log(L))
        u = min(u, L)

        # Вычисляем attention scores для выбранных запросов
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.value_proj(values)

        # Измеряем "остроту" запроса: max(QK^T) - mean(QK^T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(D)
        M = scores.max(-1)[0] - scores.mean(-1)

        # Выбираем top-u запросов с наибольшей остротой
        M_top = M.topk(u, sorted=False)[1]

        # Вычисляем разреженное внимание только для выбранных запросов
        Q_sampled = Q[torch.arange(B)[:, None], M_top]
        attn = torch.softmax(
            torch.matmul(Q_sampled, K.transpose(-2, -1)) / np.sqrt(D),
            dim=-1
        )

        # Агрегируем значения
        context = torch.matmul(attn, V)

        return context
```

**Интуиция**: Не все запросы требуют полного вычисления внимания. "Острые" запросы (те, у которых доминирует внимание к определённым ключам) наиболее важны.

### Голова предсказания

Stockformer поддерживает несколько типов выхода:

```python
class PredictionHead(nn.Module):
    def __init__(self, d_model, output_type='regression'):
        super().__init__()
        self.output_type = output_type

        if output_type == 'regression':
            # Прямое предсказание цены
            self.head = nn.Linear(d_model, 1)
            self.loss_fn = nn.MSELoss()

        elif output_type == 'direction':
            # Бинарная классификация вверх/вниз
            self.head = nn.Linear(d_model, 1)
            self.loss_fn = nn.BCEWithLogitsLoss()

        elif output_type == 'allocation':
            # Веса портфеля через tanh
            self.head = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Tanh()  # Ограничивает выход до [-1, 1]
            )
            self.loss_fn = lambda pred, ret: -torch.mean(pred * ret)

    def forward(self, x):
        return self.head(x)
```

## Представление мультивариантных данных

### Логарифмическое процентное изменение

Сырые цены преобразуются для стабильного обучения:

```python
def log_percent_change(close, open_price):
    """
    Преобразование цен в логарифмическое процентное изменение.

    Преимущества:
    - Стабилизация дисперсии для разных ценовых масштабов
    - BTC по $40,000 и ETH по $2,000 становятся сравнимыми
    - Стационарный ряд (важно для нейросетей)
    """
    return np.log(close / open_price + 1)
```

### Мульти-тикерные входы

Структура данных для мультивариантного предсказания:

```python
# Структура данных для Stockformer
# Форма: [batch, seq_len, n_assets, features]

data = {
    'prices': torch.tensor([
        # Временной шаг 1
        [[45000, 2500, 100],   # [BTC, ETH, SOL] цены закрытия
         [44800, 2480, 99]],   # Временной шаг 2
         # ...
    ]),
    'volumes': torch.tensor([...]),
    'returns': torch.tensor([...]),
}
```

### Инженерия признаков

Рекомендуемые признаки для каждого актива:

| Признак | Описание | Расчёт |
|---------|----------|--------|
| `log_return` | Логарифмическое изменение цены | ln(close/prev_close) |
| `volume_change` | Относительный объём | vol/vol_ma_20 |
| `volatility` | Скользящая волатильность | std(returns, 20) |
| `rsi` | Индекс относительной силы | Стандартный расчёт RSI |
| `correlation` | Парная корреляция | rolling_corr(asset_i, asset_j) |
| `funding_rate` | Ставка фандинга | Из API биржи |
| `open_interest` | Открытый интерес | Из API биржи |

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Dict

def prepare_multivariate_data(
    symbols: List[str],
    lookback: int = 168,  # 7 дней часовых данных
    horizon: int = 24     # Прогноз на 24 часа
) -> Dict:
    """
    Подготовка данных для обучения Stockformer.

    Аргументы:
        symbols: Список торговых пар (например, ['BTCUSDT', 'ETHUSDT'])
        lookback: Количество исторических временных шагов
        horizon: Горизонт прогнозирования

    Возвращает:
        Словарь с X (признаки) и y (целевые значения)
    """

    all_data = []

    for symbol in symbols:
        # Загрузка данных с Bybit или другого источника
        df = load_bybit_data(symbol)

        # Расчёт признаков
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_change'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility'] = df['log_return'].rolling(20).std()

        all_data.append(df)

    # Выравнивание всех dataframe по временной метке
    aligned_data = pd.concat(all_data, axis=1, keys=symbols)

    # Создание последовательностей
    X, y = [], []
    for i in range(lookback, len(aligned_data) - horizon):
        X.append(aligned_data.iloc[i-lookback:i].values)
        y.append(aligned_data.iloc[i+horizon]['log_return'].values)

    return {
        'X': np.array(X),  # [n_samples, lookback, n_assets * n_features]
        'y': np.array(y),  # [n_samples, n_assets]
        'symbols': symbols
    }
```

### 02: Обучение модели

```python
# python/03_train_model.py

import torch
from stockformer import Stockformer

# Конфигурация модели
config = {
    'n_assets': 5,
    'd_model': 128,
    'n_heads': 8,
    'n_encoder_layers': 3,
    'dropout': 0.1,
    'attention_type': 'probsparse',
    'output_type': 'allocation'
}

# Инициализация модели
model = Stockformer(**config)

# Цикл обучения с планировщиком learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

for epoch in range(100):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        # Прямой проход
        predictions = model(batch_x)

        # Расчёт потерь (зависит от output_type)
        loss = model.compute_loss(predictions, batch_y)

        # Обратный проход
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Валидация
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)
```

### 03: Визуализация внимания

```python
# python/04_cross_asset_prediction.py

def visualize_attention(attention_weights, symbols):
    """
    Создание тепловой карты кросс-активного внимания.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Усреднение по головам и батчу
    avg_attention = attention_weights.mean(dim=[0, 1]).numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        avg_attention,
        xticklabels=symbols,
        yticklabels=symbols,
        annot=True,
        cmap='Blues'
    )
    plt.title('Веса кросс-активного внимания')
    plt.xlabel('Ключ (Источник)')
    plt.ylabel('Запрос (Цель)')
    plt.savefig('attention_heatmap.png')
```

## Реализация на Rust

См. [rust_stockformer](rust_stockformer/) для полной реализации на Rust с данными Bybit.

```
rust_stockformer/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Основные экспорты библиотеки
│   ├── api/                # Клиент Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент для Bybit
│   │   └── types.rs        # Типы ответов API
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Утилиты загрузки данных
│   │   ├── features.rs     # Инженерия признаков
│   │   └── dataset.rs      # Датасет для обучения
│   ├── model/              # Архитектура Stockformer
│   │   ├── mod.rs
│   │   ├── embedding.rs    # Слой token embedding
│   │   ├── attention.rs    # Кросс-тикерное и ProbSparse внимание
│   │   ├── encoder.rs      # Стек энкодеров
│   │   └── stockformer.rs  # Полная модель
│   └── strategy/           # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Движок бэктестинга
└── examples/
    ├── fetch_data.rs       # Загрузка данных Bybit
    ├── train.rs            # Обучение модели
    └── backtest.rs         # Запуск бэктеста
```

### Быстрый старт (Rust)

```bash
# Перейти в проект Rust
cd rust_stockformer

# Загрузить данные с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Обучить модель
cargo run --example train -- --epochs 100 --batch-size 32

# Запустить бэктест
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

См. [python/](python/) для реализации на Python.

### Быстрый старт (Python)

```bash
# Установить зависимости
pip install -r requirements.txt

# Загрузить данные
python data_loader.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Обучить модель
python train.py --config configs/default.yaml

# Запустить бэктест
python backtest.py --model checkpoints/best_model.pt
```

## Лучшие практики

### Когда использовать Stockformer

**Хорошие случаи использования:**
- Торговля коррелированными активами (крипто, секторные ETF)
- Аллокация портфеля по нескольким активам
- Обнаружение кросс-активных зависимостей
- Долгосрочные прогнозы (часы-дни)

**Не идеально для:**
- Высокочастотная торговля (задержка инференса)
- Предсказание одного актива (используйте более простые модели)
- Очень маленькие датасеты (<1000 примеров)

### Рекомендации по гиперпараметрам

| Параметр | Рекомендуется | Примечания |
|----------|---------------|------------|
| `d_model` | 128-256 | Согласовать с n_heads |
| `n_heads` | 8 | Должен делить d_model |
| `n_assets` | 5-20 | Больше требует больше данных |
| `lookback` | 168 (7 дней часовых) | Зависит от частоты данных |
| `dropout` | 0.1-0.2 | Выше для маленьких датасетов |

### Частые ошибки

1. **Нестабильность градиента**: Используйте клиппинг градиентов и планирование learning rate
2. **Переобучение**: Применяйте dropout, используйте раннюю остановку
3. **Утечка данных**: Обеспечьте правильное разделение train/val/test
4. **Коллапс корреляций**: Мониторьте веса внимания на разнообразие

## Ресурсы

### Статьи

- [Transformer Based Time-Series Forecasting For Stock](https://arxiv.org/abs/2502.09625) — Оригинальная статья Stockformer
- [Stockformer: A Price-Volume Factor Stock Selection Model](https://arxiv.org/abs/2401.06139) — Продвинутый вариант с wavelet transform
- [MASTER: Market-Guided Stock Transformer](https://arxiv.org/abs/2312.15235) — Связанная архитектура
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Оригинальный Transformer

### Реализации

- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — Библиотека временных рядов
- [Informer](https://github.com/zhouhaoyi/Informer2020) — Реализация ProbSparse attention
- [Autoformer](https://github.com/thuml/Autoformer) — Связанная архитектура

### Связанные главы

- [Глава 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Многогоризонтное прогнозирование
- [Глава 41: Higher Order Transformers](../41_higher_order_transformers) — Продвинутые механизмы внимания
- [Глава 47: Cross-Attention Multi-Asset](../47_cross_attention_multi_asset) — Кросс-активное моделирование

---

## Уровень сложности

**Продвинутый**

Необходимые знания:
- Архитектура Transformer и механизмы внимания
- Основы прогнозирования временных рядов
- Базовые концепции портфельной теории
- Библиотеки ML для PyTorch/Rust
