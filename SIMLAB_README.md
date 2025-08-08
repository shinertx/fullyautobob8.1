# 🧪 SimLab: Parallel Simulation Engine

**Background, always-on simulation & replay engine for v26meme**

## 🎯 What It Does

- **Parallel Replay**: Runs historical data replays in background without touching live trading
- **Strategy Testing**: Evaluates all current strategies on OHLCV data
- **Zero Risk**: No side effects on live/paper trading operations
- **Automated Discovery**: Auto-discovers data files and queues simulations
- **Performance Metrics**: Calculates win rates, Sharpe ratios, drawdowns

## 🚀 Key Features

### ⚡ **Always-On Operation**
- Runs continuously in background
- Auto-discovers new data files every 5 minutes
- Parallel execution with configurable concurrency

### 📊 **Comprehensive Analysis**
- Tests all active strategies simultaneously
- Tracks P&L, win rates, Sharpe ratios
- Monitors stop-loss and take-profit hits
- Calculates maximum drawdown

### 🔌 **Seamless Integration**
- Uses existing strategy sandbox execution
- Reuses current strategy registry
- No changes to live trading code paths
- Results stored in separate database tables

## 📁 File Structure

```
├── simlab.py           # Main simulation engine
├── data/               # Historical data files
│   ├── BTC_USDC_5m.csv
│   ├── ETH_USDT_15m.csv
│   └── ...
└── tools/
    ├── sim_enqueue.py  # Manual simulation utility
    └── data_fetcher.py # Automatic data download tool
├── fetch_data.sh       # Quick data fetch script
```

## 📈 Data Format

Files should be named: `{BASE}_{QUOTE}_{TIMEFRAME}.csv`

**Example**: `BTC_USDC_5m.csv`, `ETH_USDT_15m.parquet`

**Required columns** (case-insensitive):
```csv
timestamp,open,high,low,close,volume
```

- **timestamp**: Unix timestamp (ms or seconds)
- **open,high,low,close**: Price data
- **volume**: Trading volume

## ⚙️ Configuration

Environment variables:

```bash
# Enable/disable SimLab
SIMLAB_ENABLED=true

# Data directory
SIMLAB_DATA_DIR=data

# Parallel simulation limit
SIMLAB_CONCURRENCY=2

# Default exchange name for simulations
SIM_DEFAULT_EXCHANGE=sim
```

## 🔧 Database Schema

SimLab creates three new tables:

### `sim_runs`
- Simulation metadata and status
- Run parameters and timing

### `sim_trades`
- Individual trade records per strategy
- Entry/exit prices and P&L

### `sim_metrics`
- Aggregated performance metrics
- Win rates, Sharpe ratios, drawdowns

## 📊 Usage Examples

### Automatic Data Fetching (Recommended)
Get tons of historical data automatically:

```bash
# Quick fetch: 30 days of data for 20+ symbols
./fetch_data.sh

# Custom fetch: specific symbols and timeframes
python3 tools/data_fetcher.py --symbols BTC/USD ETH/USD SOL/USD --timeframes 5m 1h --days 60

# Bulk fetch: everything available (100+ files)
python3 tools/data_fetcher.py --days 90
```

**Data Sources** (no API keys needed):
- **Coinbase Pro**: Primary source, most reliable
- **Kraken**: Backup source, good coverage  
- **CoinGecko**: Daily data fallback

### Automatic Operation
Just drop data files in `./data/` and SimLab will auto-discover them:

```bash
# Add data file manually
cp historical_data.csv data/BTC_USDC_5m.csv

# SimLab automatically detects and queues simulation
# Check logs for: "🧪 SimLab enqueued BTC_USDC_5m"
```

### Manual Enqueueing
```bash
python tools/sim_enqueue.py
```

### Check Results
```bash
python3 -c "
import sqlite3, pandas as pd
con = sqlite3.connect('v26meme.db')
print('=== Latest Simulation Runs ===')
print(pd.read_sql_query('SELECT * FROM sim_runs ORDER BY created_at DESC LIMIT 5', con))
print('\n=== Top Strategy Performance ===')
print(pd.read_sql_query('SELECT * FROM sim_metrics ORDER BY sharpe_like DESC LIMIT 5', con))
con.close()
"
```

## 🎯 Integration with Strategy Evolution

The simulation results feed back into the main system's:

1. **Pattern Discovery**: Validates patterns across historical data
2. **Strategy Evolution**: Identifies high-performing parameter combinations
3. **Risk Management**: Tests strategies under various market conditions

## 🔄 Workflow

1. **Auto-Discovery**: SimLab scans `data/` directory every 5 minutes
2. **Queue Management**: New files are automatically queued for simulation
3. **Parallel Execution**: Multiple simulations run concurrently
4. **Result Storage**: Trades and metrics saved to database
5. **Feedback Loop**: Results inform strategy evolution (future enhancement)

## 🛡️ Safety Features

- **Isolated Execution**: Separate state from live trading
- **Error Handling**: Robust error recovery and logging
- **Resource Limits**: Configurable concurrency limits
- **Zero Side Effects**: Cannot affect live positions or equity

## 🚀 Future Enhancements

- **Synthetic Scenarios**: Generate artificial market conditions
- **Multi-Timeframe**: Cross-timeframe strategy validation
- **Performance Optimization**: Handle larger datasets efficiently
- **Automatic Feedback**: Direct integration with strategy promotion/demotion
- **Real-time Data**: Live data streaming integration
- **Advanced Fetching**: Options data, social sentiment, on-chain metrics

## 🔧 Data Fetcher Features

The built-in data fetcher (`tools/data_fetcher.py`) provides:

- **Multi-Source**: Automatically tries Coinbase → Kraken → CoinGecko
- **Rate Limited**: Respects API limits, won't get blocked
- **Bulk Download**: Can fetch 100+ files in one command
- **Smart Retry**: Falls back to different exchanges if one fails
- **No API Keys**: Uses free public endpoints
- **Multiple Timeframes**: 5m, 15m, 1h, 4h, 1d data
- **Popular Symbols**: BTC, ETH, SOL, ADA, MATIC, DOGE, and more

**Example Bulk Fetch Output**:
```
🚀 Starting bulk data fetch (30 days back)...
✅ Fetched 8640 candles for BTC/USD 5m from Coinbase
✅ Fetched 720 candles for ETH/USD 1h from Coinbase
💾 Saved 8640 rows to data/BTC_USD_5m.csv
💾 Saved 720 rows to data/ETH_USD_1h.csv
...
📁 147 data files in data/
```

---

**SimLab enables your trading system to learn from 1000+ lifetimes of market data while safely trading live markets.**
