# 🏛️ v26meme Institutional Trading Dashboard

Comprehensive institutional-grade real-time monitoring and analytics platform
combining all dashboard features with SimLab integration for professional
quantitative fund operations.

## 🌟 Features

### 📊 **Institutional KPIs**
- **Portfolio NAV**: Live equity tracking with auto-refresh.
- **Sharpe Ratio**: Real-time risk-adjusted return metric.
- **Max Drawdown**: Key risk control indicator.
- **Daily P&L**: Intra-day performance measurement.
- **Win Rate**: Live trade success percentage.
- **Active Strategies**: Count of currently operating strategies.

### 🧪 **SimLab Integration**
- **Total Simulations**: See total number of historical backtests run.
- **Simulation Trades**: View total trades analyzed by the simulation engine.
- **Success Rate**: Monitor the completion rate of simulations.
- **Top Performers**: Identify the most promising strategies from simulations.

### 📈 **Advanced Charting & Risk**
- **Live Equity Curve**: Visualize portfolio growth over time.
- **Risk Metrics**: Track Value at Risk (VaR 95% & 99%) and Annualized Volatility.

### 💼 **Live Position Management**
- Real-time view of all open positions.
- P&L, side, entry price, and notional value for each position.
- Sortable and color-coded for quick analysis.

### 🧬 **Strategy Performance Analytics**
- In-depth table of all active strategies.
- Track trades, total P&L, win rate, and average P&L per trade.
- Monitor average confidence scores from the AI.

### � **Recent Activity**
- View the 50 most recent closed trades.
- Analyze execution details including P&L, duration, and associated strategy.

## 🔧 Technical Details

- **Backend**: FastAPI with WebSockets for high-performance, low-latency streaming.
- **Database**: Direct connection to `v26meme.db` for real-time data.
- **Updates**: Real-time 1-second refresh interval.
- **Design**: Responsive, multi-device layout suitable for professional monitoring.

## 📈 Getting Started

1.  The dashboard is started automatically as part of the main system script:
    ```bash
    ./start_system.sh
    ```

2.  Access the interface in your browser:
    - **Local**: http://localhost:8000
    - **Network**: http://YOUR_IP:8000 (for remote access)

## 🔗 API Endpoints

The institutional dashboard provides a comprehensive JSON API for programmatic access.

- **Main Dashboard HTML**: `GET /`
- **WebSocket Stream**: `WS /ws`
- **Institutional Overview**: `GET /api/institutional-overview`
- **Portfolio Analytics**: `GET /api/portfolio`
- **Risk Analytics**: `GET /api/risk`
- **SimLab Metrics**: `GET /api/simlab`
- **Live Positions**: `GET /api/positions`
- **Strategy Performance**: `GET /api/strategies`
- **Recent Trades**: `GET /api/trades`
- **Equity Curve Data**: `GET /api/equity-curve`
- **System Health**: `GET /health`

## ⚙️ Configuration

The dashboard uses these environment variables, typically set in a `.env` file:
- `DB_PATH`: Path to the SQLite database (default: `v26meme.db`)
- `WEB_PORT`: Dashboard port (default: 8000)
- `LOG_LEVEL`: Logging detail (default: INFO)
