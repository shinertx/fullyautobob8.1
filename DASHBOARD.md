# 🚀 v26meme Trading Dashboard

Professional real-time monitoring interface for the v26meme autonomous trading system.

## 🌟 Features

### 📊 **Real-Time Performance Monitoring**
- Live equity tracking with auto-refresh (5 second intervals)
- Current cash and position value
- Total and daily P&L tracking
- Win rate and performance metrics

### 💼 **Position Management**
- Real-time open positions view
- Entry and current prices
- P&L per position with color coding
- Position size and side indicators

### 🧬 **Strategy Analytics**
- Active strategies performance tracking
- Win rates and trade counts
- Total P&L per strategy
- Status indicators (Active/Paper/Micro)

### 💡 **Pattern Discovery**
- Latest discovered patterns
- Confidence scores
- Average returns
- Multi-symbol pattern support

## 🔧 Technical Details

- **Backend**: FastAPI for high performance
- **Database**: SQLite3 for reliable storage
- **Updates**: Real-time 5-second refresh
- **Security**: CORS enabled
- **API**: JSON endpoints available

## 📈 Getting Started

1. Start the dashboard:
```bash
python3 dashboard.py
```

2. Access the interface:
- **Local**: http://localhost:8080
- **Network**: http://YOUR_IP:8080 (for remote access)

## 🔗 API Endpoints

- **Main Dashboard**: `GET /`
- **Raw Data**: `GET /api/data`

## 💡 Tips

- Keep the dashboard open in a dedicated browser window
- Use dark mode for better visibility
- Monitor win rates for strategy evaluation
- Watch for high-confidence patterns

## ⚙️ Configuration

The dashboard uses these environment variables:
- `WEB_PORT`: Dashboard port (default: 8080)
- `LOG_LEVEL`: Logging detail (default: INFO)
