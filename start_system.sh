#!/bin/bash
# v26meme Trading System Launcher
# Starts both the trading bot and web dashboard

set -e

echo "🚀 Starting v26meme Trading System"
echo "=================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if database exists
if [ ! -f "v26meme.db" ]; then
    echo "📚 Creating new database..."
fi

if [ -f "web_dashboard.py" ]; then
  echo "🌐 Starting Web Dashboard on http://localhost:8000"
  python web_dashboard.py &
  DASHBOARD_PID=$!
else
  echo "ℹ️ Dashboard not found (web_dashboard.py). Skipping dashboard startup."
  DASHBOARD_PID=""
fi

echo "⏱️  Waiting 3 seconds for dashboard to start..."
sleep 3

echo "🤖 Starting Trading Bot"
export TRADING_MODE=${TRADING_MODE:-PAPER}
# Add a loop to keep the bot running
while true; do
    echo "🔄 Starting/Restarting trading bot..."
    python v26meme_full.py > v26meme_full.log 2>&1 &
    BOT_PID=$!
    echo "🤖 Bot started with PID: $BOT_PID"
    wait $BOT_PID
    echo "⚠️ Bot process stopped. Restarting in 5 seconds..."
    sleep 5
done &
BOT_LOOP_PID=$!

echo ""
echo "✅ System started successfully!"
echo "📊 Dashboard: http://localhost:8000"
echo "🤖 Bot Loop Process ID: $BOT_LOOP_PID"
if [ -n "$DASHBOARD_PID" ]; then
  echo "🌐 Dashboard Process ID: $DASHBOARD_PID"
fi
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo "🛑 Stopping services..."; if [ -n "$DASHBOARD_PID" ]; then kill $BOT_LOOP_PID $DASHBOARD_PID 2>/dev/null; else kill $BOT_LOOP_PID 2>/dev/null; fi; exit 0' INT
wait
