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

echo "🌐 Starting Web Dashboard on http://localhost:8000"
python web_dashboard.py &
DASHBOARD_PID=$!

echo "⏱️  Waiting 3 seconds for dashboard to start..."
sleep 3

echo "🤖 Starting Trading Bot"
python v26meme_full.py > v26meme_full.log 2>&1 &
BOT_PID=$!

echo ""
echo "✅ System started successfully!"
echo "📊 Dashboard: http://localhost:8000"
echo "🤖 Bot Process ID: $BOT_PID"
echo "🌐 Dashboard Process ID: $DASHBOARD_PID"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for interrupt
trap 'echo "🛑 Stopping services..."; kill $BOT_PID $DASHBOARD_PID 2>/dev/null; exit 0' INT
wait
