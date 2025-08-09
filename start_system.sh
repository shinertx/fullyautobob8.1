
#!/bin/bash
# v26meme Trading System Launcher
# Starts both the trading bot and web dashboard

set -e

echo "🚀 Starting v26meme Trading System"
echo "=================================="

# Check if virtual environment exists
# if [ ! -d ".venv" ]; then
#     echo "❌ Virtual environment not found. Please run setup first."
#     exit 1
# fi

# Activate virtual environment
# source .venv/bin/activate

# Check if database exists
if [ ! -f "v26meme.db" ]; then
    echo "📚 Creating new database..."
fi

if [ -f "institutional_dashboard.py" ]; then
  echo "🏛️ Starting Institutional Trading Dashboard on http://localhost:8080"
  nohup /usr/bin/python3 -u institutional_dashboard.py > institutional_dashboard.log 2>&1 &
  DASHBOARD_PID=$!
else
  echo "❌ Institutional dashboard not found (institutional_dashboard.py). Skipping dashboard startup."
  DASHBOARD_PID=""
fi

echo "⏱️  Waiting 3 seconds for dashboard to start..."
sleep 3

echo "🤖 Starting Trading Bot"
# The bot needs to start first to create the database schema
/usr/bin/python3 -u v26meme_full.py > v26meme_full.log 2>&1 &
BOT_PID=$!
echo "🤖 Bot started with PID: $BOT_PID"

echo "⏱️  Waiting 5 seconds for bot to initialize database..."
sleep 5

# Monitor the bot and restart if it crashes
(
  while true; do
      wait $BOT_PID
      echo "⚠️ Bot process stopped. Restarting in 5 seconds..."
      sleep 5
      /usr/bin/python3 -u v26meme_full.py > v26meme_full.log 2>&1 &
      BOT_PID=$!
      echo "🤖 Bot restarted with PID: $BOT_PID"
  done
) &
BOT_LOOP_PID=$!

echo ""
echo "✅ System started successfully!"
echo "🏛️ Institutional Dashboard: http://localhost:8080"
echo "🤖 Bot Loop Process ID: $BOT_LOOP_PID"
if [ -n "$DASHBOARD_PID" ]; then
  echo "🌐 Dashboard Process ID: $DASHBOARD_PID"
fi
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo "🛑 Stopping services..."; if [ -n "$DASHBOARD_PID" ]; then kill $BOT_LOOP_PID $DASHBOARD_PID 2>/dev/null; else kill $BOT_LOOP_PID 2>/dev/null; fi; exit 0' INT
wait
