#!/bin/bash
# Background data fetcher for SimLab
# Can run this and disconnect/close laptop

echo "🚀 Starting background data fetch for SimLab..."
echo "💻 This will run safely in the background"
echo "📊 Check progress: tail -f data_fetch_background.log"
echo "🛑 Stop anytime: pkill -f data_fetcher_v2"

cd "$(dirname "$0")"
nohup python3 tools/data_fetcher_v2.py --background --max-files 1000 > data_fetch_background.log 2>&1 &

PID=$!
echo "🔥 Background fetch started (PID: $PID)"
echo "📋 Log file: data_fetch_background.log"
echo "🛑 To stop: kill $PID"
echo ""
echo "Safe to close terminal and take laptop! ✈️"
