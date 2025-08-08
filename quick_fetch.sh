#!/bin/bash
# Quick start script for background data fetching
# This is the SAFE way to fetch massive data and take your laptop

echo "🚀 v26meme Background Data Fetcher"
echo "=================================="
echo ""
echo "This will download 1000+ files in the background"
echo "✅ Safe to close laptop and travel"
echo "✅ Uses optimized Parquet format (90% smaller)"
echo "✅ Graceful shutdown on CTRL+C"
echo ""

read -p "Start background fetch? (y/N): " choice
case "$choice" in 
  y|Y ) 
    echo "🔥 Starting background data fetch..."
    echo "📋 Log: tail -f data_fetch_background.log"
    echo "🛑 Stop: pkill -f data_fetcher_v2"
    echo ""
    
    cd "$(dirname "$0")"
    nohup python3 tools/data_fetcher_v2.py --background --max-files 1000 > data_fetch_background.log 2>&1 &
    
    PID=$!
    echo "✅ Started background fetch (PID: $PID)"
    echo ""
    echo "Safe to close terminal and take laptop! ✈️"
    echo ""
    echo "Progress check:"
    echo "  tail -f data_fetch_background.log"
    echo ""
    echo "To stop:"
    echo "  kill $PID"
    ;;
  * ) 
    echo "Cancelled."
    ;;
esac
