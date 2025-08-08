#!/bin/bash
# Multiple data fetch options for SimLab

echo "🚀 SimLab Data Fetcher Options"
echo "==============================="
echo ""
echo "Choose your data amount:"
echo "1) 🔥 MASSIVE (1 year, 1000+ files) - RECOMMENDED"
echo "2) 💪 HUGE (6 months, 500+ files)"
echo "3) 📊 LARGE (3 months, 300+ files)" 
echo "4) 📈 MEDIUM (1 month, 100+ files)"
echo "5) 🧪 CUSTOM (specify your own)"
echo ""

read -p "Select option (1-5): " choice

cd /home/benjaminjones/fullyautobob8.1

case $choice in
    1)
        echo "🔥 MASSIVE FETCH: 1 year, 100+ symbols, 14 timeframes each"
        python3 tools/data_fetcher.py --days 365 --max-files 1500
        ;;
    2)
        echo "💪 HUGE FETCH: 6 months, 100+ symbols, 14 timeframes each"
        python3 tools/data_fetcher.py --days 180 --max-files 1000
        ;;
    3)
        echo "📊 LARGE FETCH: 3 months, 100+ symbols, 14 timeframes each"
        python3 tools/data_fetcher.py --days 90 --max-files 800
        ;;
    4)
        echo "📈 MEDIUM FETCH: 1 month, 50 symbols, 10 timeframes each"
        python3 tools/data_fetcher.py --days 30 --symbols BTC/USD ETH/USD SOL/USD ADA/USD MATIC/USD DOGE/USD LINK/USD UNI/USD AAVE/USD AVAX/USD --timeframes 5m 15m 1h 4h 1d
        ;;
    5)
        echo "🧪 CUSTOM FETCH:"
        read -p "Days of history: " days
        read -p "Max files: " maxfiles
        python3 tools/data_fetcher.py --days $days --max-files $maxfiles
        ;;
    *)
        echo "❌ Invalid option. Running default MASSIVE fetch..."
        python3 tools/data_fetcher.py --days 365 --max-files 1500
        ;;
esac

echo ""
echo "📊 Data files created:"
find data/ -name "*.csv" 2>/dev/null | wc -l
echo " CSV files in data/ directory"

echo ""
echo "🧪 SimLab will auto-discover ALL these files!"
echo "🚀 Start trading: python3 v26meme_full.py"
