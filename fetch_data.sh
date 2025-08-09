#!/bin/bash
# MASSIVE data fetch script for SimLab - GET EVERYTHING!

echo "🚀 SimLab MASSIVE Data Fetcher"
echo "==============================="
echo "🔥 Preparing to download 1000+ historical datasets!"
echo ""

cd /home/benjaminjones/fullyautobob8.1

echo "📥 Fetching 1 YEAR of historical data for 100+ symbols..."
echo "⏱️  14 different timeframes per symbol"
echo "🎯 Targeting 1000+ data files total"
echo "⚠️  This will take 10-30 minutes - be patient!"
echo ""

# Start the MASSIVE fetch
python3 tools/data_fetcher.py --days 365 --max-files 1500

echo ""
echo "📊 Data files created:"
find data/ -name "*.csv" | wc -l
echo " CSV files in data/ directory"

echo ""
echo "📈 Sample of what we got:"
ls data/*.csv | head -10

echo ""
echo "🧪 SimLab will auto-discover ALL these files!"
echo "   With this much data, your AI will learn from YEARS of market cycles"
echo ""
echo "🚀 Start the main system to begin massive parallel simulations:"
echo "   python3 v26meme_full.py"
echo ""
echo "💡 Want even MORE data? Run:"
echo "   python3 tools/data_fetcher.py --days 730  # 2 years!"
echo "   python3 tools/data_fetcher.py --days 1095 # 3 years!!"
