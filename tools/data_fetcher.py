#!/usr/bin/env python3
"""
Data Fetcher for SimLab
Automatically downloads historical OHLCV data from multiple exchanges
"""

import asyncio
import aiohttp
import pandas as pd
import ccxt.async_support as ccxt
import os
from datetime import datetime, timedelta
from typing import Optional
import argparse

class DataFetcher:
    """Fetches historical data from multiple sources for SimLab"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.session = None
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # MASSIVE symbol list - get everything!
        self.symbols = [
            # Major pairs - all quote currencies
            'BTC/USDT', 'BTC/USDC', 'BTC/USD', 'BTC/EUR',
            'ETH/USDT', 'ETH/USDC', 'ETH/USD', 'ETH/EUR', 'ETH/BTC',
            
            # Top 50 by market cap
            'SOL/USDT', 'SOL/USDC', 'SOL/USD',
            'ADA/USDT', 'ADA/USDC', 'ADA/USD',
            'AVAX/USDT', 'AVAX/USDC', 'AVAX/USD',
            'DOT/USDT', 'DOT/USDC', 'DOT/USD',
            'MATIC/USDT', 'MATIC/USDC', 'MATIC/USD',
            'LINK/USDT', 'LINK/USDC', 'LINK/USD',
            'UNI/USDT', 'UNI/USDC', 'UNI/USD',
            'ATOM/USDT', 'ATOM/USDC', 'ATOM/USD',
            'LTC/USDT', 'LTC/USDC', 'LTC/USD',
            'XRP/USDT', 'XRP/USDC', 'XRP/USD',
            'DOGE/USDT', 'DOGE/USDC', 'DOGE/USD',
            'SHIB/USDT', 'SHIB/USDC',
            
            # DeFi coins
            'AAVE/USDT', 'AAVE/USDC', 'AAVE/USD',
            'COMP/USDT', 'COMP/USDC', 'COMP/USD',
            'MKR/USDT', 'MKR/USDC', 'MKR/USD',
            'SNX/USDT', 'SNX/USDC', 'SNX/USD',
            'CRV/USDT', 'CRV/USDC', 'CRV/USD',
            'YFI/USDT', 'YFI/USDC', 'YFI/USD',
            'SUSHI/USDT', 'SUSHI/USDC', 'SUSHI/USD',
            '1INCH/USDT', '1INCH/USDC', '1INCH/USD',
            
            # Layer 1 & 2 coins
            'NEAR/USDT', 'NEAR/USDC', 'NEAR/USD',
            'FTM/USDT', 'FTM/USDC', 'FTM/USD',
            'ALGO/USDT', 'ALGO/USDC', 'ALGO/USD',
            'VET/USDT', 'VET/USDC', 'VET/USD',
            'ICP/USDT', 'ICP/USDC', 'ICP/USD',
            'FLOW/USDT', 'FLOW/USDC', 'FLOW/USD',
            'HBAR/USDT', 'HBAR/USDC', 'HBAR/USD',
            'XTZ/USDT', 'XTZ/USDC', 'XTZ/USD',
            'EOS/USDT', 'EOS/USDC', 'EOS/USD',
            'WAVES/USDT', 'WAVES/USDC', 'WAVES/USD',
            
            # Meme & trending coins
            'PEPE/USDT', 'PEPE/USDC',
            'WIF/USDT', 'WIF/USDC',
            'BONK/USDT', 'BONK/USDC',
            'FLOKI/USDT', 'FLOKI/USDC',
            'MEME/USDT', 'MEME/USDC',
            
            # Gaming & metaverse
            'SAND/USDT', 'SAND/USDC', 'SAND/USD',
            'MANA/USDT', 'MANA/USDC', 'MANA/USD',
            'AXS/USDT', 'AXS/USDC', 'AXS/USD',
            'ENJ/USDT', 'ENJ/USDC', 'ENJ/USD',
            'GALA/USDT', 'GALA/USDC', 'GALA/USD',
            'IMX/USDT', 'IMX/USDC', 'IMX/USD',
            
            # Enterprise & utility
            'RNDR/USDT', 'RNDR/USDC', 'RNDR/USD',
            'FIL/USDT', 'FIL/USDC', 'FIL/USD',
            'AR/USDT', 'AR/USDC', 'AR/USD',
            'GRT/USDT', 'GRT/USDC', 'GRT/USD',
            'LPT/USDT', 'LPT/USDC', 'LPT/USD',
            'BAT/USDT', 'BAT/USDC', 'BAT/USD',
            'ZRX/USDT', 'ZRX/USDC', 'ZRX/USD',
            
            # AI & tech tokens
            'FET/USDT', 'FET/USDC', 'FET/USD',
            'AGIX/USDT', 'AGIX/USDC', 'AGIX/USD',
            'OCEAN/USDT', 'OCEAN/USDC', 'OCEAN/USD',
            
            # Privacy coins
            'XMR/USDT', 'XMR/USDC', 'XMR/USD',
            'ZEC/USDT', 'ZEC/USDC', 'ZEC/USD',
            'DASH/USDT', 'DASH/USDC', 'DASH/USD',
            
            # Cross-chain & interop
            'DOT/USDT', 'DOT/USDC', 'DOT/USD',
            'KSM/USDT', 'KSM/USDC', 'KSM/USD',
            'ATOM/USDT', 'ATOM/USDC', 'ATOM/USD'
        ]
        
        # ALL timeframes - get everything for maximum pattern discovery
        self.timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_coinbase_data(self, symbol: str, timeframe: str, days_back: int = 365) -> Optional[pd.DataFrame]:
        """Fetch data from Coinbase Pro (free, no API key needed)"""
        try:
            # Convert symbol format for Coinbase
            base, quote = symbol.split('/')
            coinbase_symbol = f"{base}-{quote}"
            
            # Convert timeframe - Coinbase supports these granularities
            timeframe_map = {
                '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '8h': 28800,
                '12h': 43200, '1d': 86400, '3d': 259200, '1w': 604800
            }
            
            if timeframe not in timeframe_map:
                return None
                
            granularity = timeframe_map[timeframe]
            
            # Calculate time range - fetch MASSIVE history
            end_time = datetime.utcnow()
            
            # Coinbase has a 300 candle limit per request, so we'll chunk it
            all_data = []
            current_end = end_time
            
            # Fetch in chunks to get maximum history
            for chunk in range(min(10, days_back // 30)):  # Up to 10 chunks (300 days max from Coinbase)
                current_start = current_end - timedelta(days=30)
                
                # Coinbase API endpoint
                url = f"https://api.exchange.coinbase.com/products/{coinbase_symbol}/candles"
                params = {
                    'start': current_start.isoformat(),
                    'end': current_end.isoformat(),
                    'granularity': granularity
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        chunk_data = await response.json()
                        if chunk_data:
                            all_data.extend(chunk_data)
                
                current_end = current_start
                await asyncio.sleep(0.1)  # Rate limiting
            
            if all_data:
                # Convert to DataFrame
                df = pd.DataFrame(all_data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp')
                
                # Convert timestamp to milliseconds for consistency
                df['timestamp'] = (df['timestamp'].astype('int64') // 10**6).astype('int64')
                
                print(f"✅ Fetched {len(df)} candles for {symbol} {timeframe} from Coinbase")
                return df
                    
        except Exception as e:
            print(f"❌ Failed to fetch {symbol} {timeframe} from Coinbase: {e}")
            
        return None

    async def fetch_kraken_data(self, symbol: str, timeframe: str, days_back: int = 365) -> Optional[pd.DataFrame]:
        """Fetch data from Kraken (free API) - get MASSIVE history"""
        try:
            exchange = ccxt.kraken({'enableRateLimit': True})
            
            # Convert timeframe
            since = int((datetime.utcnow() - timedelta(days=days_back)).timestamp() * 1000)
            
            # Kraken allows up to 720 candles per request, so fetch in chunks
            all_data = []
            current_since = since
            
            for chunk in range(min(20, days_back // 30)):  # Up to 20 chunks for massive history
                try:
                    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=720)
                    if ohlcv:
                        all_data.extend(ohlcv)
                        # Update since to last timestamp
                        if len(ohlcv) > 0:
                            current_since = ohlcv[-1][0] + 1
                        else:
                            break
                    await asyncio.sleep(1)  # Rate limiting for Kraken
                except Exception:
                    break  # Stop if we hit an error
                    
            await exchange.close()
            
            if all_data:
                df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                print(f"✅ Fetched {len(df)} candles for {symbol} {timeframe} from Kraken")
                return df
                
        except Exception as e:
            print(f"❌ Failed to fetch {symbol} {timeframe} from Kraken: {e}")
            
        return None

    async def fetch_coingecko_data(self, symbol: str, days_back: int = 30) -> Optional[pd.DataFrame]:
        """Fetch daily data from CoinGecko (free API, daily only)"""
        try:
            # Convert symbol to CoinGecko format
            base = symbol.split('/')[0].lower()
            
            # Map common symbols to CoinGecko IDs
            symbol_map = {
                'btc': 'bitcoin', 'eth': 'ethereum', 'sol': 'solana',
                'ada': 'cardano', 'matic': 'polygon', 'doge': 'dogecoin',
                'dot': 'polkadot', 'link': 'chainlink', 'uni': 'uniswap',
                'aave': 'aave', 'avax': 'avalanche-2', 'atom': 'cosmos'
            }
            
            coin_id = symbol_map.get(base, base)
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            params = {'vs_currency': 'usd', 'days': str(days_back)}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                    df['volume'] = 0  # CoinGecko OHLC doesn't include volume
                    
                    print(f"✅ Fetched {len(df)} daily candles for {symbol} from CoinGecko")
                    return df
                    
        except Exception as e:
            print(f"❌ Failed to fetch {symbol} from CoinGecko: {e}")
            
        return None

    def save_dataframe(self, df: pd.DataFrame, symbol: str, timeframe: str, exchange: str = "multi", compress: bool = True):
        """Save DataFrame to CSV file with optional compression"""
        # Clean symbol for filename
        clean_symbol = symbol.replace('/', '_')
        
        if compress:
            filename = f"{clean_symbol}_{timeframe}.csv.gz"
            filepath = os.path.join(self.data_dir, filename)
        else:
            filename = f"{clean_symbol}_{timeframe}.csv"
            filepath = os.path.join(self.data_dir, filename)
        
        # Ensure proper column order and types
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.dropna()
        
        # Save to CSV (compressed or uncompressed)
        if compress:
            df.to_csv(filepath, index=False, compression='gzip')
            print(f"💾 Saved {len(df)} rows to {filepath} (compressed)")
        else:
            df.to_csv(filepath, index=False)
            print(f"💾 Saved {len(df)} rows to {filepath}")
        
        return filepath

    async def fetch_symbol_timeframe(self, symbol: str, timeframe: str, days_back: int = 365, compress: bool = True):
        """Fetch data for a specific symbol and timeframe from multiple sources - DEFAULT 1 YEAR!"""
        print(f"\n🔍 Fetching {symbol} {timeframe} ({days_back} days)...")
        
        # Try Coinbase first (most reliable)
        df = await self.fetch_coinbase_data(symbol, timeframe, days_back)
        
        # Fallback to Kraken if Coinbase fails
        if df is None:
            df = await self.fetch_kraken_data(symbol, timeframe, days_back)
        
        # For daily data, can also try CoinGecko
        if df is None and timeframe == '1d':
            df = await self.fetch_coingecko_data(symbol, days_back)
            
        if df is not None and len(df) > 0:
            self.save_dataframe(df, symbol, timeframe, compress=compress)
            return True
            
        print(f"❌ No data found for {symbol} {timeframe}")
        return False

    async def fetch_all_data(self, days_back: int = 365, max_concurrent: int = 3, compress: bool = True):
        """Fetch MASSIVE amounts of data for all symbols and timeframes - DEFAULT 1 YEAR!"""
        print(f"🚀 Starting MASSIVE data fetch ({days_back} days back)...")
        print(f"📊 Will attempt to fetch {len(self.symbols)} symbols × {len(self.timeframes)} timeframes = {len(self.symbols) * len(self.timeframes)} total files")
        if compress:
            print("🗜️ Files will be compressed to save disk space!")
        
        tasks = []
        semaphore = asyncio.Semaphore(max_concurrent)  # Lower concurrency to avoid rate limits
        
        async def fetch_with_semaphore(symbol, timeframe):
            async with semaphore:
                await self.fetch_symbol_timeframe(symbol, timeframe, days_back, compress)
                # More aggressive rate limiting for massive requests
                await asyncio.sleep(1.0)
        
        # Create tasks for all symbol/timeframe combinations
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                task = fetch_with_semaphore(symbol, timeframe)
                tasks.append(task)
        
        print(f"🔥 Starting {len(tasks)} download tasks with {max_concurrent} concurrent connections...")
        
        # Execute all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count files created (both .csv and .csv.gz)
        files = os.listdir(self.data_dir)
        csv_files = [f for f in files if f.endswith('.csv') or f.endswith('.csv.gz')]
        
        print("\n🎉 MASSIVE data fetch complete!")
        print(f"📁 {len(csv_files)} data files in {self.data_dir}/")
        print(f"🧪 SimLab will have {len(csv_files)} historical datasets to simulate!")
        print(f"📊 Estimated total candles: {len(csv_files) * 500} (~{len(csv_files) * 500 / 1000}K)")
        
        return csv_files

async def main():
    parser = argparse.ArgumentParser(description="Fetch MASSIVE historical data for SimLab")
    parser.add_argument("--days", type=int, default=365, help="Days of history to fetch (default: 365 = 1 YEAR!)")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch (default: 100+ symbols)")
    parser.add_argument("--timeframes", nargs="+", help="Specific timeframes to fetch (default: 14 timeframes)")
    parser.add_argument("--data-dir", default="data", help="Directory to save data files")
    parser.add_argument("--max-files", type=int, default=2000, help="Maximum number of files to create")
    parser.add_argument("--compress", action="store_true", default=True, help="Compress CSV files with gzip (default: True)")
    parser.add_argument("--no-compress", dest="compress", action="store_false", help="Disable compression")
    
    args = parser.parse_args()
    
    print("🚀 MASSIVE Data Fetcher for SimLab")
    print("================================")
    print(f"📅 History: {args.days} days")
    print(f"💾 Max files: {args.max_files}")
    print(f"📂 Data dir: {args.data_dir}")
    
    async with DataFetcher(args.data_dir) as fetcher:
        if args.symbols:
            fetcher.symbols = args.symbols
        if args.timeframes:
            fetcher.timeframes = args.timeframes
            
        # Show what we're about to fetch
        total_combinations = len(fetcher.symbols) * len(fetcher.timeframes)
        print(f"🎯 Attempting {total_combinations} combinations:")
        print(f"   📊 {len(fetcher.symbols)} symbols")
        print(f"   ⏱️ {len(fetcher.timeframes)} timeframes")
        print(f"   📈 Up to {args.days} days of history each")
        print()
        
        if total_combinations > args.max_files:
            print(f"⚠️ Total combinations ({total_combinations}) exceeds max files ({args.max_files})")
            print(f"   Limiting to first {args.max_files // len(fetcher.timeframes)} symbols")
            fetcher.symbols = fetcher.symbols[:args.max_files // len(fetcher.timeframes)]
            
        await fetcher.fetch_all_data(days_back=args.days, compress=args.compress)

if __name__ == "__main__":
    asyncio.run(main())
