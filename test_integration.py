import asyncio
import os
from dotenv import load_dotenv

async def test_exchange_connectivity():
    """Test real exchange connections"""
    load_dotenv()
    
    # Test each exchange
    exchanges_to_test = ['coinbase', 'kraken']
    
    for exchange_name in exchanges_to_test:
        print(f"\n📡 Testing {exchange_name}...")
        
        api_key = os.getenv(f'{exchange_name.upper()}_API_KEY')
        if not api_key:
            print(f"  ⚠️ No API key for {exchange_name}")
            continue
            
        import ccxt.async_support as ccxt
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'apiKey': api_key,
            'secret': os.getenv(f'{exchange_name.upper()}_SECRET'),
            'enableRateLimit': True
        })
        
        try:
            # Test public endpoint
            markets = await exchange.load_markets()
            print(f"  ✅ Connected: {len(markets)} markets")
            
            # Test ticker fetch
            ticker = await exchange.fetch_ticker('BTC/USDT')
            print(f"  ✅ BTC price: ${ticker['last']:,.2f}")
            
            # Test balance (if API key has permissions)
            try:
                balance = await exchange.fetch_balance()
                print(f"  ✅ Balance fetch works")
            except:
                print(f"  ⚠️ Balance fetch failed (check API permissions)")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        finally:
            await exchange.close()

async def test_strategy_generation():
    """Test AI strategy generation"""
    from v26meme_full import OpenAIManager
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️ No OpenAI API key")
        return
        
    manager = OpenAIManager(api_key)
    
    test_prompt = """
    Generate a simple trading strategy that buys when RSI < 30.
    async def execute_strategy(state, opp):
    """
    
    try:
        code = await manager.generate_strategy(test_prompt)
        print(f"✅ Strategy generated: {len(code)} chars")
        assert "async def execute_strategy" in code
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_exchange_connectivity())
    asyncio.run(test_strategy_generation())
