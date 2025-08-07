import asyncio
import time
import psutil
import tracemalloc

async def test_memory_usage():
    """Monitor memory usage during operation"""
    tracemalloc.start()
    
    from v26meme_full import AutonomousTrader
    
    trader = AutonomousTrader()
    await trader.initialize()
    
    # Baseline
    snapshot1 = tracemalloc.take_snapshot()
    
    # Run for a bit
    for _ in range(10):
        await trader._scan_markets()
        await asyncio.sleep(1)
    
    # Check memory growth
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("\n📊 Memory Usage Growth:")
    for stat in top_stats[:10]:
        print(stat)
    
    # Check for leaks
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"\nTotal Memory: {memory_mb:.2f} MB")
    
    assert memory_mb < 500  # Should not exceed 500MB

async def test_latency():
    """Test response times"""
    from v26meme_full import AutonomousTrader
    
    trader = AutonomousTrader()
    await trader.initialize()
    
    # Test market scan speed
    start = time.time()
    opportunities = await trader._scan_markets()
    scan_time = time.time() - start
    
    print(f"\n⏱️ Market scan: {scan_time:.2f}s for {len(opportunities)} opportunities")
    assert scan_time < 10  # Should complete within 10 seconds

if __name__ == "__main__":
    asyncio.run(test_memory_usage())
    asyncio.run(test_latency())
