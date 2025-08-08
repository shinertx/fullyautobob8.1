#!/usr/bin/env python3
"""
Manual SimLab enqueue utility for v26meme
Usage: python tools/sim_enqueue.py
"""
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simlab import SimLab

async def main():
    db = os.getenv("DB_PATH", "v26meme.db")
    # minimal shims for standalone enqueue
    async def dummy_run(strategy, opp): 
        return {"action":"hold","conf":0.0}
    
    def empty_fetch(): 
        return {}
    
    lab = SimLab(
        db_path=db, 
        run_strategy_callable=dummy_run, 
        fetch_strategies_callable=empty_fetch
    )
    
    # Example: enqueue a BTC/USDC 5m replay
    run_id = await lab.enqueue_replay(
        "BTC/USDC", 
        "5m", 
        "", 
        "", 
        params={"file": "data/BTC_USDC_5m.csv"}
    )
    print(f"Enqueued simulation run: {run_id}")

if __name__ == "__main__":
    asyncio.run(main())
