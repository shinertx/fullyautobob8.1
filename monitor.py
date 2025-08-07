#!/usr/bin/env python3
"""Real-time monitoring of the trading bot"""

import sqlite3
import time
from datetime import datetime

def monitor_health():
    conn = sqlite3.connect('v26meme.db')
    
    while True:
        # Check last update
        cursor = conn.execute(
            "SELECT last_update, equity, daily_pnl FROM system_state ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        
        if row:
            last_update = datetime.fromisoformat(row[0])
            age = (datetime.utcnow() - last_update).seconds
            
            if age > 60:
                print(f"⚠️ WARNING: No updates for {age} seconds")
            
            print(f"💰 Equity: ${row[1]:.2f} | Daily P&L: ${row[2]:.2f}")
        
        # Check active positions
        cursor = conn.execute("SELECT COUNT(*) FROM current_positions")
        pos_count = cursor.fetchone()[0]
        print(f"📊 Active Positions: {pos_count}")
        
        time.sleep(10)

if __name__ == "__main__":
    monitor_health()
