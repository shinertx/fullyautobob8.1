#!/usr/bin/env python3
"""Script to clear strategies with hardcoded symbols"""
import sqlite3
import sys
import os

def clear_bad_strategies():
    """Clear strategies that contain hardcoded symbol names"""
    
    db_path = 'v26meme.db'
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First, check how many bad strategies exist
        cursor.execute("""
            SELECT COUNT(*) FROM strategies 
            WHERE name LIKE '%SWELL%' 
            OR name LIKE '%PROVE%' 
            OR description LIKE '%SWELL/USDC%'
            OR description LIKE '%PROVE/USDC%'
            OR code LIKE "%opp['symbol'] == 'SWELL%"
            OR code LIKE "%opp['symbol'] == 'PROVE%"
        """)
        
        count_before = cursor.fetchone()[0]
        print(f"Found {count_before} strategies with hardcoded symbols")
        
        if count_before == 0:
            print("No bad strategies to clean up!")
            conn.close()
            return
        
        # Delete strategies with specific symbol names
        cursor.execute("""
            DELETE FROM strategies 
            WHERE name LIKE '%SWELL%' 
            OR name LIKE '%PROVE%' 
            OR description LIKE '%SWELL/USDC%'
            OR description LIKE '%PROVE/USDC%'
            OR code LIKE "%opp['symbol'] == 'SWELL%"
            OR code LIKE "%opp['symbol'] == 'PROVE%"
        """)
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        print(f"✅ Deleted {deleted_count} bad strategies with hardcoded symbols")
        
        # Check remaining strategies
        cursor.execute("SELECT COUNT(*) FROM strategies")
        remaining = cursor.fetchone()[0]
        print(f"📊 {remaining} strategies remaining in database")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error cleaning strategies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clear_bad_strategies()
