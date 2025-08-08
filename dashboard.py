#!/usr/bin/env python3
"""
v26meme Trading Dashboard - Professional Real-time Monitoring Interface

Features:
- Live equity and P&L tracking
- Open positions monitoring
- Strategy performance analysis
- Pattern discovery visualization
- Real-time data updates every 5 seconds
"""

import sqlite3
import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_data():
    """Get all dashboard data from database"""
    try:
        conn = sqlite3.connect('v26meme.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get system state
        cursor.execute("""
            SELECT equity, cash, total_pnl, daily_pnl, win_rate, last_update 
            FROM system_state ORDER BY last_update DESC LIMIT 1
        """)
        state = cursor.fetchone()
        
        # Get positions
        cursor.execute("""
            SELECT symbol, side, amount, entry_price, current_price, pnl, pnl_pct 
            FROM current_positions ORDER BY opened_at DESC
        """)
        positions = cursor.fetchall()
        
        # Get strategies
        cursor.execute("""
            SELECT name, status, total_trades, total_pnl, win_rate 
            FROM strategies ORDER BY total_pnl DESC LIMIT 10
        """)
        strategies = cursor.fetchall()
        
        # Get patterns
        cursor.execute("""
            SELECT type, symbol, confidence, avg_return 
            FROM patterns ORDER BY confidence DESC LIMIT 10
        """)
        patterns = cursor.fetchall()
        
        conn.close()
        
        return {
            'state': state,
            'positions': positions,
            'strategies': strategies,
            'patterns': patterns
        }
    except Exception as e:
        print(f"Database error: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page"""
    data = get_db_data()
    if not data:
        return "<h1>Database Error</h1>"
    
    state = data['state']
    positions = data['positions']
    strategies = data['strategies']
    patterns = data['patterns']
    
    equity = state[0] if state else 10000
    cash = state[1] if state else 0
    total_pnl = state[2] if state else 0
    daily_pnl = state[3] if state else 0
    win_rate = state[4] if state else 0
    
    roi = ((equity - 10000) / 10000) * 100 if equity else 0
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>v26meme Trading Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .metric {{ background: #2a2a2a; padding: 20px; border-radius: 10px; border: 1px solid #444; }}
        .metric h3 {{ margin: 0 0 10px 0; color: #ccc; }}
        .metric .value {{ font-size: 2em; font-weight: bold; }}
        .positive {{ color: #4ade80; }}
        .negative {{ color: #f87171; }}
        .neutral {{ color: #60a5fa; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #444; }}
        th {{ background: #333; }}
        .status {{ padding: 5px 10px; border-radius: 5px; font-size: 0.8em; }}
        .status.active {{ background: #4ade80; color: black; }}
        .status.paper {{ background: #fbbf24; color: black; }}
        .status.micro {{ background: #60a5fa; color: black; }}
        .refresh {{ position: fixed; top: 20px; right: 20px; background: #4ade80; color: black; padding: 10px; border-radius: 5px; text-decoration: none; }}
    </style>
    <script>
        // Auto-refresh every 5 seconds
        setTimeout(() => window.location.reload(), 5000);
    </script>
</head>
<body>
    <div class="header">
        <h1>🚀 v26meme Trading Dashboard</h1>
        <p>$200 → $1M Challenge • Real-time Monitoring</p>
        <p>Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <a href="/" class="refresh">🔄 Refresh</a>
    
    <div class="metrics">
        <div class="metric">
            <h3>💰 Current Equity</h3>
            <div class="value positive">${equity:,.2f}</div>
            <div class="{'positive' if roi >= 0 else 'negative'}">ROI: {roi:+.2f}%</div>
        </div>
        
        <div class="metric">
            <h3>💵 Available Cash</h3>
            <div class="value neutral">${cash:,.2f}</div>
            <div>{(cash/equity*100):.1f}% of equity</div>
        </div>
        
        <div class="metric">
            <h3>📈 Total P&L</h3>
            <div class="value {'positive' if total_pnl >= 0 else 'negative'}">${total_pnl:+.2f}</div>
            <div>Daily: ${daily_pnl:+.2f}</div>
        </div>
        
        <div class="metric">
            <h3>🎯 Win Rate</h3>
            <div class="value neutral">{(win_rate*100):.1f}%</div>
            <div>Performance</div>
        </div>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div class="metric">
            <h3>💼 Open Positions ({len(positions)})</h3>
            <table>
                <thead>
                    <tr><th>Symbol</th><th>Side</th><th>Amount</th><th>Entry</th><th>Current</th><th>P&L</th></tr>
                </thead>
                <tbody>
    """
    
    if positions:
        for pos in positions:
            pnl_color = "positive" if pos[5] >= 0 else "negative"
            html += f"""
                    <tr>
                        <td>{pos[0]}</td>
                        <td><span class="status {'active' if pos[1] == 'buy' else 'paper'}">{pos[1].upper()}</span></td>
                        <td>{pos[2]:.4f}</td>
                        <td>${pos[3]:.4f}</td>
                        <td>${pos[4]:.4f}</td>
                        <td class="{pnl_color}">${pos[5]:+.2f} ({(pos[6]*100):+.2f}%)</td>
                    </tr>
            """
    else:
        html += "<tr><td colspan='6' style='text-align: center; color: #888;'>No open positions</td></tr>"
    
    html += """
                </tbody>
            </table>
        </div>
        
        <div class="metric">
            <h3>🧬 Active Strategies ({len(strategies)})</h3>
            <table>
                <thead>
                    <tr><th>Strategy</th><th>Status</th><th>Trades</th><th>P&L</th><th>Win Rate</th></tr>
                </thead>
                <tbody>
    """
    
    if strategies:
        for strat in strategies:
            pnl_color = "positive" if strat[3] >= 0 else "negative"
            status_class = strat[1].lower()
            html += f"""
                    <tr>
                        <td>{strat[0]}</td>
                        <td><span class="status {status_class}">{strat[1]}</span></td>
                        <td>{strat[2]}</td>
                        <td class="{pnl_color}">${strat[3]:+.2f}</td>
                        <td>{(strat[4]*100):.1f}%</td>
                    </tr>
            """
    else:
        html += "<tr><td colspan='5' style='text-align: center; color: #888;'>No strategies loaded</td></tr>"
    
    html += """
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="metric">
        <h3>💡 Recent Patterns ({len(patterns)})</h3>
        <table>
            <thead>
                <tr><th>Type</th><th>Symbol</th><th>Confidence</th><th>Avg Return</th></tr>
            </thead>
            <tbody>
    """
    
    if patterns:
        for pat in patterns:
            html += f"""
                <tr>
                    <td>{pat[0]}</td>
                    <td>{pat[1] or 'Multi'}</td>
                    <td>{(pat[2]*100):.1f}%</td>
                    <td class="positive">{(pat[3]*100):.2f}%</td>
                </tr>
            """
    else:
        html += "<tr><td colspan='4' style='text-align: center; color: #888;'>No patterns discovered</td></tr>"
    
    html += """
            </tbody>
        </table>
    </div>
</body>
</html>
    """
    
    return HTMLResponse(content=html)

@app.get("/api/data")
async def get_data():
    """API endpoint for data"""
    return get_db_data()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
