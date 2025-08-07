#!/usr/bin/env python3
"""
Mobile Dashboard Companion - Lightweight mobile-optimized version
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def mobile_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>v26meme Mobile</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
        .metric-card { background: linear-gradient(135deg, #1f2937 0%, #374151 100%); }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="p-4 space-y-4">
        <!-- Header -->
        <div class="text-center py-4">
            <h1 class="text-2xl font-bold">🚀 v26meme</h1>
            <p class="text-sm text-gray-400">$200 → $1M Challenge</p>
            <div id="status" class="text-green-400 text-sm mt-2">🟢 Connected</div>
        </div>

        <!-- Key Metrics Grid -->
        <div class="grid grid-cols-2 gap-3">
            <div class="metric-card p-4 rounded-lg border border-gray-600">
                <div class="text-xs text-gray-400">Equity</div>
                <div id="equity" class="text-xl font-bold text-green-400">$200.00</div>
                <div id="roi" class="text-xs text-green-400">+0.00%</div>
            </div>
            
            <div class="metric-card p-4 rounded-lg border border-gray-600">
                <div class="text-xs text-gray-400">Daily P&L</div>
                <div id="dailyPnl" class="text-xl font-bold">$0.00</div>
                <div id="trades" class="text-xs text-gray-400">0 trades</div>
            </div>
            
            <div class="metric-card p-4 rounded-lg border border-gray-600">
                <div class="text-xs text-gray-400">Win Rate</div>
                <div id="winRate" class="text-xl font-bold text-blue-400">0%</div>
                <div id="strategies" class="text-xs text-gray-400">0 strategies</div>
            </div>
            
            <div class="metric-card p-4 rounded-lg border border-gray-600">
                <div class="text-xs text-gray-400">Progress</div>
                <div id="progress" class="text-xl font-bold text-purple-400">0.02%</div>
                <div id="days" class="text-xs text-gray-400">90 days left</div>
            </div>
        </div>

        <!-- Quick Stats -->
        <div class="metric-card p-4 rounded-lg border border-gray-600">
            <h3 class="font-bold mb-3">📊 Quick Stats</h3>
            <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                    <span>Patterns Found</span>
                    <span id="patterns" class="text-blue-400">0</span>
                </div>
                <div class="flex justify-between">
                    <span>Days Running</span>
                    <span id="daysRunning" class="text-gray-300">0</span>
                </div>
                <div class="flex justify-between">
                    <span>Avg Daily Return</span>
                    <span id="avgDaily" class="text-purple-400">0.00%</span>
                </div>
                <div class="flex justify-between">
                    <span>Required Daily</span>
                    <span id="required" class="text-yellow-400">0.00%</span>
                </div>
            </div>
        </div>

        <!-- Open Positions -->
        <div class="metric-card p-4 rounded-lg border border-gray-600">
            <h3 class="font-bold mb-3">💼 Positions</h3>
            <div id="positionsList" class="text-sm text-gray-400">No open positions</div>
        </div>

        <!-- Recent Activity -->
        <div class="metric-card p-4 rounded-lg border border-gray-600">
            <h3 class="font-bold mb-3">🔄 Recent Activity</h3>
            <div id="activity" class="text-sm text-gray-400">No recent activity</div>
        </div>

        <!-- Last Update -->
        <div class="text-center text-xs text-gray-500 py-4">
            Last update: <span id="lastUpdate">Never</span>
        </div>
    </div>

    <script>
        // Connect to main dashboard WebSocket
        let ws;
        
        function connect() {
            ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                document.getElementById('status').textContent = '🟢 Connected';
                document.getElementById('status').className = 'text-green-400 text-sm mt-2';
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateMobile(data);
            };
            
            ws.onclose = () => {
                document.getElementById('status').textContent = '🔴 Reconnecting...';
                document.getElementById('status').className = 'text-red-400 text-sm mt-2';
                setTimeout(connect, 3000);
            };
        }
        
        function updateMobile(data) {
            if (data.error) return;
            
            const overview = data.overview || {};
            
            // Update main metrics
            document.getElementById('equity').textContent = `$${(overview.equity || 200).toFixed(2)}`;
            document.getElementById('dailyPnl').textContent = `$${(overview.daily_pnl || 0).toFixed(2)}`;
            document.getElementById('winRate').textContent = `${((overview.win_rate || 0) * 100).toFixed(1)}%`;
            document.getElementById('progress').textContent = `${((overview.equity || 200) / 1000000 * 100).toFixed(2)}%`;
            
            // Update ROI
            const roi = overview.roi_percent || 0;
            const roiEl = document.getElementById('roi');
            roiEl.textContent = `${roi >= 0 ? '+' : ''}${roi.toFixed(2)}%`;
            roiEl.className = roi >= 0 ? 'text-xs text-green-400' : 'text-xs text-red-400';
            
            // Update daily P&L color
            const dailyPnl = overview.daily_pnl || 0;
            const dailyEl = document.getElementById('dailyPnl');
            dailyEl.className = dailyPnl >= 0 ? 'text-xl font-bold text-green-400' : 'text-xl font-bold text-red-400';
            
            // Update quick stats
            document.getElementById('trades').textContent = `${overview.trades_today || 0} trades`;
            document.getElementById('strategies').textContent = `${overview.active_strategies || 0} strategies`;
            document.getElementById('days').textContent = `${overview.days_remaining || 90} days left`;
            document.getElementById('patterns').textContent = overview.total_patterns || 0;
            document.getElementById('daysRunning').textContent = overview.days_running || 0;
            document.getElementById('avgDaily').textContent = `${(overview.daily_return_avg || 0).toFixed(2)}%`;
            document.getElementById('required').textContent = `${(overview.required_daily_return || 0).toFixed(2)}%`;
            
            // Update positions
            const positions = data.positions || [];
            const positionsEl = document.getElementById('positionsList');
            if (positions.length === 0) {
                positionsEl.innerHTML = '<div class="text-gray-400">No open positions</div>';
            } else {
                positionsEl.innerHTML = positions.slice(0, 3).map(pos => `
                    <div class="flex justify-between items-center py-1 border-b border-gray-700">
                        <div>
                            <span class="font-medium">${pos.symbol}</span>
                            <span class="text-xs px-1 rounded ${pos.side === 'buy' ? 'bg-green-600' : 'bg-red-600'}">${pos.side.toUpperCase()}</span>
                        </div>
                        <div class="text-right">
                            <div class="${pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'}">$${pos.pnl.toFixed(2)}</div>
                            <div class="text-xs text-gray-400">${(pos.pnl_pct * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                `).join('');
            }
            
            // Update activity
            const trades = data.trades || [];
            const activityEl = document.getElementById('activity');
            if (trades.length === 0) {
                activityEl.innerHTML = '<div class="text-gray-400">No recent trades</div>';
            } else {
                activityEl.innerHTML = trades.slice(0, 3).map(trade => `
                    <div class="flex justify-between items-center py-1 border-b border-gray-700">
                        <div>
                            <span class="font-medium">${trade.symbol}</span>
                            <span class="text-xs px-1 rounded ${trade.side === 'buy' ? 'bg-green-600' : 'bg-red-600'}">${trade.side.toUpperCase()}</span>
                        </div>
                        <div class="text-right">
                            <div class="${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}">$${trade.pnl.toFixed(2)}</div>
                            <div class="text-xs text-gray-400">${new Date(trade.closed_at).toLocaleTimeString()}</div>
                        </div>
                    </div>
                `).join('');
            }
            
            // Update timestamp
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        }
        
        // Start connection
        connect();
        
        // Refresh page on double tap
        let lastTap = 0;
        document.addEventListener('touchend', (e) => {
            const currentTime = new Date().getTime();
            const tapLength = currentTime - lastTap;
            if (tapLength < 500 && tapLength > 0) {
                location.reload();
            }
            lastTap = currentTime;
        });
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    print("📱 Starting Mobile Dashboard on port 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
