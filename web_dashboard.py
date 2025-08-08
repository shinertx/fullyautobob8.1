#!/usr/bin/env python3
"""
v26meme Trading Dashboard - Professional Real-Time Web Interface
High-frequency updates, beautiful charts, comprehensive monitoring
"""

import asyncio
import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TradingDashboard:
    """Professional trading dashboard with real-time updates"""
    
    def __init__(self, db_path: str = "v26meme.db"):
        self.db_path = db_path
        self.active_connections: List[WebSocket] = []
        self.app = FastAPI(title="v26meme Trading Dashboard", version="1.0.0")
        self.setup_routes()
        self.setup_middleware()
    
    def setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the main dashboard"""
            return self.get_dashboard_html()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await self.connect(websocket)
            try:
                while True:
                    # Send updates every 2 seconds
                    data = await self.get_dashboard_data()
                    await websocket.send_text(json.dumps(data))
                    await asyncio.sleep(2)
            except WebSocketDisconnect:
                self.disconnect(websocket)
        
        @self.app.get("/api/overview")
        async def get_overview():
            """Get system overview data"""
            return await self.get_system_overview()
        
        @self.app.get("/api/positions")
        async def get_positions():
            """Get current positions"""
            return await self.get_current_positions()
        
        @self.app.get("/api/strategies")
        async def get_strategies():
            """Get strategy performance"""
            return await self.get_strategy_performance()
        
        @self.app.get("/api/patterns")
        async def get_patterns():
            """Get discovered patterns"""
            return await self.get_pattern_data()
        
        @self.app.get("/api/trades")
        async def get_trades():
            """Get recent trades"""
            return await self.get_recent_trades()
        
        @self.app.get("/api/equity-chart")
        async def get_equity_chart():
            """Get equity curve data"""
            return await self.get_equity_curve()
        
        @self.app.get("/api/performance")
        async def get_performance():
            """Get detailed performance metrics"""
            return await self.get_performance_metrics()
    
    async def connect(self, websocket: WebSocket):
        """Connect new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info(f"🔗 Dashboard client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket client"""
        self.active_connections.remove(websocket)
        log.info(f"📴 Dashboard client disconnected. Total: {len(self.active_connections)}")
    
    def get_db_connection(self):
        """Get database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            log.error(f"Database connection error: {e}")
            return None
    
    async def get_dashboard_data(self) -> Dict:
        """Get all dashboard data"""
        try:
            overview = await self.get_system_overview()
            positions = await self.get_current_positions()
            strategies = await self.get_strategy_performance()
            trades = await self.get_recent_trades(limit=10)
            patterns = await self.get_pattern_data(limit=20)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overview": overview,
                "positions": positions,
                "strategies": strategies,
                "trades": trades,
                "patterns": patterns,
                "status": "operational"
            }
        except Exception as e:
            log.error(f"Error getting dashboard data: {e}")
            return {"error": str(e), "status": "error"}
    
    async def get_system_overview(self) -> Dict:
        """Get system overview metrics"""
        conn = self.get_db_connection()
        if not conn:
            return {"error": "Database connection failed"}
        
        try:
            cursor = conn.cursor()
            
            # Get latest system state
            cursor.execute("""
                SELECT equity, cash, total_pnl, daily_pnl, win_rate, last_update
                FROM system_state ORDER BY last_update DESC LIMIT 1
            """)
            state = cursor.fetchone()
            
            # Get total trades today
            today = datetime.utcnow().date()
            cursor.execute("""
                SELECT COUNT(*) as trades_today, 
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins_today
                FROM trades 
                WHERE DATE(closed_at) = ?
            """, (today,))
            trades_today = cursor.fetchone()
            
            # Get active strategies count
            cursor.execute("""
                SELECT COUNT(*) as active_strategies
                FROM strategies 
                WHERE status IN ('active', 'micro')
            """)
            active_strategies = cursor.fetchone()
            
            # Get discovered patterns count
            cursor.execute("SELECT COUNT(*) as total_patterns FROM patterns")
            patterns_count = cursor.fetchone()
            
            # Calculate performance metrics
            initial_capital = float(os.getenv("INITIAL_CAPITAL", 200.0))
            current_equity = state['equity'] if state else initial_capital
            total_pnl = state['total_pnl'] if state else 0
            roi = ((current_equity / initial_capital) - 1) * 100
            
            # Days since start (assuming started recently)
            start_date = datetime(2025, 8, 6)  # Adjust based on actual start
            days_running = (datetime.utcnow() - start_date).days + 1
            daily_return = roi / days_running if days_running > 0 else 0
            
            # Target calculations
            target_equity = 1_000_000
            days_remaining = 90 - days_running
            required_daily_return = ((target_equity / current_equity) ** (1/days_remaining) - 1) * 100 if days_remaining > 0 else 0
            
            return {
                "equity": current_equity,
                "cash": state[1] if state and len(state) > 1 else 0,  # cash is second column
                "total_pnl": total_pnl,
                "daily_pnl": state[3] if state and len(state) > 3 else 0,  # daily_pnl is fourth column
                "roi_percent": roi,
                "win_rate": state[4] if state and len(state) > 4 else 0,  # win_rate is fifth column
                "trades_today": trades_today[0] if trades_today else 0,
                "wins_today": trades_today[1] if trades_today else 0,
                "active_strategies": active_strategies[0] if active_strategies else 0,
                "total_patterns": patterns_count[0] if patterns_count else 0,
                "days_running": days_running,
                "days_remaining": days_remaining,
                "daily_return_avg": daily_return,
                "required_daily_return": required_daily_return,
                "target_equity": target_equity,
                "initial_capital": initial_capital,
                "last_update": state[5] if state and len(state) > 5 else datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            log.error(f"Error getting system overview: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    async def get_current_positions(self) -> List[Dict]:
        """Get current open positions"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, strategy_id, symbol, exchange, side, amount, 
                       entry_price, current_price, pnl, pnl_pct, opened_at,
                       stop_loss, take_profit
                FROM current_positions 
                ORDER BY opened_at DESC
            """)
            
            positions = []
            for row in cursor.fetchall():
                positions.append({
                    "id": row['id'],
                    "strategy_id": row['strategy_id'],
                    "symbol": row['symbol'],
                    "exchange": row['exchange'],
                    "side": row['side'],
                    "amount": row['amount'],
                    "entry_price": row['entry_price'],
                    "current_price": row['current_price'],
                    "pnl": row['pnl'],
                    "pnl_pct": row['pnl_pct'],
                    "opened_at": row['opened_at'],
                    "stop_loss": row['stop_loss'],
                    "take_profit": row['take_profit']
                })
            
            return positions
        
        except Exception as e:
            log.error(f"Error getting positions: {e}")
            return []
        finally:
            conn.close()
    
    async def get_strategy_performance(self) -> List[Dict]:
        """Get strategy performance data"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, status, total_trades, winning_trades, total_pnl, 
                       win_rate, sharpe_ratio, max_drawdown, last_trade
                FROM strategies 
                WHERE status != 'retired'
                ORDER BY total_pnl DESC
            """)
            
            strategies = []
            for row in cursor.fetchall():
                strategies.append({
                    "name": row['name'],
                    "status": row['status'],
                    "total_trades": row['total_trades'],
                    "winning_trades": row['winning_trades'],
                    "total_pnl": row['total_pnl'],
                    "win_rate": row['win_rate'],
                    "sharpe_ratio": row['sharpe_ratio'],
                    "max_drawdown": row['max_drawdown'],
                    "last_trade": row['last_trade']
                })
            
            return strategies
        
        except Exception as e:
            log.error(f"Error getting strategies: {e}")
            return []
        finally:
            conn.close()
    
    async def get_pattern_data(self, limit: int = 50) -> List[Dict]:
        """Get discovered patterns"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT type, symbol, description, confidence, win_rate, 
                       avg_return, sample_size, discovered_at
                FROM patterns 
                ORDER BY confidence DESC, discovered_at DESC 
                LIMIT ?
            """, (limit,))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    "type": row['type'],
                    "symbol": row['symbol'],
                    "description": row['description'],
                    "confidence": row['confidence'],
                    "win_rate": row['win_rate'],
                    "avg_return": row['avg_return'],
                    "sample_size": row['sample_size'],
                    "discovered_at": row['discovered_at']
                })
            
            return patterns
        
        except Exception as e:
            log.error(f"Error getting patterns: {e}")
            return []
        finally:
            conn.close()
    
    async def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, side, amount, price, pnl, pnl_pct, 
                       opened_at, closed_at, strategy_id
                FROM trades 
                ORDER BY closed_at DESC 
                LIMIT ?
            """, (limit,))
            
            trades = []
            for row in cursor.fetchall():
                trades.append({
                    "symbol": row['symbol'],
                    "side": row['side'],
                    "amount": row['amount'],
                    "price": row['price'],
                    "pnl": row['pnl'],
                    "pnl_pct": row['pnl_pct'],
                    "opened_at": row['opened_at'],
                    "closed_at": row['closed_at'],
                    "strategy_id": row['strategy_id']
                })
            
            return trades
        
        except Exception as e:
            log.error(f"Error getting trades: {e}")
            return []
        finally:
            conn.close()
    
    async def get_equity_curve(self) -> Dict:
        """Get equity curve data for charting"""
        conn = self.get_db_connection()
        if not conn:
            return {"timestamps": [], "equity": [], "pnl": []}
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT last_update, equity, total_pnl
                FROM system_state 
                ORDER BY last_update ASC
            """)
            
            timestamps = []
            equity_values = []
            pnl_values = []
            
            for row in cursor.fetchall():
                timestamps.append(row['last_update'])
                equity_values.append(row['equity'])
                pnl_values.append(row['total_pnl'])
            
            return {
                "timestamps": timestamps,
                "equity": equity_values,
                "pnl": pnl_values
            }
        
        except Exception as e:
            log.error(f"Error getting equity curve: {e}")
            return {"timestamps": [], "equity": [], "pnl": []}
        finally:
            conn.close()
    
    async def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            # Get daily P&L for last 30 days
            cursor.execute("""
                SELECT DATE(closed_at) as trade_date, SUM(pnl) as daily_pnl
                FROM trades 
                WHERE closed_at >= datetime('now', '-30 days')
                GROUP BY DATE(closed_at)
                ORDER BY trade_date DESC
            """)
            daily_pnl = cursor.fetchall()
            
            # Get win/loss distribution
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_win,
                    MIN(pnl) as max_loss
                FROM trades
            """)
            trade_stats = cursor.fetchone()
            
            return {
                "daily_pnl": [{"date": row['trade_date'], "pnl": row['daily_pnl']} for row in daily_pnl],
                "trade_stats": {
                    "total_trades": trade_stats['total_trades'] if trade_stats else 0,
                    "wins": trade_stats['wins'] if trade_stats else 0,
                    "losses": trade_stats['losses'] if trade_stats else 0,
                    "avg_pnl": trade_stats['avg_pnl'] if trade_stats else 0,
                    "max_win": trade_stats['max_win'] if trade_stats else 0,
                    "max_loss": trade_stats['max_loss'] if trade_stats else 0
                }
            }
        
        except Exception as e:
            log.error(f"Error getting performance metrics: {e}")
            return {}
        finally:
            conn.close()
    
    def get_dashboard_html(self) -> str:
        """Generate the dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>v26meme Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card { backdrop-filter: blur(10px); }
        .pulse-green { animation: pulse-green 2s infinite; }
        .pulse-red { animation: pulse-red 2s infinite; }
        @keyframes pulse-green { 0%, 100% { background-color: rgb(34, 197, 94); } 50% { background-color: rgb(22, 163, 74); } }
        @keyframes pulse-red { 0%, 100% { background-color: rgb(239, 68, 68); } 50% { background-color: rgb(220, 38, 38); } }
    </style>
</head>
<body class="bg-gray-900 text-white">
    <!-- Header -->
    <div class="gradient-bg p-6 shadow-lg">
        <div class="max-w-7xl mx-auto flex justify-between items-center">
            <div>
                <h1 class="text-3xl font-bold">🚀 v26meme Trading Dashboard</h1>
                <p class="text-lg opacity-90">$200 → $1M Challenge • Real-time Monitoring</p>
            </div>
            <div class="text-right">
                <div id="status" class="text-lg font-semibold">🔴 Connecting...</div>
                <div id="lastUpdate" class="text-sm opacity-75">Last update: Never</div>
            </div>
        </div>
    </div>

    <!-- Main Dashboard -->
    <div class="max-w-7xl mx-auto p-6 space-y-6">
        
        <!-- Key Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">Current Equity</p>
                        <p id="equity" class="text-3xl font-bold text-green-400">$200.00</p>
                    </div>
                    <div class="text-4xl">💰</div>
                </div>
                <div class="mt-4">
                    <div id="roi" class="text-sm text-green-400">+0.00%</div>
                </div>
            </div>
            
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">Daily P&L</p>
                        <p id="dailyPnl" class="text-3xl font-bold">$0.00</p>
                    </div>
                    <div class="text-4xl">📈</div>
                </div>
                <div class="mt-4">
                    <div id="tradesToday" class="text-sm text-gray-400">0 trades today</div>
                </div>
            </div>
            
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">Win Rate</p>
                        <p id="winRate" class="text-3xl font-bold text-blue-400">0%</p>
                    </div>
                    <div class="text-4xl">🎯</div>
                </div>
                <div class="mt-4">
                    <div id="activeStrategies" class="text-sm text-gray-400">0 active strategies</div>
                </div>
            </div>
            
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">Target Progress</p>
                        <p id="targetProgress" class="text-3xl font-bold text-purple-400">0.02%</p>
                    </div>
                    <div class="text-4xl">🎪</div>
                </div>
                <div class="mt-4">
                    <div id="daysRemaining" class="text-sm text-gray-400">90 days remaining</div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">📊 Equity Curve</h3>
                <canvas id="equityChart" width="400" height="200"></canvas>
            </div>
            
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">🔥 Daily Performance</h3>
                <canvas id="dailyChart" width="400" height="200"></canvas>
            </div>
        </div>

        <!-- Data Tables Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Current Positions -->
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">💼 Open Positions</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="border-b border-gray-700">
                                <th class="text-left py-2">Symbol</th>
                                <th class="text-left py-2">Side</th>
                                <th class="text-left py-2">Amount</th>
                                <th class="text-left py-2">Entry</th>
                                <th class="text-left py-2">Current</th>
                                <th class="text-left py-2">P&L</th>
                            </tr>
                        </thead>
                        <tbody id="positionsTable">
                            <tr>
                                <td colspan="6" class="text-center py-4 text-gray-400">No open positions</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Active Strategies -->
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">🧬 Active Strategies</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="border-b border-gray-700">
                                <th class="text-left py-2">Strategy</th>
                                <th class="text-left py-2">Status</th>
                                <th class="text-left py-2">Trades</th>
                                <th class="text-left py-2">P&L</th>
                                <th class="text-left py-2">Win Rate</th>
                            </tr>
                        </thead>
                        <tbody id="strategiesTable">
                            <tr>
                                <td colspan="5" class="text-center py-4 text-gray-400">No strategies loaded</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Patterns and Trades Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Recent Patterns -->
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">💡 Recent Patterns</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="border-b border-gray-700">
                                <th class="text-left py-2">Type</th>
                                <th class="text-left py-2">Symbol</th>
                                <th class="text-left py-2">Confidence</th>
                                <th class="text-left py-2">Return</th>
                            </tr>
                        </thead>
                        <tbody id="patternsTable">
                            <tr>
                                <td colspan="4" class="text-center py-4 text-gray-400">No patterns discovered</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- System Health -->
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">⚡ System Health</h3>
                <div class="space-y-4">
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Bot Status</span>
                        <span id="botStatus" class="text-green-400">🟢 Running</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Patterns Discovered</span>
                        <span id="patternsCount" class="text-blue-400">0</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Avg Daily Return</span>
                        <span id="avgDailyReturn" class="text-purple-400">0.00%</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Required Daily Return</span>
                        <span id="requiredDailyReturn" class="text-yellow-400">0.00%</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Days Running</span>
                        <span id="daysRunning" class="text-gray-300">0</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Recent Trades -->
        <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
            <h3 class="text-xl font-bold mb-4">📋 Recent Trades</h3>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b border-gray-700">
                            <th class="text-left py-2">Symbol</th>
                            <th class="text-left py-2">Side</th>
                            <th class="text-left py-2">Amount</th>
                            <th class="text-left py-2">Price</th>
                            <th class="text-left py-2">P&L</th>
                            <th class="text-left py-2">P&L %</th>
                            <th class="text-left py-2">Closed</th>
                        </tr>
                    </thead>
                    <tbody id="tradesTable">
                        <tr>
                            <td colspan="7" class="text-center py-4 text-gray-400">No trades executed</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        let ws;
        let equityChart, dailyChart;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = function(event) {
                document.getElementById('status').textContent = '🟢 Connected';
                document.getElementById('status').className = 'text-lg font-semibold text-green-400';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function(event) {
                document.getElementById('status').textContent = '🔴 Disconnected';
                document.getElementById('status').className = 'text-lg font-semibold text-red-400';
                // Reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateDashboard(data) {
            if (data.error) {
                console.error('Dashboard error:', data.error);
                return;
            }
            
            const overview = data.overview || {};
            
            // Update key metrics
            document.getElementById('equity').textContent = `$${(overview.equity || 200).toFixed(2)}`;
            document.getElementById('dailyPnl').textContent = `$${(overview.daily_pnl || 0).toFixed(2)}`;
            document.getElementById('winRate').textContent = `${((overview.win_rate || 0) * 100).toFixed(1)}%`;
            document.getElementById('targetProgress').textContent = `${((overview.equity || 200) / 1000000 * 100).toFixed(2)}%`;
            
            // Update ROI
            const roi = overview.roi_percent || 0;
            const roiElement = document.getElementById('roi');
            roiElement.textContent = `${roi >= 0 ? '+' : ''}${roi.toFixed(2)}%`;
            roiElement.className = roi >= 0 ? 'text-sm text-green-400' : 'text-sm text-red-400';
            
            // Update daily P&L color
            const dailyPnl = overview.daily_pnl || 0;
            const dailyPnlElement = document.getElementById('dailyPnl');
            dailyPnlElement.className = dailyPnl >= 0 ? 'text-3xl font-bold text-green-400' : 'text-3xl font-bold text-red-400';
            
            // Update other stats
            document.getElementById('tradesToday').textContent = `${overview.trades_today || 0} trades today`;
            document.getElementById('activeStrategies').textContent = `${overview.active_strategies || 0} active strategies`;
            document.getElementById('daysRemaining').textContent = `${overview.days_remaining || 90} days remaining`;
            
            // Update system health
            document.getElementById('patternsCount').textContent = overview.total_patterns || 0;
            document.getElementById('avgDailyReturn').textContent = `${(overview.daily_return_avg || 0).toFixed(2)}%`;
            document.getElementById('requiredDailyReturn').textContent = `${(overview.required_daily_return || 0).toFixed(2)}%`;
            document.getElementById('daysRunning').textContent = overview.days_running || 0;
            
            // Update timestamp
            document.getElementById('lastUpdate').textContent = `Last update: ${new Date().toLocaleTimeString()}`;
            
            // Update tables
            updatePositionsTable(data.positions || []);
            updateStrategiesTable(data.strategies || []);
            updatePatternsTable(data.patterns || []);
            updateTradesTable(data.trades || []);
        }
        
        function updatePositionsTable(positions) {
            const tbody = document.getElementById('positionsTable');
            if (positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-gray-400">No open positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(position => `
                <tr class="border-b border-gray-700">
                    <td class="py-2">${position.symbol}</td>
                    <td class="py-2">
                        <span class="px-2 py-1 rounded text-xs ${position.side === 'buy' ? 'bg-green-600' : 'bg-red-600'}">
                            ${position.side.toUpperCase()}
                        </span>
                    </td>
                    <td class="py-2">${position.amount.toFixed(4)}</td>
                    <td class="py-2">$${position.entry_price.toFixed(4)}</td>
                    <td class="py-2">$${position.current_price.toFixed(4)}</td>
                    <td class="py-2 ${position.pnl >= 0 ? 'text-green-400' : 'text-red-400'}">
                        $${position.pnl.toFixed(2)} (${(position.pnl_pct * 100).toFixed(2)}%)
                    </td>
                </tr>
            `).join('');
        }
        
        function updateStrategiesTable(strategies) {
            const tbody = document.getElementById('strategiesTable');
            if (strategies.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-gray-400">No strategies loaded</td></tr>';
                return;
            }
            
            tbody.innerHTML = strategies.slice(0, 10).map(strategy => `
                <tr class="border-b border-gray-700">
                    <td class="py-2">${strategy.name}</td>
                    <td class="py-2">
                        <span class="px-2 py-1 rounded text-xs ${getStatusColor(strategy.status)}">
                            ${strategy.status}
                        </span>
                    </td>
                    <td class="py-2">${strategy.total_trades}</td>
                    <td class="py-2 ${strategy.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}">
                        $${strategy.total_pnl.toFixed(2)}
                    </td>
                    <td class="py-2">${(strategy.win_rate * 100).toFixed(1)}%</td>
                </tr>
            `).join('');
        }
        
        function updatePatternsTable(patterns) {
            const tbody = document.getElementById('patternsTable');
            if (patterns.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-gray-400">No patterns discovered</td></tr>';
                return;
            }
            
            tbody.innerHTML = patterns.slice(0, 10).map(pattern => `
                <tr class="border-b border-gray-700">
                    <td class="py-2">${pattern.type}</td>
                    <td class="py-2">${pattern.symbol || 'Multi'}</td>
                    <td class="py-2">${(pattern.confidence * 100).toFixed(1)}%</td>
                    <td class="py-2 text-green-400">${(pattern.avg_return * 100).toFixed(2)}%</td>
                </tr>
            `).join('');
        }
        
        function updateTradesTable(trades) {
            const tbody = document.getElementById('tradesTable');
            if (trades.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-gray-400">No trades executed</td></tr>';
                return;
            }
            
            tbody.innerHTML = trades.slice(0, 10).map(trade => `
                <tr class="border-b border-gray-700">
                    <td class="py-2">${trade.symbol}</td>
                    <td class="py-2">
                        <span class="px-2 py-1 rounded text-xs ${trade.side === 'buy' ? 'bg-green-600' : 'bg-red-600'}">
                            ${trade.side.toUpperCase()}
                        </span>
                    </td>
                    <td class="py-2">${trade.amount.toFixed(4)}</td>
                    <td class="py-2">$${trade.price.toFixed(4)}</td>
                    <td class="py-2 ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}">
                        $${trade.pnl.toFixed(2)}
                    </td>
                    <td class="py-2 ${trade.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}">
                        ${(trade.pnl_pct * 100).toFixed(2)}%
                    </td>
                    <td class="py-2">${new Date(trade.closed_at).toLocaleString()}</td>
                </tr>
            `).join('');
        }
        
        function getStatusColor(status) {
            switch(status) {
                case 'active': return 'bg-green-600';
                case 'micro': return 'bg-blue-600';
                case 'paper': return 'bg-yellow-600';
                default: return 'bg-gray-600';
            }
        }
        
        function initCharts() {
            // Equity Chart
            const equityCtx = document.getElementById('equityChart').getContext('2d');
            equityChart = new Chart(equityCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Equity',
                        data: [],
                        borderColor: 'rgb(34, 197, 94)',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: { 
                            beginAtZero: false,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'white' }
                        },
                        x: { 
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'white' }
                        }
                    }
                }
            });
            
            // Daily P&L Chart
            const dailyCtx = document.getElementById('dailyChart').getContext('2d');
            dailyChart = new Chart(dailyCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Daily P&L',
                        data: [],
                        backgroundColor: function(context) {
                            return context.parsed.y >= 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)';
                        }
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: { 
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'white' }
                        },
                        x: { 
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'white' }
                        }
                    }
                }
            });
        }
        
        // Initialize everything
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            connectWebSocket();
        });
    </script>
</body>
</html>
        """

# FastAPI app instance
dashboard = TradingDashboard()
app = dashboard.app

if __name__ == "__main__":
    log.info("🌐 Starting v26meme Trading Dashboard...")
    log.info("📊 Dashboard will be available at: http://localhost:8000")
    log.info("🔗 Real-time WebSocket updates enabled")
    
    uvicorn.run(
        "web_dashboard:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
