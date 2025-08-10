#!/usr/bin/env python3
"""
v26meme Institutional Trading Dashboard
=======================================

Comprehensive institutional-grade real-time monitoring and analytics platform
combining all dashboard features with SimLab integration for professional
quantitative fund operations.

Features:
- Real-time WebSocket streaming with <100ms latency
- SimLab simulation engine integration
- Professional institutional analytics
- Risk management and compliance monitoring
- Multi-device responsive design
- Comprehensive API suite
"""

import asyncio
import json
import sqlite3
import os
import math
import time
from datetime import datetime
from typing import Dict, List
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class InstitutionalDashboard:
    """Institutional-grade trading dashboard with comprehensive analytics"""
    
    def __init__(self, db_path: str = "v26meme.db"):
        self.db_path = db_path
        self.active_connections: List[WebSocket] = []
        self.app = FastAPI(
            title="v26meme Institutional Trading Dashboard",
            version="2.0.0",
            description="Professional quantitative fund monitoring platform"
        )
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
            """Serve the main institutional dashboard"""
            return self.get_institutional_dashboard_html()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time institutional data streaming"""
            await self.connect(websocket)
            try:
                while True:
                    # Send comprehensive updates every second
                    data = await self.get_institutional_dashboard_data()
                    await websocket.send_text(json.dumps(data))
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                self.disconnect(websocket)
        
        # Core institutional APIs
        @self.app.get("/api/institutional-overview")
        async def get_institutional_overview():
            """Get comprehensive institutional metrics"""
            return await self.get_institutional_metrics()
        
        @self.app.get("/api/portfolio")
        async def get_portfolio():
            """Get real-time portfolio metrics"""
            return await self.get_portfolio_analytics()
        
        @self.app.get("/api/risk")
        async def get_risk():
            """Get comprehensive risk metrics"""
            return await self.get_risk_analytics()
        
        @self.app.get("/api/simlab")
        async def get_simlab():
            """Get SimLab simulation analytics"""
            return await self.get_simlab_metrics()
        
        @self.app.get("/api/positions")
        async def get_positions():
            """Get current positions with institutional details"""
            return await self.get_institutional_positions()
        
        @self.app.get("/api/strategies")
        async def get_strategies():
            """Get strategy performance with advanced metrics"""
            return await self.get_institutional_strategies()
        
        @self.app.get("/api/performance-attribution")
        async def get_performance_attribution():
            """Get detailed performance attribution analysis"""
            return await self.get_performance_attribution()
        
        @self.app.get("/api/patterns")
        async def get_patterns():
            """Get discovered patterns with institutional analysis"""
            return await self.get_institutional_patterns()
        
        @self.app.get("/api/trades")
        async def get_trades():
            """Get trade execution analysis"""
            return await self.get_institutional_trades()
        
        @self.app.get("/api/equity-curve")
        async def get_equity_curve():
            """Get equity curve with institutional metrics"""
            return await self.get_institutional_equity_curve()
        
        @self.app.get("/health")
        async def health():
            """Enhanced health check with system metrics"""
            return await self.get_system_health()
    
    async def connect(self, websocket: WebSocket):
        """Connect new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info(f"🔗 Institutional client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket client"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        log.info(f"📴 Institutional client disconnected. Total: {len(self.active_connections)}")
    
    def get_db_connection(self):
        """Get database connection with row factory"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            class StdevFunc:
                def __init__(self):
                    self.M = 0.0
                    self.S = 0.0
                    self.k = 0

                def step(self, value):
                    if value is None:
                        return
                    t = value - self.M
                    self.k += 1
                    self.M += t / self.k
                    self.S += t * (value - self.M)

                def finalize(self):
                    if self.k < 2:
                        return None
                    return math.sqrt(self.S / (self.k - 1))

            conn.create_aggregate("stdev", 1, StdevFunc)  # type: ignore  # type: ignore  # type: ignore  # type: ignore  # type: ignore
            return conn
        except Exception as e:
            log.error(f"Database connection error: {e}")
            return None
    
    async def get_institutional_dashboard_data(self) -> Dict:
        """Get comprehensive institutional dashboard data"""
        try:
            # Parallel data gathering for performance
            institutional_metrics, portfolio_analytics, risk_analytics, simlab_metrics = await asyncio.gather(
                self.get_institutional_metrics(),
                self.get_portfolio_analytics(),
                self.get_risk_analytics(),
                self.get_simlab_metrics()
            )
            
            positions, strategies, patterns, trades = await asyncio.gather(
                self.get_institutional_positions(),
                self.get_institutional_strategies(),
                self.get_institutional_patterns(),
                self.get_institutional_trades()
            )
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "institutional_metrics": institutional_metrics,
                "portfolio": portfolio_analytics,
                "risk": risk_analytics,
                "simlab": simlab_metrics,
                "positions": positions,
                "strategies": strategies,
                "patterns": patterns,
                "trades": trades,
                "status": "operational"
            }
        except Exception as e:
            log.error(f"Error getting institutional dashboard data: {e}")
            return {"error": str(e), "status": "error"}
    
    async def get_institutional_metrics(self) -> Dict:
        """Get comprehensive institutional performance metrics"""
        conn = self.get_db_connection()
        if not conn:
            return {"error": "Database connection failed"}
        
        try:
            cursor = conn.cursor()
            
            # Core metrics
            cursor.execute("""
                SELECT equity, cash, total_pnl, daily_pnl, win_rate, last_update
                FROM system_state ORDER BY last_update DESC LIMIT 1
            """)
            state = cursor.fetchone()
            
            # Portfolio analytics
            initial_capital = float(os.getenv("INITIAL_BALANCE", 200.0))
            current_equity = state['equity'] if state else initial_capital
            total_return = ((current_equity / initial_capital) - 1) * 100
            
            # Risk-adjusted metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    AVG(pnl_pct) as avg_return,
                    STDEV(pnl_pct) as return_volatility,
                    MAX(pnl_pct) as max_return,
                    MIN(pnl_pct) as max_loss
                FROM trades WHERE pnl_pct IS NOT NULL
            """)
            trade_stats = cursor.fetchone()
            
            # Calculate Sharpe ratio (approximation)
            if trade_stats and trade_stats['return_volatility'] and trade_stats['return_volatility'] > 0:
                sharpe_ratio = (trade_stats['avg_return'] or 0) / trade_stats['return_volatility'] * math.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calculate maximum drawdown
            cursor.execute("""
                SELECT equity FROM system_state 
                ORDER BY last_update ASC
            """)
            equity_history = [row['equity'] for row in cursor.fetchall()]
            max_dd = self.calculate_max_drawdown(equity_history)
            
            # Time-based metrics
            start_date = datetime(2025, 8, 6)  # Adjust based on actual start
            days_running = (datetime.utcnow() - start_date).days + 1
            annualized_return = total_return * (365 / days_running) if days_running > 0 else 0
            
            return {
                "current_equity": current_equity,
                "total_return_pct": total_return,
                "annualized_return_pct": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": max_dd,
                "daily_pnl": state['daily_pnl'] if state else 0,
                "win_rate": state['win_rate'] if state else 0,
                "total_trades": trade_stats['total_trades'] if trade_stats else 0,
                "volatility": trade_stats['return_volatility'] if trade_stats else 0,
                "days_running": days_running,
                "last_update": state['last_update'] if state else datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            log.error(f"Error getting institutional metrics: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if not equity_values or len(equity_values) < 2:
            return 0.0
        
        peak = equity_values[0]
        max_dd = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    async def get_portfolio_analytics(self) -> Dict:
        """Get comprehensive portfolio analytics"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            # Position analysis
            cursor.execute("""
                SELECT 
                    COUNT(*) as position_count,
                    SUM(pnl) as total_position_pnl,
                    AVG(pnl) as avg_position_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_positions,
                    SUM(amount * current_price) as total_exposure
                FROM current_positions
            """)
            position_stats = cursor.fetchone()
            
            # Sector/symbol exposure
            cursor.execute("""
                SELECT symbol, SUM(amount * current_price) as exposure
                FROM current_positions
                GROUP BY symbol
                ORDER BY exposure DESC
                LIMIT 10
            """)
            exposures = [dict(row) for row in cursor.fetchall()]
            
            return {
                "position_count": position_stats['position_count'] if position_stats else 0,
                "total_exposure": position_stats['total_exposure'] if position_stats else 0,
                "position_pnl": position_stats['total_position_pnl'] if position_stats else 0,
                "winning_positions": position_stats['winning_positions'] if position_stats else 0,
                "top_exposures": exposures
            }
        
        except Exception as e:
            log.error(f"Error getting portfolio analytics: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    async def get_risk_analytics(self) -> Dict:
        """Get comprehensive risk management metrics"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            # Risk metrics from recent trades
            cursor.execute("""
                SELECT 
                    STDEV(pnl_pct) * SQRT(252) as annualized_vol,
                    PERCENTILE(pnl_pct, 5) as var_95,
                    PERCENTILE(pnl_pct, 1) as var_99,
                    AVG(pnl_pct) as avg_return
                FROM trades 
                WHERE pnl_pct IS NOT NULL AND closed_at >= datetime('now', '-30 days')
            """)
            risk_stats = cursor.fetchone()
            
            # Concentration risk
            cursor.execute("""
                SELECT 
                    symbol,
                    SUM(amount * current_price) as exposure,
                    COUNT(*) as position_count
                FROM current_positions
                GROUP BY symbol
                ORDER BY exposure DESC
                LIMIT 5
            """)
            concentrations = [dict(row) for row in cursor.fetchall()]
            
            return {
                "annualized_volatility": risk_stats['annualized_vol'] if risk_stats else 0,
                "var_95": risk_stats['var_95'] if risk_stats else 0,
                "var_99": risk_stats['var_99'] if risk_stats else 0,
                "average_return": risk_stats['avg_return'] if risk_stats else 0,
                "concentration_risk": concentrations
            }
        
        except Exception as e:
            log.error(f"Error getting risk analytics: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    async def get_simlab_metrics(self) -> Dict:
        """Get comprehensive SimLab simulation metrics"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            # Check if SimLab tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sim_runs'")
            if not cursor.fetchone():
                return {"error": "SimLab tables not found"}
            
            # Simulation summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as completed_runs,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running_runs,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_runs
                FROM sim_runs
            """)
            run_stats = cursor.fetchone()
            
            # Trade analysis
            cursor.execute("SELECT COUNT(*) as total_sim_trades FROM sim_trades")
            trade_stats = cursor.fetchone()
            
            # Top performing strategies from simulations
            cursor.execute("""
                SELECT 
                    strategy_name,
                    COUNT(*) as simulations,
                    AVG(sharpe_like) as avg_sharpe,
                    AVG(win_rate) as avg_win_rate,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_pnl
                FROM sim_metrics 
                WHERE strategy_name IS NOT NULL
                GROUP BY strategy_name
                ORDER BY avg_sharpe DESC
                LIMIT 10
            """)
            top_strategies = [dict(row) for row in cursor.fetchall()]
            
            # Recent simulation activity
            cursor.execute("""
                SELECT COUNT(*) as recent_runs
                FROM sim_runs 
                WHERE created_at >= datetime('now', '-24 hours')
            """)
            recent_activity = cursor.fetchone()
            
            return {
                "total_simulations": run_stats['total_runs'] if run_stats else 0,
                "completed_simulations": run_stats['completed_runs'] if run_stats else 0,
                "running_simulations": run_stats['running_runs'] if run_stats else 0,
                "error_simulations": run_stats['error_runs'] if run_stats else 0,
                "total_simulation_trades": trade_stats['total_sim_trades'] if trade_stats else 0,
                "recent_activity_24h": recent_activity['recent_runs'] if recent_activity else 0,
                "top_performing_strategies": top_strategies
            }
        
        except Exception as e:
            log.error(f"Error getting SimLab metrics: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    async def get_institutional_positions(self) -> List[Dict]:
        """Get current positions with institutional analytics"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, side, amount, entry_price, current_price, 
                       pnl, pnl_pct, opened_at, strategy_name,
                       (amount * current_price) as notional_value,
                       (pnl / (amount * entry_price)) as return_on_capital
                FROM current_positions 
                ORDER BY ABS(pnl) DESC
            """)
            
            positions = []
            for row in cursor.fetchall():
                positions.append({
                    "symbol": row['symbol'],
                    "side": row['side'],
                    "amount": row['amount'],
                    "entry_price": row['entry_price'],
                    "current_price": row['current_price'],
                    "pnl": row['pnl'],
                    "pnl_pct": row['pnl_pct'],
                    "notional_value": row['notional_value'],
                    "return_on_capital": row['return_on_capital'],
                    "opened_at": row['opened_at'],
                    "strategy_name": row['strategy_name']
                })
            
            return positions
        
        except Exception as e:
            log.error(f"Error getting institutional positions: {e}")
            return []
        finally:
            conn.close()
    
    async def get_institutional_strategies(self) -> List[Dict]:
        """Get strategy performance with institutional metrics"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, status, total_trades, total_pnl, win_rate,
                       avg_trade_size, last_trade, confidence_avg,
                       (total_pnl / NULLIF(total_trades, 0)) as avg_pnl_per_trade,
                       (total_pnl / NULLIF(avg_trade_size * total_trades, 0)) as return_on_investment
                FROM strategies 
                ORDER BY total_pnl DESC
                LIMIT 20
            """)
            
            strategies = []
            for row in cursor.fetchall():
                strategies.append({
                    "name": row['name'],
                    "status": row['status'],
                    "total_trades": row['total_trades'],
                    "total_pnl": row['total_pnl'],
                    "win_rate": row['win_rate'],
                    "avg_pnl_per_trade": row['avg_pnl_per_trade'],
                    "return_on_investment": row['return_on_investment'],
                    "confidence_avg": row['confidence_avg'],
                    "last_trade": row['last_trade']
                })
            
            return strategies
        
        except Exception as e:
            log.error(f"Error getting institutional strategies: {e}")
            return []
        finally:
            conn.close()
    
    async def get_performance_attribution(self) -> Dict:
        """Get detailed performance attribution analysis"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            # Strategy attribution
            cursor.execute("""
                SELECT 
                    strategy_name,
                    SUM(pnl) as strategy_pnl,
                    COUNT(*) as strategy_trades,
                    AVG(pnl) as avg_trade_pnl
                FROM trades 
                WHERE strategy_name IS NOT NULL
                GROUP BY strategy_name
                ORDER BY strategy_pnl DESC
            """)
            strategy_attribution = [dict(row) for row in cursor.fetchall()]
            
            # Symbol attribution
            cursor.execute("""
                SELECT 
                    symbol,
                    SUM(pnl) as symbol_pnl,
                    COUNT(*) as symbol_trades,
                    AVG(pnl) as avg_symbol_pnl
                FROM trades 
                GROUP BY symbol
                ORDER BY symbol_pnl DESC
                LIMIT 10
            """)
            symbol_attribution = [dict(row) for row in cursor.fetchall()]
            
            return {
                "strategy_attribution": strategy_attribution,
                "symbol_attribution": symbol_attribution
            }
        
        except Exception as e:
            log.error(f"Error getting performance attribution: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    async def get_institutional_patterns(self) -> List[Dict]:
        """Get discovered patterns with institutional analysis"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pattern_type, symbol, confidence, detected_at,
                       avg_return, frequency, last_seen
                FROM patterns 
                WHERE detected_at >= datetime('now', '-7 days')
                ORDER BY confidence DESC
                LIMIT 20
            """)
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    "pattern_type": row['pattern_type'],
                    "symbol": row['symbol'],
                    "confidence": row['confidence'],
                    "avg_return": row['avg_return'],
                    "frequency": row['frequency'],
                    "detected_at": row['detected_at'],
                    "last_seen": row['last_seen']
                })
            
            return patterns
        
        except Exception as e:
            log.error(f"Error getting institutional patterns: {e}")
            return []
        finally:
            conn.close()
    
    async def get_institutional_trades(self) -> List[Dict]:
        """Get recent trades with execution analysis"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, side, amount, price, pnl, pnl_pct,
                       opened_at, closed_at, strategy_name,
                       (amount * price) as notional,
                       JULIANDAY(closed_at) - JULIANDAY(opened_at) as duration_days
                FROM trades 
                ORDER BY closed_at DESC 
                LIMIT 50
            """)
            
            trades = []
            for row in cursor.fetchall():
                trades.append({
                    "symbol": row['symbol'],
                    "side": row['side'],
                    "amount": row['amount'],
                    "price": row['price'],
                    "pnl": row['pnl'],
                    "pnl_pct": row['pnl_pct'],
                    "notional": row['notional'],
                    "duration_days": row['duration_days'],
                    "opened_at": row['opened_at'],
                    "closed_at": row['closed_at'],
                    "strategy_name": row['strategy_name']
                })
            
            return trades
        
        except Exception as e:
            log.error(f"Error getting institutional trades: {e}")
            return []
        finally:
            conn.close()
    
    async def get_institutional_equity_curve(self) -> Dict:
        """Get equity curve with institutional performance metrics"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT last_update, equity, total_pnl, daily_pnl
                FROM system_state 
                ORDER BY last_update ASC
            """)
            
            equity_data = []
            pnl_data = []
            timestamps = []
            
            for row in cursor.fetchall():
                timestamps.append(row['last_update'])
                equity_data.append(row['equity'])
                pnl_data.append(row['total_pnl'])
            
            # Calculate rolling metrics
            returns = []
            if len(equity_data) > 1:
                for i in range(1, len(equity_data)):
                    ret = (equity_data[i] / equity_data[i-1] - 1) * 100
                    returns.append(ret)
            
            return {
                "timestamps": timestamps,
                "equity_values": equity_data,
                "pnl_values": pnl_data,
                "returns": returns
            }
        
        except Exception as e:
            log.error(f"Error getting institutional equity curve: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    async def get_system_health(self) -> Dict:
        """Get comprehensive system health metrics"""
        conn = self.get_db_connection()
        if not conn:
            return {"status": "error", "message": "Database unavailable"}
        
        try:
            cursor = conn.cursor()
            
            # Check data freshness
            cursor.execute("SELECT MAX(last_update) as latest FROM system_state")
            latest_update = cursor.fetchone()
            
            if latest_update and latest_update['latest']:
                last_update_time = datetime.fromisoformat(latest_update['latest'].replace('Z', '+00:00'))
                data_age_seconds = (datetime.utcnow() - last_update_time).total_seconds()
            else:
                data_age_seconds = float('inf')
            
            # System metrics
            health_status = {
                "status": "healthy" if data_age_seconds < 300 else "warning",  # 5 minutes threshold
                "timestamp": datetime.utcnow().isoformat(),
                "data_age_seconds": data_age_seconds,
                "database_status": "connected",
                "active_websockets": len(self.active_connections),
                "system_uptime": "operational"
            }
            
            return health_status
        
        except Exception as e:
            log.error(f"Error getting system health: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            conn.close()
    
    def get_institutional_dashboard_html(self) -> str:
        """Generate the comprehensive institutional dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>v26meme Institutional Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .institutional-gradient { background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 25%, #7c2d12 50%, #dc2626 75%, #991b1b 100%); }
        .card { backdrop-filter: blur(10px); transition: transform 0.2s; }
        .card:hover { transform: translateY(-2px); }
        .metric-positive { color: #10b981; }
        .metric-negative { color: #ef4444; }
        .metric-neutral { color: #6b7280; }
        .pulse-green { animation: pulse-green 2s infinite; }
        .pulse-red { animation: pulse-red 2s infinite; }
        @keyframes pulse-green { 0%, 100% { background-color: rgb(34, 197, 94); } 50% { background-color: rgb(22, 163, 74); } }
        @keyframes pulse-red { 0%, 100% { background-color: rgb(239, 68, 68); } 50% { background-color: rgb(220, 38, 38); } }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .table-row:hover { background-color: rgba(255, 255, 255, 0.05); }
    </style>
</head>
<body class="bg-gray-900 text-white">
    <!-- Institutional Header -->
    <div class="institutional-gradient p-6 shadow-2xl">
        <div class="max-w-8xl mx-auto flex justify-between items-center">
            <div>
                <h1 class="text-4xl font-bold">🏛️ v26meme Institutional Dashboard</h1>
                <p class="text-xl opacity-90">Professional Quantitative Fund Operations • Real-time Analytics</p>
            </div>
            <div class="text-right">
                <div id="systemStatus" class="text-xl font-semibold">🔴 Initializing...</div>
                <div id="lastUpdate" class="text-sm opacity-75">Connecting to institutional systems...</div>
                <div id="wsStatus" class="text-sm">WebSocket: Connecting</div>
            </div>
        </div>
    </div>

    <!-- Main Dashboard -->
    <div class="max-w-8xl mx-auto p-6 space-y-6">
        
        <!-- Institutional KPIs -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
            <div class="bg-gray-800 card rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Portfolio NAV</div>
                <div id="portfolioNav" class="text-2xl font-bold text-green-400">$200.00</div>
                <div id="totalReturn" class="text-xs text-gray-300">+0.00%</div>
            </div>
            
            <div class="bg-gray-800 card rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Sharpe Ratio</div>
                <div id="sharpeRatio" class="text-2xl font-bold text-blue-400">0.00</div>
                <div class="text-xs text-gray-300">Risk-Adjusted</div>
            </div>
            
            <div class="bg-gray-800 card rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Max Drawdown</div>
                <div id="maxDrawdown" class="text-2xl font-bold text-red-400">0.00%</div>
                <div class="text-xs text-gray-300">Risk Control</div>
            </div>
            
            <div class="bg-gray-800 card rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Daily P&L</div>
                <div id="dailyPnl" class="text-2xl font-bold">$0.00</div>
                <div class="text-xs text-gray-300">Today's Performance</div>
            </div>
            
            <div class="bg-gray-800 card rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Win Rate</div>
                <div id="winRate" class="text-2xl font-bold text-purple-400">0%</div>
                <div class="text-xs text-gray-300">Success Rate</div>
            </div>
            
            <div class="bg-gray-800 card rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Strategies Active</div>
                <div id="activeStrategies" class="text-2xl font-bold text-cyan-400">0</div>
                <div class="text-xs text-gray-300">Diversification</div>
            </div>
        </div>

        <!-- SimLab Integration Panel -->
        <div class="bg-gradient-to-r from-purple-900 to-blue-900 rounded-lg p-6 border-2 border-purple-500">
            <h2 class="text-2xl font-bold mb-4">🧪 SimLab: Institutional Simulation Engine</h2>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="bg-black bg-opacity-30 rounded-lg p-4">
                    <div class="text-sm text-purple-200 mb-1">Total Simulations</div>
                    <div id="totalSimulations" class="text-2xl font-bold text-white">0</div>
                    <div class="text-xs text-purple-300">Historical Backtests</div>
                </div>
                <div class="bg-black bg-opacity-30 rounded-lg p-4">
                    <div class="text-sm text-purple-200 mb-1">Simulation Trades</div>
                    <div id="simulationTrades" class="text-2xl font-bold text-white">0</div>
                    <div class="text-xs text-purple-300">Analyzed Positions</div>
                </div>
                <div class="bg-black bg-opacity-30 rounded-lg p-4">
                    <div class="text-sm text-purple-200 mb-1">Success Rate</div>
                    <div id="simulationSuccess" class="text-2xl font-bold text-white">100%</div>
                    <div class="text-xs text-purple-300">Completion Rate</div>
                </div>
                <div class="bg-black bg-opacity-30 rounded-lg p-4">
                    <div class="text-sm text-purple-200 mb-1">Recent Activity</div>
                    <div id="recentActivity" class="text-2xl font-bold text-white">0</div>
                    <div class="text-xs text-purple-300">Last 24h</div>
                </div>
            </div>
        </div>

        <!-- Charts Section -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">📊 Equity Curve</h3>
                <canvas id="equityChart" width="400" height="200"></canvas>
            </div>
            
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">🎯 Risk Metrics</h3>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-400">VaR (95%)</span>
                        <span id="var95" class="text-red-400">0.00%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">VaR (99%)</span>
                        <span id="var99" class="text-red-400">0.00%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Volatility (Ann.)</span>
                        <span id="annualizedVol" class="text-yellow-400">0.00%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Beta</span>
                        <span id="beta" class="text-blue-400">0.00</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Data Tables -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Positions -->
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">💼 Live Positions</h3>
                <div class="overflow-y-auto max-h-80">
                    <table class="w-full text-sm">
                        <thead class="sticky top-0 bg-gray-700">
                            <tr>
                                <th class="text-left p-2">Symbol</th>
                                <th class="text-left p-2">Side</th>
                                <th class="text-left p-2">P&L</th>
                                <th class="text-left p-2">ROI</th>
                            </tr>
                        </thead>
                        <tbody id="positionsTable">
                            <tr><td colspan="4" class="text-center py-4 text-gray-400">No positions</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Strategies -->
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">🎯 Strategy Performance</h3>
                <div class="overflow-y-auto max-h-80">
                    <table class="w-full text-sm">
                        <thead class="sticky top-0 bg-gray-700">
                            <tr>
                                <th class="text-left p-2">Strategy</th>
                                <th class="text-left p-2">Trades</th>
                                <th class="text-left p-2">P&L</th>
                                <th class="text-left p-2">Win%</th>
                            </tr>
                        </thead>
                        <tbody id="strategiesTable">
                            <tr><td colspan="4" class="text-center py-4 text-gray-400">No strategies</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Discovered Patterns -->
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">💡 Discovered Patterns</h3>
                <div class="overflow-y-auto max-h-80">
                    <table class="w-full text-sm">
                        <thead class="sticky top-0 bg-gray-700">
                            <tr>
                                <th class="text-left p-2">Type</th>
                                <th class="text-left p-2">Symbol</th>
                                <th class="text-left p-2">Confidence</th>
                                <th class="text-left p-2">Avg. Return</th>
                            </tr>
                        </thead>
                        <tbody id="patternsTable">
                            <tr><td colspan="4" class="text-center py-4 text-gray-400">No patterns discovered</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- SimLab Top Performers -->
            <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">🏆 SimLab Top Performers</h3>
                <div class="overflow-y-auto max-h-80">
                    <table class="w-full text-sm">
                        <thead class="sticky top-0 bg-gray-700">
                            <tr>
                                <th class="text-left p-2">Strategy</th>
                                <th class="text-left p-2">Sharpe</th>
                                <th class="text-left p-2">Sims</th>
                                <th class="text-left p-2">Win%</th>
                            </tr>
                        </thead>
                        <tbody id="simlabStrategiesTable">
                            <tr><td colspan="4" class="text-center py-4 text-gray-400">No simulation data</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Recent Activity -->
        <div class="bg-gray-800 card rounded-lg p-6 border border-gray-700">
            <h3 class="text-xl font-bold mb-4">📋 Recent Trade Execution</h3>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b border-gray-700">
                            <th class="text-left p-2">Symbol</th>
                            <th class="text-left p-2">Side</th>
                            <th class="text-left p-2">Amount</th>
                            <th class="text-left p-2">Price</th>
                            <th class="text-left p-2">P&L</th>
                            <th class="text-left p-2">P&L %</th>
                            <th class="text-left p-2">Duration</th>
                            <th class="text-left p-2">Strategy</th>
                            <th class="text-left p-2">Closed</th>
                        </tr>
                    </thead>
                    <tbody id="tradesTable">
                        <tr><td colspan="9" class="text-center py-4 text-gray-400">No recent trades</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let ws;
        let equityChart;
        let updateInterval;
        
        // Utility functions
        function formatNumber(num, decimals = 2) {
            if (num === null || num === undefined) return 'N/A';
            if (Math.abs(num) >= 1000000) return (num/1000000).toFixed(1) + 'M';
            if (Math.abs(num) >= 1000) return (num/1000).toFixed(1) + 'K';
            return parseFloat(num).toFixed(decimals);
        }
        
        function formatPercent(num) {
            if (num === null || num === undefined) return 'N/A';
            return (num).toFixed(2) + '%';
        }
        
        function getColorClass(value, threshold = 0) {
            if (value === null || value === undefined) return '';
            return value > threshold ? 'metric-positive' : value < threshold ? 'metric-negative' : 'metric-neutral';
        }
        
        function formatDuration(days) {
            if (days < 1) return `${(days * 24).toFixed(1)}h`;
            return `${days.toFixed(1)}d`;
        }
        
        // Update dashboard with institutional data
        function updateInstitutionalDashboard(data) {
            if (data.error) {
                console.error('Dashboard error:', data.error);
                return;
            }
            
            const metrics = data.institutional_metrics || {};
            const portfolio = data.portfolio || {};
            const risk = data.risk || {};
            const simlab = data.simlab || {};
            
            // Update institutional KPIs
            document.getElementById('portfolioNav').textContent = `$${formatNumber(metrics.current_equity || 200)}`;
            document.getElementById('portfolioNav').className = `text-2xl font-bold ${getColorClass(metrics.current_equity - 200)}`;
            
            document.getElementById('totalReturn').textContent = `${formatPercent(metrics.total_return_pct || 0)}`;
            document.getElementById('totalReturn').className = `text-xs ${getColorClass(metrics.total_return_pct)}`;
            
            document.getElementById('sharpeRatio').textContent = formatNumber(metrics.sharpe_ratio || 0);
            document.getElementById('sharpeRatio').className = `text-2xl font-bold ${getColorClass(metrics.sharpe_ratio - 1)}`;
            
            document.getElementById('maxDrawdown').textContent = formatPercent(metrics.max_drawdown_pct || 0);
            
            document.getElementById('dailyPnl').textContent = `$${formatNumber(metrics.daily_pnl || 0)}`;
            document.getElementById('dailyPnl').className = `text-2xl font-bold ${getColorClass(metrics.daily_pnl)}`;
            
            document.getElementById('winRate').textContent = formatPercent((metrics.win_rate || 0) * 100);
            
            // Update SimLab metrics
            document.getElementById('totalSimulations').textContent = formatNumber(simlab.total_simulations || 0, 0);
            document.getElementById('simulationTrades').textContent = formatNumber(simlab.total_simulation_trades || 0, 0);
            document.getElementById('recentActivity').textContent = formatNumber(simlab.recent_activity_24h || 0, 0);
            
            const completionRate = simlab.total_simulations > 0 ? 
                (simlab.completed_simulations / simlab.total_simulations * 100) : 100;
            document.getElementById('simulationSuccess').textContent = formatPercent(completionRate);
            
            // Update risk metrics
            document.getElementById('var95').textContent = formatPercent(risk.var_95 || 0);
            document.getElementById('var99').textContent = formatPercent(risk.var_99 || 0);
            document.getElementById('annualizedVol').textContent = formatPercent(risk.annualized_volatility || 0);
            
            // Update positions table
            updatePositionsTable(data.positions || []);
            
            // Update strategies table
            updateStrategiesTable(data.strategies || []);
            
            // Update SimLab strategies table
            updateSimlabStrategiesTable(simlab.top_performing_strategies || []);
            
            // Update patterns table
            updatePatternsTable(data.patterns || []);
            
            // Update trades table
            updateTradesTable(data.trades || []);
            
            // Update status
            document.getElementById('systemStatus').textContent = '🟢 Institutional Systems Operational';
            document.getElementById('systemStatus').className = 'text-xl font-semibold text-green-400';
            document.getElementById('lastUpdate').textContent = `Last update: ${new Date().toLocaleTimeString()}`;
        }
        
        function updatePositionsTable(positions) {
            const tbody = document.getElementById('positionsTable');
            if (positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-gray-400">No positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.slice(0, 10).map(pos => `
                <tr class="table-row border-b border-gray-700">
                    <td class="p-2">${pos.symbol}</td>
                    <td class="p-2">
                        <span class="px-2 py-1 rounded text-xs ${pos.side === 'buy' ? 'bg-green-600' : 'bg-red-600'}">
                            ${pos.side.toUpperCase()}
                        </span>
                    </td>
                    <td class="p-2 ${getColorClass(pos.pnl)}">$${formatNumber(pos.pnl)}</td>
                    <td class="p-2 ${getColorClass(pos.return_on_capital)}">${formatPercent(pos.return_on_capital * 100)}</td>
                </tr>
            `).join('');
        }
        
        function updateStrategiesTable(strategies) {
            const tbody = document.getElementById('strategiesTable');
            if (strategies.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-gray-400">No strategies</td></tr>';
                return;
            }
            
            tbody.innerHTML = strategies.slice(0, 10).map(strat => `
                <tr class="table-row border-b border-gray-700">
                    <td class="p-2">${strat.name || 'Unknown'}</td>
                    <td class="p-2">${strat.total_trades || 0}</td>
                    <td class="p-2 ${getColorClass(strat.total_pnl)}">$${formatNumber(strat.total_pnl)}</td>
                    <td class="p-2 ${getColorClass((strat.win_rate || 0) - 0.5)}">${formatPercent((strat.win_rate || 0) * 100)}</td>
                </tr>
            `).join('');
        }
        
        function updateSimlabStrategiesTable(simStrategies) {
            const tableBody = document.getElementById('simlabStrategiesTable');
            if (simStrategies.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-gray-400">No simulation data</td></tr>';
                return;
            }
            
            tableBody.innerHTML = simStrategies.slice(0, 10).map(strat => `
                <tr class="table-row border-b border-gray-700">
                    <td class="p-2">${(strat.strategy_name || 'Unknown').substring(0, 12)}...</td>
                    <td class="p-2 ${getColorClass(strat.avg_sharpe - 1)}">${formatNumber(strat.avg_sharpe)}</td>
                    <td class="p-2">${strat.simulations || 0}</td>
                    <td class="p-2 ${getColorClass((strat.avg_win_rate || 0) - 0.5)}">${formatPercent((strat.avg_win_rate || 0) * 100)}</td>
                </tr>
            `).join('');
        }

        function updatePatternsTable(patterns) {
            const tableBody = document.getElementById('patternsTable');
            if (!patterns || patterns.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-gray-400">No patterns discovered</td></tr>';
                return;
            }
            tableBody.innerHTML = patterns.map(p => `
                <tr class="table-row">
                    <td class="p-2">${p.pattern_type}</td>
                    <td class="p-2">${p.symbol || 'Multi'}</td>
                    <td class="p-2">${formatPercent(p.confidence, 1)}</td>
                    <td class="p-2 ${formatSign(p.avg_return)}">${formatPercent(p.avg_return, 2)}</td>
                </tr>
            `).join('');
        }

        function updateTradesTable(trades) {
            const tableBody = document.getElementById('tradesTable');
            if (!trades || trades.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="9" class="text-center py-4 text-gray-400">No recent trades</td></tr>';
                return;
            }
            
            tableBody.innerHTML = trades.slice(0, 20).map(t => `
                <tr class="table-row">
                    <td class="p-2">${t.symbol}</td>
                    <td class="p-2 ${t.side === 'buy' ? 'text-green-400' : 'text-red-400'}">${t.side.toUpperCase()}</td>
                    <td class="p-2 ${formatSign(t.pnl)}">${formatNumber(t.pnl, 2)}</td>
                    <td class="p-2">${t.strategy_name}</td>
                    <td class="p-2">${new Date(t.closed_at).toLocaleTimeString()}</td>
                </tr>
            `).join('');
        }

        // Initial load
        document.addEventListener('DOMContentLoaded', function() {
            initChart();
            connectWebSocket();
            
            // Fallback polling if WebSocket fails
            setTimeout(() => {
                if (ws.readyState !== WebSocket.OPEN) {
                    console.log('WebSocket failed, using polling fallback');
                    updateInterval = setInterval(async () => {
                        try {
                            const response = await fetch('/api/institutional-overview');
                            const data = await response.json();
                            updateInstitutionalDashboard({institutional_metrics: data});
                        } catch (error) {
                            console.error('Polling error:', error);
                        }
                    }, 5000);
                }
            }, 5000);
        });
    </script>
</body>
</html>
        """

# Create FastAPI app instance
dashboard = InstitutionalDashboard()
app = dashboard.app

def wait_for_db(db_path: str):
    """Wait for the database to be created and initialized."""
    log.info("⏳ Waiting for database to be initialized by the main bot...")
    while True:
        try:
            if os.path.exists(db_path):
                # Use a timeout and read-only mode to avoid locking issues if the bot is writing
                conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=1.0)
                cursor = conn.cursor()
                # Check if a key table exists (e.g., system_state)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_state'")
                if cursor.fetchone():
                    log.info("✅ Database and key tables are ready.")
                    conn.close()
                    break
                conn.close()
                log.info("DB file found, but key tables not yet created. Waiting...")
            else:
                log.info("Database file not found. Waiting for main bot to create it...")
        except sqlite3.OperationalError as e:
            # This can happen if the DB is locked, which is fine. We just wait.
            log.info(f"DB is busy or not ready yet, waiting... Error: {e}")
        except Exception as e:
            log.error(f"An unexpected error occurred while waiting for DB: {e}")
        
        time.sleep(5)


if __name__ == "__main__":
    db_file = os.getenv("DB_PATH", "v26meme.db")
    wait_for_db(db_file)

    log.info("🏛️ Starting v26meme Institutional Trading Dashboard...")
    log.info("📊 Professional quantitative fund interface initializing...")
    log.info("🚀 Dashboard will be available at: http://localhost:8080")
    
    # Start the server
    port = int(os.getenv("WEB_PORT", 8080))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
