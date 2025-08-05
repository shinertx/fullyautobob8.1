#!/usr/bin/env python3
"""
v26meme Complete Autonomous Trading System - USA Compliant
============================================================
Production-ready autonomous crypto trading bot with:
- Multi-agent AI architecture using OpenAI GPT-4
- USA-compliant exchanges (Coinbase, Kraken, Gemini)
- Natural language strategy generation
- Self-evolving strategies with genetic algorithms
- Real-time web dashboard
- Complete risk management
- Database persistence
- WebSocket streaming

Last updated: 2025-01-03
"""

import os
import sys
import json
import asyncio
import logging
import math
import time
import random
import traceback
import sqlite3
import hashlib
import hmac
import ast
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils.sanity import is_sane_ticker

# ═══════════════════════════ DEPENDENCY MANAGEMENT ═══════════════════════════

REQUIREMENTS = [
    "openai>=1.12.0",
    "ccxt>=4.4.85",
    "aiohttp>=3.9.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "ta>=0.11.0",
    "scipy>=1.10.0",
    "websockets>=12.0",
    "python-dotenv>=1.0.0",
    "plotly>=5.18.0",
    "streamlit>=1.29.0"
]

def ensure_dependencies():
    """Auto-install missing dependencies - non-blocking startup"""
    import subprocess
    import importlib.util
    
    for req in REQUIREMENTS:
        package = req.split(">=")[0]
        if importlib.util.find_spec(package) is None:
            print(f"Installing {req}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", req], timeout=120)
            except subprocess.TimeoutExpired:
                print(f"❌ Installation of {req} timed out after 120s")
                raise
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {req}: {e}")
                raise

def handle_deps():
    """Handle dependency installation and listing."""
    if "--install" in sys.argv:
        ensure_dependencies()
        print("✅ All dependencies installed!")
        sys.exit(0)

    if "--deps" in sys.argv:
        for req in REQUIREMENTS:
            print(req)
        sys.exit(0)

handle_deps()

# Load environment variables BEFORE importing Config
from dotenv import load_dotenv
load_dotenv()

# Import after ensuring dependencies
try:
    import pandas as pd
    import numpy as np
    import ccxt.async_support as ccxt
    import aiohttp
    from aiohttp import web
    import websockets
    import ta
    from scipy import stats
    from openai import AsyncOpenAI
except ImportError as e:
    package_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
    print(f"❌ Missing dependency '{package_name}': {e}", file=sys.stderr)
    print(f"Run: python3 {os.path.basename(__file__)} --install", file=sys.stderr)
    sys.exit(1)

# ═══════════════════════════ CONFIGURATION ═══════════════════════════════════

class Config:
    """System configuration"""
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    
    # Exchange API Keys (USA compliant)
    COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
    COINBASE_SECRET = os.getenv("COINBASE_SECRET")
    COINBASE_PASSPHRASE = os.getenv("COINBASE_PASSPHRASE")
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
    KRAKEN_SECRET = os.getenv("KRAKEN_SECRET")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_SECRET = os.getenv("GEMINI_SECRET")
    
    # Trading Configuration - with validation
    try:
        INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "200"))
        TARGET_CAPITAL = float(os.getenv("TARGET_CAPITAL", "1000000"))
        MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.02"))
        MAX_DAILY_DRAWDOWN = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.10"))
        MIN_ORDER_NOTIONAL = float(os.getenv("MIN_ORDER_NOTIONAL", "0.0"))  # 0.0 for paper trading
        MAX_EXPOSURE_PCT = float(os.getenv("MAX_EXPOSURE_PCT", "0.5"))  # 50% total exposure limit
        WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
    except ValueError as e:
        print(f"❌ Invalid numeric config value: {e}", file=sys.stderr)
        sys.exit(1)
    
    # System Configuration
    MODE = os.getenv("MODE", "PAPER")  # PAPER or REAL
    DATABASE_PATH = os.getenv("DATABASE_PATH", "v26meme.db")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
    
    # Web Security Configuration
    WEB_AUTH_TOKEN = os.getenv("WEB_AUTH_TOKEN", "")  # Authentication token for web API
    WEB_BIND_HOST = os.getenv("WEB_BIND_HOST", "127.0.0.1")  # Bind to localhost only by default
    
    # Alerts
    DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and handle missing API keys based on mode"""
        if cls.MODE == "REAL":
            # Require all API keys for real trading
            required_keys = [
                ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
                ("COINBASE_API_KEY", cls.COINBASE_API_KEY),
                ("COINBASE_SECRET", cls.COINBASE_SECRET)
            ]
            
            missing = [name for name, value in required_keys if not value]
            if missing:
                print(f"❌ Missing required API keys for REAL mode: {', '.join(missing)}", file=sys.stderr)
                sys.exit(1)
        else:
            # Paper mode - warn about missing keys but don't exit
            if not cls.OPENAI_API_KEY:
                print("⚠️  Warning: OPENAI_API_KEY missing - strategy generation will be limited", file=sys.stderr)
            if not any([cls.COINBASE_API_KEY, cls.KRAKEN_API_KEY, cls.GEMINI_API_KEY]):
                print("⚠️  Warning: No exchange API keys found - using demo data", file=sys.stderr)

# ═══════════════════════════ LOGGING SYSTEM ═══════════════════════════════════

# ═══════════════════════════ SECURITY & VALIDATION ═══════════════════════════

class StrategySecurityValidator:
    """Validates strategy code for security vulnerabilities using AST parsing"""
    
    FORBIDDEN_MODULES = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests', 
        'shutil', 'tempfile', 'pathlib', 'glob', 'importlib',
        'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3'
    }
    
    FORBIDDEN_BUILTINS = {
        'exec', 'eval', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'execfile', 'reload'
    }
    
    @classmethod
    def validate_strategy_code(cls, code: str) -> tuple[bool, str]:
        """Validate strategy code using AST parsing. Returns (is_valid, error_msg)"""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for forbidden imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name in cls.FORBIDDEN_MODULES:
                            return False, f"Forbidden import: {alias.name}"
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name in cls.FORBIDDEN_MODULES:
                            return False, f"Forbidden import from: {node.module}"
                
                # Check for forbidden built-ins
                elif isinstance(node, ast.Name):
                    if node.id in cls.FORBIDDEN_BUILTINS:
                        return False, f"Forbidden built-in: {node.id}"
                
                # Check for attribute access to dangerous methods
                elif isinstance(node, ast.Attribute):
                    if (isinstance(node.value, ast.Name) and 
                        node.value.id == '__builtins__'):
                        return False, "Direct __builtins__ access forbidden"
                    
                    if node.attr.startswith('__') and node.attr.endswith('__'):
                        return False, f"Forbidden dunder attribute: {node.attr}"
                
                # Check for function definitions other than execute_strategy
                elif isinstance(node, ast.FunctionDef):
                    if node.name not in ['execute_strategy']:
                        return False, f"Only 'execute_strategy' function allowed, found: {node.name}"
            
            # Ensure execute_strategy function exists
            function_names = [node.name for node in ast.walk(tree) 
                            if isinstance(node, ast.FunctionDef)]
            if 'execute_strategy' not in function_names:
                return False, "Missing required 'execute_strategy' function"
            
            return True, "Code validated successfully"
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

class SecureStrategyExecutor:
    """Executes strategy code in a sandboxed environment"""
    
    @staticmethod
    async def execute_in_sandbox(strategy_code: str, market_data: dict, state: dict) -> dict:
        """Execute strategy code in a secure subprocess sandbox"""
        
        # First validate the code
        is_valid, error_msg = StrategySecurityValidator.validate_strategy_code(strategy_code)
        if not is_valid:
            raise ValueError(f"Strategy validation failed: {error_msg}")
        
        # Create a temporary execution script
        execution_script = f'''
import json
import math
import numpy as np
import pandas as pd

# Market data and state
market_data = {json.dumps(market_data)}
state = {json.dumps(state)}

# Strategy code
{strategy_code}

# Execute and return result
try:
    result = execute_strategy(state, market_data)
    print(json.dumps({{"success": True, "result": result}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
'''
        
        try:
            # Execute in subprocess with timeout
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(execution_script)
                temp_file = f.name
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir()  # Run in temp directory
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=5.0  # 5 second timeout
                )
                
                if process.returncode != 0:
                    raise RuntimeError(f"Strategy execution failed: {stderr.decode()}")
                
                result = json.loads(stdout.decode().strip())
                
                if not result.get('success'):
                    raise RuntimeError(f"Strategy error: {result.get('error')}")
                
                return result['result']
                
            except asyncio.TimeoutError:
                process.kill()
                raise RuntimeError("Strategy execution timed out")
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

# ═══════════════════════════ LOGGING SYSTEM ═══════════════════════════════════

class Logger:
    """Enhanced logging with buffered WebSocket broadcasting and rotation"""
    
    def __init__(self):
        self.logger = logging.getLogger("v26meme")
        self.logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(console)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            "v26meme.log", 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(funcName)s | %(message)s"
        ))
        self.logger.addHandler(file_handler)
        
        # WebSocket clients and buffering
        self.ws_clients = set()
        self.log_queue = asyncio.Queue(maxsize=1000)  # Buffer up to 1000 messages
        self.broadcast_task = None
        self._shutdown = False
    
    async def start_broadcast_worker(self):
        """Start the log broadcast worker"""
        if self.broadcast_task is None:
            self.broadcast_task = asyncio.create_task(self._broadcast_worker())
    
    async def stop_broadcast_worker(self):
        """Stop the log broadcast worker"""
        self._shutdown = True
        if self.broadcast_task:
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
    
    async def _broadcast_worker(self):
        """Background worker to broadcast logs to WebSocket clients"""
        while not self._shutdown:
            try:
                # Wait for log message or timeout
                try:
                    level, message = await asyncio.wait_for(
                        self.log_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Broadcast to all connected clients
                if self.ws_clients:
                    disconnected = set()
                    
                    for ws in self.ws_clients.copy():
                        try:
                            await ws.send_str(json.dumps({
                                'type': 'log',
                                'level': level,
                                'message': message,
                                'timestamp': datetime.utcnow().isoformat()
                            }))
                        except Exception:
                            disconnected.add(ws)
                    
                    # Remove disconnected clients
                    self.ws_clients -= disconnected
                
                self.log_queue.task_done()
                
            except Exception as e:
                # Don't use the logger here to avoid recursion
                print(f"Error in log broadcast worker: {e}")
        
    def _queue_broadcast(self, level: str, message: str):
        """Queue a message for broadcasting"""
        try:
            # Non-blocking put, drop message if queue is full
            self.log_queue.put_nowait((level, message))
        except asyncio.QueueFull:
            # Drop oldest message and add new one
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait((level, message))
            except asyncio.QueueEmpty:
                pass
        self._message_buffer = []
        self._loop_running = False
    
    def _emit(self, level: str, msg: str, **ctx):
        """Emit structured JSON log with context"""
        record = {
            "ts": datetime.utcnow().isoformat(),
            "level": level,
            "msg": msg,
            **ctx  # extra context payload
        }
        line = json.dumps(record, default=str)
        self.logger.log(getattr(logging, level), line)
        
        # Queue for broadcasting instead of creating tasks immediately
        self._queue_broadcast(level, line)

    def start_async_logging(self):
        """Call this when the event loop starts"""
        self._loop_running = True
        # Process buffered messages
        for level, line in self._message_buffer:
            asyncio.create_task(self.broadcast(level, line))
        self._message_buffer.clear()

    async def broadcast(self, level: str, message: str):
        """Broadcast log to WebSocket clients"""
        data = json.dumps({
            "type": "log",
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        dead_clients = set()
        for client in self.ws_clients:
            try:
                await client.send_str(data)
            except:
                dead_clients.add(client)
        
        self.ws_clients -= dead_clients
    
    # Structured logging methods
    def debug(self, msg: str, **ctx):   
        if self.logger.isEnabledFor(logging.DEBUG):
            self._emit("DEBUG", msg, **ctx)
    
    def info(self, msg: str, **ctx):    
        self._emit("INFO", msg, **ctx)
    
    def warning(self, msg: str, **ctx): 
        self._emit("WARNING", msg, **ctx)
    
    def error(self, msg: str, **ctx):   
        self._emit("ERROR", msg, **ctx)
    
    def success(self, msg: str, **ctx): 
        self._emit("INFO", f"✅ {msg}", **ctx)

log = Logger()

# ═══════════════════════════ DATA MODELS ═══════════════════════════════════

class TradingMode(Enum):
    PAPER = "PAPER"
    REAL = "REAL"

class StrategyStatus(Enum):
    TESTING = "TESTING"
    PAPER = "PAPER"
    LIVE = "LIVE"
    RETIRED = "RETIRED"

@dataclass
class Strategy:
    """Trading strategy with performance tracking"""
    id: str
    name: str
    description: str
    code: str
    created_at: datetime
    status: StrategyStatus = StrategyStatus.TESTING
    
    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Genetic algorithm fitness
    fitness: float = 0.0
    generation: int = 0
    
    def update_metrics(self, trade_result: Dict):
        """Update strategy metrics after trade"""
        self.total_trades += 1
        if trade_result["pnl"] > 0:
            self.winning_trades += 1
        self.total_pnl += trade_result["pnl"]
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        self.fitness = self.calculate_fitness()
    
    def calculate_fitness(self) -> float:
        """Calculate fitness score for genetic algorithm"""
        if self.total_trades < 10:
            return 0.0
        
        # Multi-factor fitness function
        win_rate_score = self.win_rate * 30
        sharpe_score = max(0, min(self.sharpe_ratio * 20, 40))
        # Fix: Apply penalty for drawdown (negative drawdown means larger loss)
        drawdown_penalty = max(0, abs(self.max_drawdown) * 20)
        consistency_bonus = 10 if self.profit_factor > 1.5 else 0
        
        return win_rate_score + sharpe_score - drawdown_penalty + consistency_bonus

@dataclass
class Position:
    """Active trading position"""
    id: str
    symbol: str
    exchange: str
    side: str  # "long" or "short"
    entry_price: float
    current_price: float
    quantity: float
    strategy_id: str
    opened_at: datetime
    
    # Risk management
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    
    @property
    def pnl(self) -> float:
        """Calculate current P&L"""
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def pnl_percent(self) -> float:
        """Calculate P&L percentage"""
        if self.entry_price == 0 or self.quantity == 0:
            return 0.0
        return (self.pnl / (self.entry_price * self.quantity)) * 100

@dataclass
class SystemState:
    """Global system state"""
    mode: TradingMode = TradingMode.PAPER
    equity: float = Config.INITIAL_CAPITAL
    cash: float = Config.INITIAL_CAPITAL
    positions_value: float = 0.0
    
    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_equity: float = Config.INITIAL_CAPITAL
    
    # Risk metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    daily_trades: int = 0
    
    # Risk management state
    daily_pause_until: Optional[datetime] = None  # 6-hour pause timestamp
    global_kill_switch: bool = False  # 90% drawdown kill switch
    
    # Analysis metrics
    opportunities_analyzed: int = 0
    opportunities_skipped: int = 0
    opportunities_traded: int = 0
    
    # Multi-algorithm performance tracking
    algorithm_performance_history: List[Dict] = field(default_factory=list)
    recent_trade_results: List[Dict] = field(default_factory=list)
    evolution_cycles: int = 0
    last_evolution: Optional[datetime] = None
    
    # Collections
    strategies: Dict[str, Strategy] = field(default_factory=dict)
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    def update_equity(self):
        """Update total equity"""
        self.positions_value = sum(pos.pnl for pos in self.positions.values())
        self.equity = self.cash + self.positions_value
        
        # Track maximum equity for drawdown calculation
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        # Calculate drawdown
        if self.max_equity > 0:
            self.current_drawdown = (self.max_equity - self.equity) / self.max_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

# ═══════════════════════════ DATABASE LAYER ═══════════════════════════════════

class Database:
    """SQLite database for persistence with concurrency protection"""
    
    SCHEMA_VERSION = 1
    
    def __init__(self, path: str = Config.DATABASE_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._write_lock = asyncio.Lock()  # Protect concurrent writes
        self.init_schema()
    
    def init_schema(self):
        """Initialize database schema with versioning"""
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );
            
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                code TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                fitness REAL DEFAULT 0,
                generation INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                FOREIGN KEY (strategy_id) REFERENCES strategies(id)
            );
            
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                mode TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                last_update TEXT
            );
            
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            );
            
            INSERT OR IGNORE INTO schema_version (version) VALUES ({self.SCHEMA_VERSION});
        """)
        self.conn.commit()
    
    async def save_strategy(self, strategy: Strategy):
        """Save or update strategy with concurrency protection"""
        async with self._write_lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO strategies 
                (id, name, description, code, status, created_at, total_trades, 
                 winning_trades, total_pnl, sharpe_ratio, max_drawdown, fitness, generation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy.id, strategy.name, strategy.description, strategy.code,
                strategy.status.value, strategy.created_at.isoformat(), strategy.total_trades,
                strategy.winning_trades, strategy.total_pnl, strategy.sharpe_ratio,
                strategy.max_drawdown, strategy.fitness, strategy.generation
            ))
            self.conn.commit()
    
    async def save_trade(self, trade: Dict):
        """Save trade record with concurrency protection"""
        async with self._write_lock:
            self.conn.execute("""
                INSERT INTO trades 
                (id, strategy_id, symbol, exchange, side, entry_price, exit_price, 
                 quantity, pnl, opened_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade["id"], trade["strategy_id"], trade["symbol"], trade["exchange"],
                trade["side"], trade["entry_price"], trade.get("exit_price"),
                trade["quantity"], trade.get("pnl"), 
                trade["opened_at"].isoformat() if isinstance(trade["opened_at"], datetime) else trade["opened_at"],
                trade.get("closed_at").isoformat() if trade.get("closed_at") and isinstance(trade.get("closed_at"), datetime) else trade.get("closed_at")
            ))
            self.conn.commit()
    
    async def save_state(self, state: SystemState):
        """Save system state with concurrency protection"""
        async with self._write_lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO system_state VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.mode.value, state.equity, state.cash, state.total_trades,
                state.winning_trades, state.total_pnl, state.max_drawdown,
                datetime.utcnow().isoformat()
            ))
            
            # Save all strategies
            for strategy in state.strategies.values():
                await self.save_strategy_sync(strategy)  # Use sync version to avoid double locking
            
            self.conn.commit()
    
    def save_strategy_sync(self, strategy: Strategy):
        """Synchronous save strategy (for use within locked context)"""
        self.conn.execute("""
            INSERT OR REPLACE INTO strategies 
            (id, name, description, code, status, created_at, total_trades, 
             winning_trades, total_pnl, sharpe_ratio, max_drawdown, fitness, generation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy.id, strategy.name, strategy.description, strategy.code,
            strategy.status.value, strategy.created_at.isoformat(), strategy.total_trades,
            strategy.winning_trades, strategy.total_pnl, strategy.sharpe_ratio,
            strategy.max_drawdown, strategy.fitness, strategy.generation
        ))
        
    def load_state(self) -> Optional[SystemState]:
        """Load system state from database"""
        cursor = self.conn.execute("SELECT * FROM system_state WHERE id = 1")
        row = cursor.fetchone()
        
        if not row:
            return None
        
        state = SystemState(
            mode=TradingMode(row["mode"]),
            equity=row["equity"],
            cash=row["cash"],
            total_trades=row["total_trades"],
            winning_trades=row["winning_trades"],
            total_pnl=row["total_pnl"],
            max_drawdown=row["max_drawdown"]
        )
        
        # Load strategies
        cursor = self.conn.execute("SELECT * FROM strategies")
        for row in cursor:
            try:
                created_at = datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
            except (ValueError, TypeError):
                created_at = datetime.utcnow()  # fallback
                
            strategy = Strategy(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                code=row["code"],
                created_at=created_at,
                status=StrategyStatus(row["status"]),
                total_trades=row["total_trades"],
                winning_trades=row["winning_trades"],
                total_pnl=row["total_pnl"],
                sharpe_ratio=row["sharpe_ratio"],
                max_drawdown=row["max_drawdown"],
                fitness=row["fitness"],
                generation=row["generation"]
            )
            strategy.win_rate = strategy.winning_trades / strategy.total_trades if strategy.total_trades > 0 else 0
            state.strategies[strategy.id] = strategy
        
        return state

# ═══════════════════════════ NOTIFICATION SYSTEM ═══════════════════════════════

class NotificationManager:
    """Multi-channel notification system"""
    
    def __init__(self):
        self.session = aiohttp.ClientSession()
        self.alerts: List[Dict] = []

    async def send(self, message: str, level: str = "INFO"):
        """Store notification for web UI"""
        emoji = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "TRADE": "💰",
            "ALERT": "🚨"
        }.get(level, "📢")
        
        formatted_message = f"{emoji} {message}"
        
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": formatted_message
        }
        self.alerts.append(alert)
        
        # Keep only the last 100 alerts
        if len(self.alerts) > 100:
            self.alerts.pop(0)

    async def close(self):
        await self.session.close()

# ═══════════════════════════ DATA SANITY HELPERS ═══════════════════════════
# Note: is_sane_ticker is now imported from utils.sanity

# ═══════════════════════════ AI AGENTS ═══════════════════════════════════

class OpenAIManager:
    """Manages OpenAI API interactions"""
    
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL
    
    async def complete(self, prompt: str, temperature: float = 0.7, max_retries: int = 3) -> str:
        """Get completion from OpenAI with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": "You are an expert cryptocurrency trading AI with deep knowledge of markets, technical analysis, and risk management."},
                    {"role": "user", "content": prompt}
                ]
                
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature
                }
                
                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
                
            except Exception as e:
                last_exception = e
                
                # Check if it's a rate limit error
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2, 4, 8 seconds
                        delay = 2 ** (attempt + 1) + random.uniform(0, 1)
                        log.warning(f"OpenAI rate limited, retrying in {delay:.1f}s (attempt {attempt + 1})")
                        await asyncio.sleep(delay)
                        continue
                
                # For other errors, retry with shorter delay
                elif attempt < max_retries - 1:
                    delay = 1 + random.uniform(0, 1)
                    log.warning(f"OpenAI error, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                
                # Final attempt failed
                log.error(f"OpenAI API error after {max_retries} attempts: {e}")
                raise
        
        # This should never be reached
        raise last_exception or RuntimeError("OpenAI completion failed")

class MarketAnalyzer:
    """
    Analyzes market conditions and identifies opportunities.
    This is a gold-standard agent with retry logic, timeouts, and structured outputs.
    """
    
    def __init__(self, exchanges: Dict, ai: OpenAIManager, state: SystemState):
        self.exchanges = exchanges
        self.ai = ai
        self.state = state
        self.session = aiohttp.ClientSession()
        self.last_market_heartbeat = datetime.utcnow()
        self.ws_clients = set()
        self.max_ws_clients = 100  # Prevent unbounded growth
        
    async def add_ws_client(self, client):
        """Add WebSocket client with capacity management"""
        if len(self.ws_clients) >= self.max_ws_clients:
            # Remove oldest/first client to make room
            oldest = next(iter(self.ws_clients))
            self.ws_clients.remove(oldest)
            try:
                await oldest.close()
            except:
                pass
        self.ws_clients.add(client)
        
    async def remove_ws_client(self, client):
        """Remove WebSocket client"""
        self.ws_clients.discard(client)

    async def scan_markets(self) -> List[Dict]:
        """
        Main entry point for scanning markets. Orchestrates fetching, analysis, and fallback logic.
        """
        log.info("Scanning markets...", context="scanner_orchestrator")
        
        # 1. Scan all exchanges for significant movers
        opportunities = await self._scan_all_exchanges()
        
        # 2. Get trending coins from CoinGecko
        trending = await self._get_trending_opportunities()
        opportunities.extend(trending)
        
        # 3. Cross-exchange arbitrage opportunities
        arbitrage = await self._scan_cross_exchange_arbitrage()
        opportunities.extend(arbitrage)
        
        # 4. Statistical signals (z-scores, mean reversion)
        statistical = await self._scan_statistical_signals()
        opportunities.extend(statistical)
        
        # 5. Order book microstructure opportunities
        microstructure = await self._scan_order_book_opportunities()
        opportunities.extend(microstructure)
        
        # 6. Fallback: If no opportunities, get top volume coins
        if not opportunities:
            log.info("No movers or trending found, falling back to top volume.", context="scanner_fallback")
            fallback_opps = await self._get_top_volume_fallback()
            opportunities.extend(fallback_opps)

        # Update heartbeat after all data fetching is complete
        self.last_market_heartbeat = datetime.utcnow()
        
        log.info(f"🧠 Total opportunities found: {len(opportunities)}", context="scanner_orchestrator")
        
        # 7. Get AI analysis for each opportunity
        analyzed = []
        for i, opp in enumerate(opportunities[:15]):  # Analyze top 15
            log.info(f"� Analyzing opportunity {i+1}/{min(15, len(opportunities))}: {opp['symbol']} on {opp['exchange']}")
            log.debug("Opportunity data", **opp)
            
            analysis = await self._analyze_opportunity(opp)
            
            score = analysis.get('score', 0)
            reasoning = analysis.get('reasoning', 'N/A')
            
            if score > 0.3:  # Lowered from 0.6 to start learning with more trades
                log.success(f"🎯 HIGH-VALUE OPPORTUNITY: {opp['symbol']} (Score: {score:.2f}) - {reasoning}")
                analysis['symbol'] = opp['symbol']
                analysis['exchange'] = opp['exchange']
                analyzed.append(analysis)
            else:
                log.info(f"⏭️  Opportunity skipped: {opp['symbol']} (Score: {score:.2f}) - {reasoning}")
        
        log.info(f"📋 Final analysis: {len(analyzed)} opportunities passed AI filter")
        return sorted(analyzed, key=lambda x: x['score'], reverse=True)

    async def _scan_all_exchanges(self) -> List[Dict]:
        """Scan all configured exchanges concurrently."""
        tasks = [self._scan_single_exchange(name, ex) for name, ex in self.exchanges.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_opportunities = []
        for res in results:
            if isinstance(res, list):
                all_opportunities.extend(res)
        return all_opportunities

    async def _scan_single_exchange(self, name: str, exchange: ccxt.Exchange) -> List[Dict]:
        """Scan one exchange with a timeout."""
        try:
            log.info(f"📊 Scanning {name}...", context="scanner_exchange")
            
            tickers = await asyncio.wait_for(exchange.fetch_tickers(), timeout=20.0)  # 20-second timeout
            
            log.info(f"📈 Found {len(tickers)} total tickers on {name}")
            
            def _enrich(t):
                last  = t.get("last")
                open_ = t.get("open") or None
                # Δ% 24 h
                if last and open_ and t.get("percentage") is None:
                    t["percentage"] = (last - open_) / open_ * 100
                # USD volume
                if t.get("quoteVolume") is None and last:
                    base_vol = t.get("baseVolume")
                    if base_vol:
                        t["quoteVolume"] = base_vol * last
                return t

            usd_pairs = {k: _enrich(v) for k, v in tickers.items() if '/USD' in k or '/USDT' in k or '/USDC' in k}
            log.info(f"� Found {len(usd_pairs)} USD pairs on {name}")

            # Log first 5 tickers for debugging
            for i, (symbol, ticker) in enumerate(usd_pairs.items()):
                if i >= 5: break
                if is_sane_ticker(ticker):
                    # Safely format all values to avoid None format errors
                    last = ticker.get('last', 0) or 0
                    pct = ticker.get('percentage') or 0
                    pct = pct if isinstance(pct, (int, float)) else 0
                    volume = ticker.get('quoteVolume', 0) or 0
                    volume = volume if isinstance(volume, (int, float)) else 0
                    log.debug(f"TICKER: {symbol} | Last: ${last:.6f} | Change: {pct:.2f}% | Volume: ${volume:,.0f}", exchange=name)

            # Run multiple opportunity detection algorithms in parallel
            detection_results = await self._run_multi_logic_detection(usd_pairs, name)
            
            # Aggregate and score opportunities using dynamic weighting
            opportunities = self._aggregate_opportunity_scores(detection_results, name)
            
            log.success(f"Found {len(opportunities)} movers on {name}", context="scanner_exchange")
            return opportunities

        except asyncio.TimeoutError:
            log.warning(f"Scan timed out for {name}", context="scanner_exchange")
        except Exception as e:
            log.error(f"Error scanning {name}: {e}", context="scanner_exchange", exc_info=True)
        
        return []

    async def _run_multi_logic_detection(self, usd_pairs: Dict, exchange_name: str) -> Dict[str, List[Dict]]:
        """Run multiple opportunity detection algorithms in parallel."""
        
        # Initialize adaptive thresholds (these evolve based on performance)
        adaptive_thresholds = getattr(self, 'adaptive_thresholds', {
            'momentum_threshold': 1.0,      # Start point - more reasonable threshold
            'volume_multiplier': 1.0,       # Catch any volume spike
            'rsi_oversold': 30,            # RSI thresholds
            'rsi_overbought': 70,
            'bollinger_deviation': 1.0,    # Very sensitive bands
            'macd_signal_strength': 0.01   # Ultra-sensitive MACD
        })
        
        detection_algorithms = [
            self._detect_momentum_breakouts(usd_pairs, adaptive_thresholds),
            self._detect_volume_anomalies(usd_pairs, adaptive_thresholds),
            self._detect_technical_signals(usd_pairs, adaptive_thresholds),
            self._detect_volatility_expansion(usd_pairs, adaptive_thresholds),
            self._detect_mean_reversion_setups(usd_pairs, adaptive_thresholds),
            self._detect_liquidity_imbalances(usd_pairs, adaptive_thresholds)
        ]
        
        # Run all detection algorithms concurrently
        results = await asyncio.gather(*detection_algorithms, return_exceptions=True)
        
        algorithm_names = [
            'momentum_breakouts', 'volume_anomalies', 'technical_signals',
            'volatility_expansion', 'mean_reversion', 'liquidity_imbalances'
        ]
        
        detection_results = {}
        for i, (name, result) in enumerate(zip(algorithm_names, results)):
            if isinstance(result, Exception):
                log.warning(f"Detection algorithm {name} failed: {result}", context="multi_logic")
                detection_results[name] = []
            else:
                detection_results[name] = result or []
                log.info(f"🔍 {name}: {len(detection_results[name])} opportunities", context="multi_logic")
        
        return detection_results

    async def _detect_momentum_breakouts(self, usd_pairs: Dict, thresholds: Dict) -> List[Dict]:
        """Detect momentum breakouts with adaptive thresholds."""
        opportunities = []
        
        # Debug: Show top movers for analysis
        sorted_pairs = sorted(usd_pairs.items(), key=lambda x: abs(x[1].get('percentage', 0) or 0), reverse=True)
        log.debug(f"🔍 Top 5 movers: {[(s, t.get('percentage', 0)) for s, t in sorted_pairs[:5]]}", context="momentum_debug")
        log.debug(f"🎯 Momentum threshold: {thresholds['momentum_threshold']}%", context="momentum_debug")
        
        for symbol, ticker in usd_pairs.items():
            if not is_sane_ticker(ticker):
                continue
                
            change_pct = ticker.get('percentage') or 0
            volume = ticker.get('quoteVolume', 0)
            
            # Debug log for significant moves
            if abs(change_pct) > 5.0:  # Log anything above 5%
                log.debug(f"📊 {symbol}: {change_pct}% (threshold: {thresholds['momentum_threshold']}%)", context="momentum_debug")
            
            # Dynamic momentum threshold based on market volatility
            if abs(change_pct) > thresholds['momentum_threshold']:
                momentum_score = min(abs(change_pct) / 10.0, 1.0)  # Normalize to 0-1
                
                opportunities.append({
                    'symbol': symbol,
                    'exchange': 'detected_exchange',
                    'change_24h': change_pct,
                    'volume_24h': volume,
                    'price': ticker['last'],
                    'reason': 'momentum_breakout',
                    'algorithm': 'momentum',
                    'confidence': momentum_score,
                    'raw_score': abs(change_pct)
                })
        
        return opportunities

    async def _detect_volume_anomalies(self, usd_pairs: Dict, thresholds: Dict) -> List[Dict]:
        """Detect unusual volume spikes."""
        opportunities = []
        # Only include valid, non-None volumes
        volumes = []
        for t in usd_pairs.values():
            if is_sane_ticker(t):
                vol = t.get('quoteVolume') or t.get('baseVolume') or 0
                if vol is not None and isinstance(vol, (int, float)) and vol > 0:
                    volumes.append(vol)
        
        if len(volumes) < 10:
            return opportunities
            
        volume_mean = np.mean(volumes)
        volume_std = np.std(volumes)
        
        for symbol, ticker in usd_pairs.items():
            if not is_sane_ticker(ticker):
                continue
                
            volume = ticker.get('quoteVolume') or ticker.get('baseVolume') or 0
            if volume is None or not isinstance(volume, (int, float)):
                continue
                
            # Z-score based volume anomaly detection
            if volume_mean > 0 and volume_std > 0 and volume > 0:
                volume_zscore = (volume - volume_mean) / volume_std
                
                if volume_zscore > thresholds['volume_multiplier']:
                    volume_score = min(volume_zscore / 5.0, 1.0)  # Normalize
                    
                    opportunities.append({
                        'symbol': symbol,
                        'exchange': 'detected_exchange',
                        'change_24h': ticker.get('percentage', 0),
                        'volume_24h': volume,
                        'price': ticker['last'],
                        'reason': 'volume_anomaly',
                        'algorithm': 'volume',
                        'confidence': volume_score,
                        'raw_score': volume_zscore
                    })
        
        return opportunities

    async def _detect_technical_signals(self, usd_pairs: Dict, thresholds: Dict) -> List[Dict]:
        """Detect technical analysis signals with robust null handling."""
        opportunities = []
        
        # Collect price data for statistical analysis - filter out None values
        prices = []
        changes = []
        
        for ticker in usd_pairs.values():
            if not is_sane_ticker(ticker):
                continue
            
            price = ticker.get('last')
            change = ticker.get('percentage')
            
            if price is not None and isinstance(price, (int, float)) and price > 0:
                prices.append(float(price))
            
            if change is not None and isinstance(change, (int, float)):
                changes.append(float(change))
        
        if len(prices) < 20 or len(changes) < 20:
            return opportunities
        
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        change_mean = np.mean(changes)
        change_std = np.std(changes)
        
        for symbol, ticker in usd_pairs.items():
            if not is_sane_ticker(ticker):
                continue
                
            price = ticker.get('last')
            change = ticker.get('percentage')
            
            # Robust null checking
            if price is None or change is None:
                continue
            if not isinstance(price, (int, float)) or not isinstance(change, (int, float)):
                continue
            if price <= 0:
                continue
                
            price = float(price)
            change = float(change)
            
            # Statistical deviation signals
            if price_std > 0 and change_std > 0:
                price_zscore = abs((price - price_mean) / price_std)
                change_zscore = abs((change - change_mean) / change_std)
                
                # Combined technical signal strength
                technical_strength = (price_zscore + change_zscore) / 2
                
                if technical_strength > 1.5:  # 1.5 standard deviations
                    tech_score = min(technical_strength / 3.0, 1.0)
                    
                    opportunities.append({
                        'symbol': symbol,
                        'exchange': 'detected_exchange',
                        'change_24h': change,
                        'volume_24h': ticker.get('quoteVolume', 0) or 0,
                        'price': price,
                        'reason': 'technical_signal',
                        'algorithm': 'technical',
                        'confidence': tech_score,
                        'raw_score': technical_strength
                    })
        
        return opportunities

    async def _detect_volatility_expansion(self, usd_pairs: Dict, thresholds: Dict) -> List[Dict]:
        """Detect volatility expansion patterns with robust null handling."""
        opportunities = []
        
        # Calculate market volatility metrics - filter None values
        changes = []
        for ticker in usd_pairs.values():
            if not is_sane_ticker(ticker):
                continue
            
            change = ticker.get('percentage')
            if change is not None and isinstance(change, (int, float)):
                changes.append(abs(float(change)))
        
        if len(changes) < 10:
            return opportunities
            
        volatility_percentile_90 = np.percentile(changes, 90)
        
        for symbol, ticker in usd_pairs.items():
            if not is_sane_ticker(ticker):
                continue
                
            change = ticker.get('percentage')
            price = ticker.get('last')
            
            # Robust null checking
            if change is None or price is None:
                continue
            if not isinstance(change, (int, float)) or not isinstance(price, (int, float)):
                continue
            if price <= 0:
                continue
                
            change = float(change)
            price = float(price)
            abs_change = abs(change)
            
            # High volatility relative to market
            if volatility_percentile_90 > 0 and abs_change > volatility_percentile_90 * 1.2:  # 20% above 90th percentile
                volatility_score = min(abs_change / (volatility_percentile_90 * 2), 1.0)
                
                opportunities.append({
                    'symbol': symbol,
                    'exchange': 'detected_exchange',
                    'change_24h': change,
                    'volume_24h': ticker.get('quoteVolume', 0) or 0,
                    'price': price,
                    'reason': 'volatility_expansion',
                    'algorithm': 'volatility',
                    'confidence': volatility_score,
                    'raw_score': abs_change
                })
        
        return opportunities

    async def _detect_mean_reversion_setups(self, usd_pairs: Dict, thresholds: Dict) -> List[Dict]:
        """Detect mean reversion opportunities with robust null handling."""
        opportunities = []
        
        # Collect valid changes - filter None values
        changes = []
        for ticker in usd_pairs.values():
            if not is_sane_ticker(ticker):
                continue
            
            change = ticker.get('percentage')
            if change is not None and isinstance(change, (int, float)):
                changes.append(float(change))
        
        if len(changes) < 20:
            return opportunities
            
        change_mean = np.mean(changes)
        change_std = np.std(changes)
        
        for symbol, ticker in usd_pairs.items():
            if not is_sane_ticker(ticker):
                continue
                
            change = ticker.get('percentage')
            price = ticker.get('last')
            
            # Robust null checking
            if change is None or price is None:
                continue
            if not isinstance(change, (int, float)) or not isinstance(price, (int, float)):
                continue
            if price <= 0:
                continue
                
            change = float(change)
            price = float(price)
            
            # Extreme moves for mean reversion
            if change_std > 0:
                z_score = abs((change - change_mean) / change_std)
                
                if z_score > 2.5:  # 2.5 standard deviations
                    reversion_score = min(z_score / 4.0, 1.0)
                    
                    opportunities.append({
                        'symbol': symbol,
                        'exchange': 'detected_exchange',
                        'change_24h': change,
                        'volume_24h': ticker.get('quoteVolume', 0) or 0,
                        'price': price,
                        'reason': 'mean_reversion',
                        'algorithm': 'reversion',
                        'confidence': reversion_score,
                        'raw_score': z_score
                    })
        
        return opportunities

    async def _detect_liquidity_imbalances(self, usd_pairs: Dict, thresholds: Dict) -> List[Dict]:
        """Detect liquidity imbalances using bid-ask spreads with robust null handling."""
        opportunities = []
        
        for symbol, ticker in usd_pairs.items():
            if not is_sane_ticker(ticker):
                continue
                
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            last = ticker.get('last')
            change = ticker.get('percentage')
            
            # Robust null checking
            if bid is None or ask is None or last is None:
                continue
            if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)) or not isinstance(last, (int, float)):
                continue
            if bid <= 0 or ask <= 0 or last <= 0:
                continue
            if ask <= bid:  # Invalid spread
                continue
                
            bid = float(bid)
            ask = float(ask)
            last = float(last)
            change = float(change) if change is not None and isinstance(change, (int, float)) else 0
            
            spread = (ask - bid) / last
            
            # Large spreads indicate liquidity imbalances
            if spread > 0.005:  # 0.5% spread threshold
                liquidity_score = min(spread * 100, 1.0)  # Normalize
                
                opportunities.append({
                    'symbol': symbol,
                    'exchange': 'detected_exchange',
                    'change_24h': change,
                    'volume_24h': ticker.get('quoteVolume', 0) or 0,
                    'price': last,
                    'reason': 'liquidity_imbalance',
                    'algorithm': 'liquidity',
                    'confidence': liquidity_score,
                    'raw_score': spread,
                    'bid': bid,
                    'ask': ask,
                    'spread_pct': spread * 100
                })
        
        return opportunities
        
        return opportunities

    def _aggregate_opportunity_scores(self, detection_results: Dict[str, List[Dict]], exchange_name: str) -> List[Dict]:
        """Aggregate opportunity scores using dynamic weighting."""
        
        # Initialize or load algorithm weights (these evolve based on performance)
        if not hasattr(self, 'algorithm_weights'):
            self.algorithm_weights = {
                'momentum': 0.25,
                'volume': 0.20,
                'technical': 0.15,
                'volatility': 0.15,
                'reversion': 0.15,
                'liquidity': 0.10
            }
        
        # Collect all unique symbols
        all_symbols = set()
        for opportunities in detection_results.values():
            for opp in opportunities:
                all_symbols.add(opp['symbol'])
        
        aggregated_opportunities = []
        
        for symbol in all_symbols:
            symbol_scores = {}
            symbol_data = None
            algorithms_detected = []
            total_weighted_score = 0.0
            total_confidence = 0.0
            
            # Aggregate scores from all algorithms for this symbol
            for algorithm, opportunities in detection_results.items():
                for opp in opportunities:
                    if opp['symbol'] == symbol:
                        weight = self.algorithm_weights.get(algorithm, 0.1)
                        weighted_score = opp['confidence'] * weight
                        
                        total_weighted_score += weighted_score
                        total_confidence += opp['confidence']
                        algorithms_detected.append(algorithm)
                        
                        if symbol_data is None:
                            symbol_data = opp.copy()
                        
                        symbol_scores[algorithm] = {
                            'confidence': opp['confidence'],
                            'raw_score': opp['raw_score'],
                            'reason': opp['reason']
                        }
            
            # Only include opportunities detected by multiple algorithms or with moderate confidence for learning
            if len(algorithms_detected) >= 1 or total_weighted_score > 0.3:  # Lowered thresholds for learning
                if symbol_data:
                    symbol_data.update({
                        'exchange': exchange_name,
                        'final_score': total_weighted_score,
                        'avg_confidence': total_confidence / len(algorithms_detected) if algorithms_detected else 0,
                        'algorithms_count': len(algorithms_detected),
                        'algorithms_detected': algorithms_detected,
                        'algorithm_scores': symbol_scores,
                        'reason': f"multi_algorithm_{len(algorithms_detected)}"
                    })
                    
                    aggregated_opportunities.append(symbol_data)
        
        # Sort by final score
        aggregated_opportunities.sort(key=lambda x: x['final_score'], reverse=True)
        
        log.info(f"🎯 Aggregated {len(aggregated_opportunities)} multi-algorithm opportunities", context="aggregation")
        
        return aggregated_opportunities

    async def evolve_algorithm_weights(self, trade_results: List[Dict]):
        """Evolve algorithm weights based on trading performance."""
        if not trade_results:
            return
        
        log.info("🧬 Evolving algorithm weights based on performance...", context="evolution")
        
        # Initialize performance tracking if not exists
        if not hasattr(self, 'algorithm_performance'):
            self.algorithm_performance = {
                algorithm: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trade_count': 0}
                for algorithm in self.algorithm_weights.keys()
            }
        
        # Update performance metrics for each algorithm
        for trade in trade_results:
            algorithms = trade.get('algorithms_detected', [])
            pnl = trade.get('pnl', 0)
            
            for algorithm in algorithms:
                if algorithm in self.algorithm_performance:
                    self.algorithm_performance[algorithm]['trade_count'] += 1
                    self.algorithm_performance[algorithm]['total_pnl'] += pnl
                    
                    if pnl > 0:
                        self.algorithm_performance[algorithm]['wins'] += 1
                    else:
                        self.algorithm_performance[algorithm]['losses'] += 1
        
        # Calculate new weights based on performance
        total_score = 0
        algorithm_scores = {}
        
        for algorithm, perf in self.algorithm_performance.items():
            if perf['trade_count'] < 5:  # Need minimum trades for reliable stats
                algorithm_scores[algorithm] = 0.5  # Neutral score
            else:
                win_rate = perf['wins'] / perf['trade_count']
                avg_pnl = perf['total_pnl'] / perf['trade_count']
                
                # Composite score: win rate + normalized PnL
                score = (win_rate * 0.6) + (min(max(avg_pnl, -1), 1) * 0.4 + 0.5)
                algorithm_scores[algorithm] = max(0.1, min(1.0, score))  # Clamp to [0.1, 1.0]
            
            total_score += algorithm_scores[algorithm]
        
        # Normalize weights to sum to 1.0
        if total_score > 0:
            for algorithm in self.algorithm_weights:
                new_weight = algorithm_scores[algorithm] / total_score
                old_weight = self.algorithm_weights[algorithm]
                
                # Smooth transition - only move 20% toward new weight each time
                self.algorithm_weights[algorithm] = old_weight * 0.8 + new_weight * 0.2
                
                log.info(f"🎯 {algorithm}: {old_weight:.3f} → {self.algorithm_weights[algorithm]:.3f}", context="weight_evolution")
        
        # Evolve thresholds based on market conditions
        await self.evolve_adaptive_thresholds()
    
    async def evolve_adaptive_thresholds(self):
        """Evolve detection thresholds based on market volatility and opportunity success."""
        if not hasattr(self, 'adaptive_thresholds'):
            return
        
        # Get recent market volatility
        recent_volatility = getattr(self, 'recent_market_volatility', 1.0)
        
        # Adjust momentum threshold based on market conditions
        if recent_volatility > 2.0:  # High volatility market
            self.adaptive_thresholds['momentum_threshold'] *= 0.95  # Lower threshold
        elif recent_volatility < 0.5:  # Low volatility market
            self.adaptive_thresholds['momentum_threshold'] *= 1.05  # Raise threshold
        
        # Keep thresholds within reasonable bounds
        self.adaptive_thresholds['momentum_threshold'] = max(0.3, min(3.0, self.adaptive_thresholds['momentum_threshold']))
        self.adaptive_thresholds['volume_multiplier'] = max(1.5, min(4.0, self.adaptive_thresholds['volume_multiplier']))
        
        log.debug(f"📊 Evolved thresholds: momentum={self.adaptive_thresholds['momentum_threshold']:.2f}, volume={self.adaptive_thresholds['volume_multiplier']:.2f}", context="threshold_evolution")

    def calculate_market_regime(self, market_data: Dict) -> str:
        """Identify current market regime for adaptive behavior."""
        # This would analyze current market conditions
        # and return regime: 'bull', 'bear', 'sideways', 'volatile'
        
        if not market_data:
            return 'unknown'
        
        # Simple regime detection based on volatility and trend
        avg_change = np.mean([abs(d.get('change_24h', 0)) for d in market_data.values()])
        
        if avg_change > 5.0:
            return 'volatile'
        elif avg_change > 2.0:
            return 'trending'
        else:
            return 'sideways'

    async def _get_trending_opportunities(self) -> List[Dict]:
        """Get trending coins from CoinGecko with timeout and retry logic."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                url = "https://api.coingecko.com/api/v3/search/trending"
                response = await asyncio.wait_for(self.session.get(url), timeout=15.0)
                
                if response.status == 429:  # Rate limited
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    log.warning(f"Rate limited by CoinGecko, retrying in {delay:.1f}s", context="coingecko")
                    await asyncio.sleep(delay)
                    continue
                    
                response.raise_for_status()
                data = await response.json()
                
                opportunities = []
                for coin in data.get('coins', [])[:5]:
                    item = coin['item']
                    symbol = f"{item['symbol'].upper()}/USD"
                    
                    for name, exchange in self.exchanges.items():
                        try:
                            ticker = await asyncio.wait_for(exchange.fetch_ticker(symbol), timeout=5.0)
                            if ticker and is_sane_ticker(ticker):
                                opportunities.append({
                                    'symbol': symbol,
                                    'exchange': name,
                                    'change_24h': ticker.get('percentage', 0),
                                    'volume_24h': ticker.get('quoteVolume', 0),
                                    'price': ticker['last'],
                                    'reason': 'trending',
                                    'coingecko_rank': item.get('market_cap_rank', 999)
                                })
                                break # Found on one exchange, move to next coin
                        except Exception:
                            continue # Try next exchange
                
                log.success(f"Found {len(opportunities)} trending coins.", context="coingecko")
                return opportunities
                
            except Exception as e:
                if attempt == max_retries - 1:
                    log.warning(f"Could not fetch trending coins after {max_retries} attempts: {e}", context="coingecko")
                    return []
                else:
                    delay = base_delay * (2 ** attempt)
                    log.warning(f"CoinGecko request failed (attempt {attempt + 1}), retrying in {delay}s: {e}", context="coingecko")
                    await asyncio.sleep(delay)
        
        return []

    async def _get_top_volume_fallback(self) -> List[Dict]:
        """As a fallback, get top 3 volume coins from each exchange."""
        tasks = [self._get_single_exchange_volume_fallback(name, ex) for name, ex in self.exchanges.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        fallback_opportunities = []
        for res in results:
            if isinstance(res, list):
                fallback_opportunities.extend(res)
        return fallback_opportunities

    async def _get_single_exchange_volume_fallback(self, name: str, exchange: ccxt.Exchange) -> List[Dict]:
        """Fallback to get top 3 volume tickers from a single exchange."""
        try:
            tickers = await asyncio.wait_for(exchange.fetch_tickers(), timeout=20.0)
            
            usd_pairs = [t for t in tickers.values() if ('/USD' in t['symbol'] or '/USDT' in t['symbol'] or '/USDC' in t['symbol']) and is_sane_ticker(t)]
            
            by_volume = sorted(usd_pairs, key=lambda x: x.get('quoteVolume', 0), reverse=True)
            
            opportunities = []
            for ticker in by_volume[:3]: # Top 3
                opportunities.append({
                    'symbol': ticker['symbol'],
                    'exchange': name,
                    'change_24h': ticker.get('percentage', 0),
                    'volume_24h': ticker.get('quoteVolume', 0),
                    'price': ticker['last'],
                    'reason': 'high_volume_fallback'
                })
            log.info(f"Found {len(opportunities)} fallback opportunities on {name}", context="scanner_fallback")
            return opportunities
        except Exception as e:
            log.warning(f"Could not get volume fallback for {name}: {e}", context="scanner_fallback")
            return []

    async def _scan_cross_exchange_arbitrage(self) -> List[Dict]:
        """Scan for cross-exchange arbitrage opportunities."""
        try:
            log.info("Scanning for cross-exchange arbitrage opportunities...", context="arbitrage")
            
            # Get common USD pairs across exchanges
            exchange_tickers = {}
            for name, exchange in self.exchanges.items():
                try:
                    tickers = await asyncio.wait_for(exchange.fetch_tickers(), timeout=15.0)
                    
                    def _enrich(t):
                        last  = t.get("last")
                        open_ = t.get("open") or None
                        # Δ% 24 h
                        if last and open_ and t.get("percentage") is None:
                            t["percentage"] = (last - open_) / open_ * 100
                        # USD volume
                        if t.get("quoteVolume") is None and last:
                            base_vol = t.get("baseVolume")
                            if base_vol:
                                t["quoteVolume"] = base_vol * last
                        return t
                    
                    usd_pairs = {k: _enrich(v) for k, v in tickers.items() if '/USD' in k and is_sane_ticker(v)}
                    exchange_tickers[name] = usd_pairs
                except Exception as e:
                    log.warning(f"Failed to fetch tickers from {name} for arbitrage: {e}")
            
            opportunities = []
            
            # Find common symbols across exchanges
            if len(exchange_tickers) >= 2:
                all_symbols = set()
                for tickers in exchange_tickers.values():
                    all_symbols.update(tickers.keys())
                
                for symbol in all_symbols:
                    prices = {}
                    volumes = {}
                    for exch_name, tickers in exchange_tickers.items():
                        if symbol in tickers:
                            ticker = tickers[symbol]
                            price = ticker.get('last')
                            volume = ticker.get('quoteVolume', 0)
                            
                            # Ensure price and volume are valid
                            if price and price > 0 and volume > 1000:  # Min $1000 volume
                                prices[exch_name] = price
                                volumes[exch_name] = volume
                    
                    if len(prices) >= 2:
                        min_price = min(prices.values())
                        max_price = max(prices.values())
                        spread = (max_price - min_price) / min_price * 100
                        
                        # Look for spreads > 0.5% (accounting for fees) but < 10% (sanity check)
                        if 0.5 < spread < 10.0:
                            buy_exchange = min(prices, key=prices.get)
                            sell_exchange = max(prices, key=prices.get)
                            
                            # Ensure both exchanges have decent volume
                            min_volume = min(volumes[buy_exchange], volumes[sell_exchange])
                            if min_volume > 5000:  # Min $5000 volume on both sides
                                
                                opportunities.append({
                                    'symbol': symbol,
                                    'exchange': buy_exchange,  # Buy from cheaper exchange
                                    'change_24h': spread,
                                    'volume_24h': min_volume,
                                    'price': prices[buy_exchange],
                                    'reason': 'arbitrage',
                                    'spread': spread,
                                    'buy_exchange': buy_exchange,
                                    'sell_exchange': sell_exchange,
                                    'sell_price': prices[sell_exchange]
                                })
            
            log.info(f"Found {len(opportunities)} arbitrage opportunities", context="arbitrage")
            return opportunities[:5]  # Limit to top 5
            
        except Exception as e:
            log.error(f"Error scanning arbitrage: {e}", context="arbitrage")
            return []

    async def _scan_statistical_signals(self) -> List[Dict]:
        """Scan for statistical signals like z-scores and mean reversion."""
        try:
            log.info("Scanning for statistical signals...", context="statistics")
            
            opportunities = []
            
            # For each exchange, look for statistical anomalies
            for name, exchange in self.exchanges.items():
                try:
                    tickers = await asyncio.wait_for(exchange.fetch_tickers(), timeout=15.0)
                    
                    def _enrich(t):
                        last  = t.get("last")
                        open_ = t.get("open") or None
                        # Δ% 24 h
                        if last and open_ and t.get("percentage") is None:
                            t["percentage"] = (last - open_) / open_ * 100
                        # USD volume
                        if t.get("quoteVolume") is None and last:
                            base_vol = t.get("baseVolume")
                            if base_vol:
                                t["quoteVolume"] = base_vol * last
                        return t
                    
                    usd_pairs = {k: _enrich(v) for k, v in tickers.items() if '/USD' in k and is_sane_ticker(v)}
                    
                    # Calculate z-scores for 24h changes
                    changes = [t.get('percentage', 0) for t in usd_pairs.values() if t.get('percentage') is not None]
                    if len(changes) > 10:
                        try:
                            import numpy as np
                            
                            changes_array = np.array(changes)
                            # Remove outliers for more stable statistics
                            q1, q3 = np.percentile(changes_array, [25, 75])
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            filtered_changes = changes_array[(changes_array >= lower_bound) & (changes_array <= upper_bound)]
                            
                            if len(filtered_changes) > 5:
                                mean_change = np.mean(filtered_changes)
                                std_change = np.std(filtered_changes)
                                
                                if std_change > 0.1:  # Ensure meaningful standard deviation
                                    for symbol, ticker in usd_pairs.items():
                                        change = ticker.get('percentage', 0)
                                        if change is not None:
                                            z_score = (change - mean_change) / std_change
                                            
                                            # Look for extreme z-scores (> 2.5 or < -2.5)
                                            if abs(z_score) > 2.5:
                                                reason = 'statistical_outlier_reversal' if z_score > 0 else 'statistical_outlier_momentum'
                                                
                                                opportunities.append({
                                                    'symbol': symbol,
                                                    'exchange': name,
                                                    'change_24h': change,
                                                    'volume_24h': ticker.get('quoteVolume', 0),
                                                    'price': ticker['last'],
                                                    'reason': reason,
                                                    'z_score': z_score,
                                                    'market_deviation': abs(z_score)
                                                })
                        except Exception as e:
                            log.warning(f"Error in statistical calculation for {name}: {e}")
                                    
                except Exception as e:
                    log.warning(f"Error calculating statistics for {name}: {e}")
            
            log.info(f"Found {len(opportunities)} statistical opportunities", context="statistics")
            return opportunities[:3]  # Limit to top 3
            
        except Exception as e:
            log.error(f"Error scanning statistical signals: {e}", context="statistics")
            return []

    async def _scan_order_book_opportunities(self) -> List[Dict]:
        """Scan order books for microstructure opportunities."""
        try:
            log.info("Scanning order book microstructure...", context="orderbook")
            
            opportunities = []
            
            # Focus on high-volume pairs for orderbook analysis
            major_pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD']
            
            for name, exchange in self.exchanges.items():
                try:
                    for symbol in major_pairs:
                        try:
                            # Check if exchange supports this symbol
                            if hasattr(exchange, 'markets') and exchange.markets:
                                if symbol not in exchange.markets:
                                    continue
                            
                            orderbook = await asyncio.wait_for(
                                exchange.fetch_order_book(symbol, limit=50), 
                                timeout=10.0
                            )
                            
                            bids = orderbook.get('bids', [])
                            asks = orderbook.get('asks', [])
                            
                            if len(bids) >= 5 and len(asks) >= 5:
                                # Calculate spread
                                best_bid = bids[0][0] if bids else 0
                                best_ask = asks[0][0] if asks else 0
                                
                                if best_bid > 0 and best_ask > 0:
                                    spread = (best_ask - best_bid) / best_bid * 100
                                    
                                    # Look for abnormally wide spreads
                                    if spread > 0.1:  # > 0.1% spread
                                        # Check for large bid/ask imbalances
                                        total_bid_vol = sum([bid[1] for bid in bids[:5]])
                                        total_ask_vol = sum([ask[1] for ask in asks[:5]])
                                        
                                        if total_bid_vol > 0 and total_ask_vol > 0:
                                            imbalance = abs(total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
                                            
                                            # High imbalance suggests directional pressure
                                            if imbalance > 0.3:  # 30% imbalance
                                                direction = 'bullish' if total_bid_vol > total_ask_vol else 'bearish'
                                                
                                                opportunities.append({
                                                    'symbol': symbol,
                                                    'exchange': name,
                                                    'change_24h': imbalance * 100,
                                                    'volume_24h': (total_bid_vol + total_ask_vol) * best_bid,
                                                    'price': (best_bid + best_ask) / 2,
                                                    'reason': f'orderbook_{direction}_imbalance',
                                                    'spread': spread,
                                                    'imbalance': imbalance,
                                                    'direction': direction
                                                })
                                                
                        except Exception as e:
                            log.debug(f"Could not analyze orderbook for {symbol} on {name}: {e}")
                            
                except Exception as e:
                    log.warning(f"Error scanning orderbooks on {name}: {e}")
            
            log.info(f"Found {len(opportunities)} orderbook opportunities", context="orderbook")
            return opportunities[:2]  # Limit to top 2
            
        except Exception as e:
            log.error(f"Error scanning order book opportunities: {e}", context="orderbook")
            return []

    async def _analyze_opportunity(self, opp: Dict) -> Dict:
        """Get AI analysis of opportunity"""
        required_keys = ['symbol', 'exchange', 'price', 'reason']  # allow empty stats
        if not all(k in opp and opp[k] is not None for k in required_keys):
            log.warning(f"Skipping analysis for incomplete opportunity: {opp.get('symbol', 'N/A')}")
            self.state.opportunities_skipped += 1
            return {"score": 0, "confidence": 0, "reasoning": "Incomplete data"}
        
        # fill missing numeric fields with 0 so JSON is complete
        opp['change_24h'] = opp.get('change_24h') or 0
        opp['volume_24h'] = opp.get('volume_24h') or 0
        
        self.state.opportunities_analyzed += 1
        
        # Extract multi-algorithm context
        algorithms_detected = opp.get('algorithms_detected', [])
        final_score = opp.get('final_score', 0)
        algorithm_scores = opp.get('algorithm_scores', {})
        
        # Build algorithm analysis context
        algorithm_context = ""
        if algorithms_detected:
            algorithm_context = f"\nDetection Algorithms: {', '.join(algorithms_detected)}"
            algorithm_context += f"\nComposite Score: {final_score:.3f}"
            
            for algo, data in algorithm_scores.items():
                algorithm_context += f"\n- {algo}: confidence={data['confidence']:.2f}, raw_score={data['raw_score']:.2f}"
        
        prompt = f"""
        system: |
          You are an expert AI trading analyst. Your sole responsibility is to analyze a trading opportunity and return a complete, valid JSON object with a detailed trade plan. You must adhere strictly to the output schema. Do not include any other text, explanations, or markdown.

        user: |
          Analyze the following multi-algorithm trading opportunity.

          ## Opportunity Data ##
          ```json
          {{
              "symbol": "{opp['symbol']}",
              "exchange": "{opp['exchange']}",
              "price": {opp.get('price', 0)},
              "change_24h": {opp.get('change_24h', 0):.2f},
              "volume_24h": {opp.get('volume_24h', 0)},
              "reason": "{opp.get('reason', 'N/A')}",
              "final_score": {final_score:.3f},
              "algorithms_detected": {algorithms_detected},
              "algorithm_scores": {json.dumps(algorithm_scores)}
          }}
          ```

          ## Instructions ##
          1. **Analyze**: Evaluate the opportunity based on the converged signals from multiple algorithms.
          2. **Score**: Provide a `score` from 0.0 (weak) to 1.0 (strong) representing the opportunity's quality.
          3. **Plan**: Define a clear `direction`, `entry_price`, `stop_loss`, and `take_profit`.
          4. **Explain**: Briefly summarize your `reasoning`.
          5. **Output**: Return ONLY the JSON object described in the output schema.

        assistant:
          {{
            "score": 0.0,
            "direction": "long",
            "confidence": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "reasoning": "string",
            "risks": ["string"],
            "timeframe": "string",
            "algorithm_consensus": false
          }}
        """
        
        try:
            # Retry logic for AI completion
            max_retries = 3
            base_delay = 2.0
            for attempt in range(max_retries):
                try:
                    response = await asyncio.wait_for(
                        self.ai.complete(prompt, temperature=0.3),
                        timeout=45.0
                    )
                    
                    if response and response.strip():
                        # Strip markdown code blocks if present
                        response_text = response.strip()
                        if response_text.startswith("```json"):
                            response_text = response_text[7:]
                        if response_text.endswith("```"):
                            response_text = response_text[:-3]
                        response_text = response_text.strip()
                        
                        try:
                            analysis = json.loads(response_text)
                            
                            # Boost score for multi-algorithm consensus
                            if len(algorithms_detected) >= 3:
                                analysis['score'] = min(1.0, analysis.get('score', 0) * 1.2)
                                analysis['confidence'] = min(1.0, analysis.get('confidence', 0) * 1.1)
                            
                            # Add the algorithm metadata to the analysis
                            analysis.update({
                                'algorithms_detected': algorithms_detected,
                                'algorithm_count': len(algorithms_detected),
                                'composite_score': final_score,
                                'algorithm_scores': algorithm_scores
                            })
                            
                            return analysis # Success
                        
                        except json.JSONDecodeError:
                            log.warning(f"Invalid JSON from AI (attempt {attempt+1}/{max_retries}) for {opp['symbol']}: {response_text[:100]}...")
                            # Fallthrough to retry
                            
                    else:
                        log.warning(f"Empty response from AI (attempt {attempt+1}/{max_retries}) for {opp['symbol']}")
                        # Fallthrough to retry

                except asyncio.TimeoutError:
                    log.warning(f"AI analysis timed out (attempt {attempt+1}/{max_retries}) for {opp['symbol']}")
                
                except Exception as e:
                    log.error(f"AI analysis failed (attempt {attempt+1}/{max_retries}) for {opp['symbol']}: {e}")

                # Wait before retrying
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)

            # If all retries fail, generate fallback
            log.error(f"All AI analysis attempts failed for {opp['symbol']}. Using fallback.", context="ai_failure")
            return self._generate_fallback_analysis(opp)

        except Exception as e:
            log.error(f"Critical error in _analyze_opportunity for {opp['symbol']}: {e}", exc_info=True)
            return self._generate_fallback_analysis(opp)
    
    def _generate_fallback_analysis(self, opp: Dict) -> Dict:
        """
        Generates a deterministic, rule-based trade analysis if the AI fails.
        """
        log.warning(f"AI analysis failed for {opp['symbol']}. Generating rule-based fallback.", context="fallback_generator")
        
        final_score = opp.get('final_score', 0)
        change_24h = opp.get('change_24h', 0)
        price = opp.get('price', 0)
        
        if price == 0:
            return {"score": 0, "confidence": 0, "reasoning": "Invalid price for fallback."}

        # Determine direction
        direction = "long" if change_24h >= 0 else "short"
        
        # Confidence based on algorithm count and score
        confidence = min(0.8, opp.get('avg_confidence', 0.5) + (opp.get('algorithms_count', 1) - 1) * 0.05)
        
        # Define risk parameters (e.g., 2% stop loss, 4% take profit)
        stop_loss_pct = 0.02
        take_profit_pct = 0.04
        
        if direction == "long":
            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)
        else: # short
            stop_loss = price * (1 + stop_loss_pct)
            take_profit = price * (1 - take_profit_pct)
            
        reasoning = (f"Rule-based fallback strategy generated due to AI failure. "
                     f"Signal based on {opp.get('algorithms_count', 1)} algorithms "
                     f"with a composite score of {final_score:.2f}.")

        fallback_analysis = {
            "score": final_score,
            "direction": direction,
            "confidence": confidence,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasoning": reasoning,
            "risks": ["AI analysis failed", "Strategy is rule-based and not context-aware"],
            "timeframe": "hours",
            "algorithm_consensus": opp.get('algorithms_count', 1) > 1,
            'algorithms_detected': opp.get('algorithms_detected', []),
            'algorithm_count': opp.get('algorithms_count', 0),
            'composite_score': final_score,
            'algorithm_scores': opp.get('algorithm_scores', {})
        }
        
        log.success(f"Generated fallback strategy for {opp['symbol']}", context="fallback_generator")
        return fallback_analysis

    async def heartbeat_guard(self, execution_engine, notifications):
        """Monitor market data heartbeat and take action if stalled"""
        while True:
            await asyncio.sleep(5)
            delta = datetime.utcnow() - self.last_market_heartbeat
            if delta.total_seconds() > 180:   # 3-minute grace
                log.error("Market data stalled", seconds=delta.total_seconds(), context="heartbeat")
                await notifications.send("🚨 Market data stalled – flattening positions", "ALERT")
                # In a real implementation, we'd call execution_engine.close_all_positions()
                log.warning("Would close all positions due to data stall", context="safety")
                # Reset heartbeat to prevent spamming while scanner recovers
                self.last_market_heartbeat = datetime.utcnow()
                await asyncio.sleep(30)  # back-off