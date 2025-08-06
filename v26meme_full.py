#!/usr/bin/env python3
"""
v26meme: Autonomous Trading Intelligence
Built to discover patterns, evolve strategies, and achieve $200 → $1M in 90 days.
"""

import os
import sys
import asyncio
import time
import json
import hashlib
import random
import math
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Third-party imports
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import ccxt.async_support as ccxt
import aiohttp
import openai
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
import nest_asyncio
nest_asyncio.apply()
load_dotenv()

# ═══════════════════════════════ CONFIGURATION ═══════════════════════════════

class Config:
    """Central configuration for the autonomous trading system."""
    
    # Core Settings
    VERSION = "26.0.0"
    INITIAL_CAPITAL = 200.0
    TARGET_CAPITAL = 1_000_000.0
    TARGET_DAYS = 90
    
    # API Keys (loaded from environment)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Exchange Configuration
    @staticmethod
    def get_exchanges():
        exchanges = {}
        
        # Coinbase (for USA users)
        if os.getenv('COINBASE_API_KEY') and os.getenv('COINBASE_SECRET'):
            exchanges['coinbase'] = {
                'apiKey': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET'),
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }

        # Kraken (for USA users)
        if os.getenv('KRAKEN_API_KEY') and os.getenv('KRAKEN_SECRET'):
            exchanges['kraken'] = {
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_SECRET'),
                'enableRateLimit': True
            }
            
        return exchanges
    
    # Trading Parameters
    MIN_TRADE_SIZE = 10.0  # Minimum $10 per trade
    MAX_POSITION_SIZE = 0.1  # Max 10% of equity per position
    STOP_LOSS_PCT = 0.05  # 5% stop loss
    TAKE_PROFIT_PCT = 0.15  # 15% take profit
    
    # Pattern Discovery
    PATTERN_DISCOVERY_INTERVAL = 3600  # Run every hour
    MIN_PATTERN_CONFIDENCE = 0.6
    MIN_PATTERN_SAMPLES = 10
    
    # Strategy Evolution
    STRATEGY_EVOLUTION_INTERVAL = 7200  # Evolve every 2 hours
    MAX_STRATEGIES = 100
    STRATEGY_GENERATIONS = 10
    
    # Risk Management
    MAX_DAILY_LOSS = 0.1  # Max 10% daily loss
    MAX_DRAWDOWN = 0.2  # Max 20% drawdown
    KELLY_FRACTION = 0.25  # Conservative Kelly
    
    # Database
    DB_PATH = "v26meme.db"
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# ═══════════════════════════════ LOGGING SETUP ═══════════════════════════════

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(f"v26meme_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
console = Console()

# ═══════════════════════════════ DATA STRUCTURES ═══════════════════════════════

class TradingMode(Enum):
    """Operating modes for the trading system."""
    PAPER = "paper"        # Testing strategies with fake money
    MICRO = "micro"        # Real money, tiny positions ($10-50)
    ACTIVE = "active"      # Full trading with risk management
    EMERGENCY = "emergency" # Something went wrong, minimal activity

class StrategyStatus(Enum):
    """Lifecycle states for trading strategies."""
    PAPER = "paper"        # Being tested in simulation
    MICRO = "micro"        # Testing with small real positions
    ACTIVE = "active"      # Fully deployed
    RETIRED = "retired"    # No longer used

@dataclass
class Pattern:
    """Represents a discovered market pattern."""
    id: str
    type: str  # e.g., 'psychological_level', 'volume_spike'
    symbol: str
    exchange: str
    description: str
    confidence: float
    win_rate: float
    avg_return: float
    sample_size: int
    parameters: Dict[str, Any]
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    success_count: int = 0

@dataclass
class Strategy:
    """Represents a trading strategy generated from patterns."""
    id: str
    name: str
    description: str
    code: str  # Python code for the strategy
    status: StrategyStatus
    pattern_id: str
    generation: int
    parent_id: Optional[str] = None
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_profit: float = 0.0
    position_multiplier: float = 1.0
    max_position_pct: float = 0.02
    stop_loss: float = 0.05
    take_profit: float = 0.15
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_trade: Optional[datetime] = None
    trade_history: List[Dict] = field(default_factory=list)

@dataclass
class Position:
    """Represents an open trading position."""
    id: str
    strategy_id: str
    symbol: str
    exchange: str
    side: str  # 'buy' or 'sell'
    amount: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=datetime.utcnow)
    pnl: float = 0.0
    pnl_pct: float = 0.0

@dataclass
class SystemState:
    """Current state of the entire trading system."""
    mode: TradingMode = TradingMode.PAPER
    equity: float = Config.INITIAL_CAPITAL
    cash: float = Config.INITIAL_CAPITAL
    positions: Dict[str, Position] = field(default_factory=dict)
    strategies: Dict[str, Strategy] = field(default_factory=dict)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    trades_today: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)

# ═══════════════════════════════ UTILITY FUNCTIONS ═══════════════════════════════

def is_sane_ticker(symbol: str) -> bool:
    """Check if a ticker symbol is valid for trading."""
    if not symbol or '/' not in symbol:
        return False
    
    # Avoid test/delisted tokens
    blacklist = ['TEST', 'DEMO', 'DOWN', 'UP', 'BULL', 'BEAR', '2S', '2L', '3S', '3L']
    for term in blacklist:
        if term in symbol.upper():
            return False
    
    # Must be against major stablecoins or BTC/ETH
    valid_quotes = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']
    quote = symbol.split('/')[1]
    
    return quote in valid_quotes

def calculate_kelly_position(win_rate: float, avg_win: float, avg_loss: float, 
                            kelly_fraction: float = 0.25) -> float:
    """
    Calculate optimal position size using Kelly Criterion.
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average profit when winning
        avg_loss: Average loss when losing (positive number)
        kelly_fraction: Fraction of Kelly to use (default 0.25 for safety)
    
    Returns:
        Fraction of capital to risk (0-1)
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    # Kelly formula: f = (p*b - q) / b
    # where p = win_rate, q = 1-win_rate, b = avg_win/avg_loss
    q = 1 - win_rate
    b = avg_win / avg_loss
    
    kelly = (win_rate * b - q) / b
    
    # Apply fraction and cap
    position_size = kelly * kelly_fraction
    
    # Never risk more than 10% on a single trade
    return min(max(0.0, position_size), 0.1)

def extract_params(code: str) -> Dict[str, Any]:
    """Extract numeric parameters from strategy code for mutation."""
    params = {}
    
    # Find common parameter patterns
    import re
    
    # Look for assignments like: threshold = 0.5
    number_pattern = r'(\w+)\s*=\s*([\d.]+)'
    for match in re.finditer(number_pattern, code):
        var_name = match.group(1)
        value = float(match.group(2))
        params[var_name] = value
    
    return params

def inject_params(code: str, params: Dict[str, Any]) -> str:
    """Inject mutated parameters back into strategy code."""
    import re
    
    for param, value in params.items():
        # Replace the parameter value in the code
        pattern = f'{param}\\s*=\\s*[\\d.]+'
        replacement = f'{param} = {value}'
        code = re.sub(pattern, replacement, code)
    
    return code

# ═══════════════════════════════ DATABASE ═══════════════════════════════

class Database:
    """SQLite database for persistent storage."""
    
    def __init__(self, db_path: str = Config.DB_PATH):
        import sqlite3
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        import sqlite3
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Drop old tables to recreate with correct schema
        cursor.execute('DROP TABLE IF EXISTS system_state')
        cursor.execute('DROP TABLE IF EXISTS strategies')
        
        # Patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                type TEXT,
                symbol TEXT,
                exchange TEXT,
                description TEXT,
                confidence REAL,
                win_rate REAL,
                avg_return REAL,
                sample_size INTEGER,
                parameters TEXT,
                discovered_at TIMESTAMP,
                last_seen TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0
            )
        ''')
        
        # Strategies table - fixed schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                code TEXT,
                status TEXT,
                pattern_id TEXT,
                generation INTEGER,
                parent_id TEXT,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                avg_profit REAL DEFAULT 0,
                position_multiplier REAL DEFAULT 1.0,
                max_position_pct REAL DEFAULT 0.02,
                stop_loss REAL DEFAULT 0.05,
                take_profit REAL DEFAULT 0.15,
                created_at TIMESTAMP,
                last_trade TIMESTAMP,
                trade_history TEXT
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                strategy_id TEXT,
                symbol TEXT,
                exchange TEXT,
                side TEXT,
                amount REAL,
                price REAL,
                pnl REAL,
                pnl_pct REAL,
                opened_at TIMESTAMP,
                closed_at TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES strategies (id)
            )
        ''')
        
        # System state table - fixed schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY,
                mode TEXT,
                equity REAL,
                cash REAL,
                daily_pnl REAL,
                total_pnl REAL,
                win_rate REAL,
                trades_today INTEGER,
                positions TEXT,
                start_time TIMESTAMP,
                last_update TIMESTAMP,
                timestamp TIMESTAMP
            )
        ''')
        
        self.conn.commit()

    async def save_pattern(self, pattern: Pattern):
        """Save a discovered pattern to database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.id,
            pattern.type,
            pattern.symbol,
            pattern.exchange,
            pattern.description,
            pattern.confidence,
            pattern.win_rate,
            pattern.avg_return,
            pattern.sample_size,
            json.dumps(pattern.parameters),
            pattern.discovered_at.isoformat(),
            pattern.last_seen.isoformat(),
            pattern.usage_count,
            pattern.success_count
        ))
        self.conn.commit()

    async def save_strategy(self, strategy: Strategy):
        """Save a strategy to database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO strategies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy.id,
            strategy.name,
            strategy.description,
            strategy.code,
            strategy.status.value,
            strategy.pattern_id,
            strategy.generation,
            strategy.parent_id,
            strategy.total_trades,
            strategy.winning_trades,
            strategy.total_pnl,
            strategy.win_rate,
            strategy.sharpe_ratio,
            strategy.max_drawdown,
            strategy.avg_profit,
            strategy.position_multiplier,
            strategy.max_position_pct,
            strategy.stop_loss,
            strategy.take_profit,
            strategy.created_at.isoformat(),
            strategy.last_trade.isoformat() if strategy.last_trade else None,
            json.dumps(strategy.trade_history)
        ))
        self.conn.commit()

    async def save_trade(self, trade: Dict):
        """Save a completed trade."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['id'],
            trade['strategy_id'],
            trade['symbol'],
            trade['exchange'],
            trade['side'],
            trade['amount'],
            trade['price'],
            trade['pnl'],
            trade['pnl_pct'],
            trade['opened_at'],
            trade['closed_at']
        ))
        self.conn.commit()

    async def save_state(self, state: SystemState):
        """Save system state snapshot."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO system_state VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            state.mode.value,
            state.equity,
            state.cash,
            state.daily_pnl,
            state.total_pnl,
            state.win_rate,
            state.trades_today,
            json.dumps({pid: asdict(pos) for pid, pos in state.positions.items()}),
            state.start_time.isoformat(),
            state.last_update.isoformat(),
            datetime.utcnow().isoformat()
        ))
        self.conn.commit()

    async def load_patterns(self) -> List[Pattern]:
        """Load all patterns from database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM patterns ORDER BY confidence DESC')
        rows = cursor.fetchall()
        
        patterns = []
        for row in rows:
            pattern = Pattern(
                id=row[0],
                type=row[1],
                symbol=row[2],
                exchange=row[3],
                description=row[4],
                confidence=row[5],
                win_rate=row[6],
                avg_return=row[7],
                sample_size=row[8],
                parameters=json.loads(row[9]),
                discovered_at=datetime.fromisoformat(row[10]),
                last_seen=datetime.fromisoformat(row[11]),
                usage_count=row[12],
                success_count=row[13]
            )
            patterns.append(pattern)
        
        return patterns

    async def load_strategies(self) -> List[Strategy]:
        """Load all strategies from database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM strategies WHERE status != "retired"')
        rows = cursor.fetchall()
        
        strategies = []
        for row in rows:
            strategy = Strategy(
                id=row[0],
                name=row[1],
                description=row[2],
                code=row[3],
                status=StrategyStatus(row[4]),
                pattern_id=row[5],
                generation=row[6],
                parent_id=row[7],
                total_trades=row[8],
                winning_trades=row[9],
                total_pnl=row[10],
                win_rate=row[11],
                sharpe_ratio=row[12],
                max_drawdown=row[13],
                avg_profit=row[14],
                position_multiplier=row[15],
                max_position_pct=row[16],
                stop_loss=row[17],
                take_profit=row[18],
                created_at=datetime.fromisoformat(row[19]),
                last_trade=datetime.fromisoformat(row[20]) if row[20] else None,
                trade_history=json.loads(row[21]) if row[21] else []
            )
            strategies.append(strategy)
        
        return strategies

# ═══════════════════════════════ LEARNING MEMORY ═══════════════════════════════

class LearningMemory:
    """
    Manages the bot's learned knowledge about market patterns and strategies.
    This is the brain's long-term memory.
    """
    
    def __init__(self, db: Database):
        self.db = db
        self.patterns = {}
        self.pattern_performance = defaultdict(list)
        
    async def initialize(self):
        """Load existing patterns from database."""
        patterns = await self.db.load_patterns()
        for pattern in patterns:
            self.patterns[pattern.id] = pattern
        log.info(f"📚 Loaded {len(self.patterns)} patterns from memory")
    
    async def save_pattern(self, pattern_dict: Dict):
        """Save a newly discovered pattern."""
        # Generate unique ID
        pattern_id = hashlib.sha256(
            f"{pattern_dict['type']}_{pattern_dict.get('symbol', 'all')}_{pattern_dict.get('exchange', 'all')}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        pattern = Pattern(
            id=pattern_id,
            type=pattern_dict['type'],
            symbol=pattern_dict.get('symbol', 'ALL'),
            exchange=pattern_dict.get('exchange', 'ALL'),
            description=pattern_dict['description'],
            confidence=pattern_dict['confidence'],
            win_rate=pattern_dict.get('win_rate', 0.5),
            avg_return=pattern_dict.get('avg_return', 0.0),
            sample_size=pattern_dict.get('sample_size', 0),
            parameters=pattern_dict
        )
        
        self.patterns[pattern_id] = pattern
        await self.db.save_pattern(pattern)
        return pattern_id
    
    async def update_pattern(self, pattern: Pattern):
        """Update an existing pattern's metrics."""
        self.patterns[pattern.id] = pattern
        await self.db.save_pattern(pattern)
    
    async def delete_pattern(self, pattern_id: str):
        """Remove a pattern from memory."""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            # Note: Not deleting from DB, just marking as low confidence
    
    async def get_high_confidence_patterns(self, limit: int = 10) -> List[Dict]:
        """Get the most promising patterns for strategy generation."""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.confidence * p.win_rate * (1 + p.avg_return),
            reverse=True
        )
        
        result = []
        for pattern in sorted_patterns[:limit]:
            result.append({
                'id': pattern.id,
                'type': pattern.type,
                'symbol': pattern.symbol,
                'exchange': pattern.exchange,
                'description': pattern.description,
                'confidence': pattern.confidence,
                'win_rate': pattern.win_rate,
                'avg_return': pattern.avg_return,
                'sample_size': pattern.sample_size,
                'parameters': pattern.parameters
            })
        
        return result
    
    async def get_all_patterns(self) -> List[Dict]:
        """Get all patterns in memory."""
        return [asdict(p) for p in self.patterns.values()]
    
    async def record_pattern_usage(self, pattern_id: str, success: bool, return_pct: float):
        """Track how well a pattern performs when used in strategies."""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.usage_count += 1
            if success:
                pattern.success_count += 1
            
            # Update win rate with Bayesian update
            pattern.win_rate = (pattern.success_count + 1) / (pattern.usage_count + 2)
            
            # Track performance
            self.pattern_performance[pattern_id].append({
                'timestamp': datetime.utcnow(),
                'success': success,
                'return': return_pct
            })
            
            # Update average return
            recent_returns = [p['return'] for p in self.pattern_performance[pattern_id][-50:]]
            if recent_returns:
                pattern.avg_return = np.mean(recent_returns)
            
            pattern.last_seen = datetime.utcnow()
            await self.update_pattern(pattern)

# ═══════════════════════════════ OPENAI MANAGER ═══════════════════════════════

class OpenAIManager:
    """Manages interactions with OpenAI API for strategy generation."""
    
    def __init__(self, api_key: str):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        self.total_tokens_used = 0
        self.generation_count = 0
        
    async def generate_strategy(self, prompt: str) -> str:
        """
        Generate strategy code from a prompt using GPT-4.
        Returns Python code for the strategy.
        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Python programmer specializing in algorithmic trading. Generate complete, working async Python functions for trading strategies."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            code = response.choices[0].message.content
            
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens
                self.generation_count += 1
                log.info(f"🤖 Generated strategy #{self.generation_count} ({response.usage.total_tokens} tokens)")

            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            return code.strip()
            
        except Exception as e:
            log.error(f"OpenAI generation failed: {e}")
            return """
async def execute_strategy(state, opp):
    '''Fallback strategy due to generation error'''
    return {"action": "hold", "conf": 0.0}
"""

    async def analyze_market_sentiment(self, news_items: List[str]) -> Dict[str, float]:
        """
        Analyze sentiment from news items.
        Returns sentiment scores for different aspects.
        """
        if not news_items:
            return {"overall": 0.0, "confidence": 0.0}
        
        try:
            prompt = f"""
Analyze the sentiment of these crypto market news items.
Return scores from -1 (very bearish) to +1 (very bullish):

News items:
{chr(10).join(news_items[:10])}

Provide JSON with: overall, bitcoin, altcoins, defi, confidence
"""
            
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            sentiment = json.loads(content)
            return sentiment
            
        except Exception as e:
            log.error(f"Sentiment analysis failed: {e}")
            return {"overall": 0.0, "confidence": 0.0}

# ═══════════════════════════════ TELEGRAM NOTIFIER ═══════════════════════════════

class TelegramNotifier:
    """Sends important alerts and updates via Telegram."""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.last_message_time = {}
        self.rate_limit = 1  # Minimum seconds between messages
        
    async def send_alert(self, message: str, parse_mode: str = "Markdown"):
        """Send an alert message to Telegram."""
        # Rate limiting
        current_time = time.time()
        if message in self.last_message_time:
            if current_time - self.last_message_time[message] < 60:  # Don't repeat same message within 60s
                return
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                data = {
                    "chat_id": self.chat_id,
                    "text": message[:4096],  # Telegram message limit
                    "parse_mode": parse_mode
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.last_message_time[message] = current_time
                        log.debug(f"📨 Telegram alert sent")
                    else:
                        log.error(f"Telegram send failed: {await response.text()}")
                        
        except Exception as e:
            log.error(f"Telegram notification error: {e}")
    
    async def send_daily_report(self, state: SystemState, top_strategies: List[Strategy]):
        """Send daily performance report."""
        # Calculate metrics
        roi = (state.equity - Config.INITIAL_CAPITAL) / Config.INITIAL_CAPITAL * 100
        days_running = (datetime.utcnow() - state.start_time).days + 1
        daily_roi = roi / days_running if days_running > 0 else 0
        
        # Build report
        report = f"""
📊 **v26meme Daily Report**

💰 **Performance**
• Equity: ${state.equity:.2f}
• ROI: {roi:+.1f}%
• Daily Avg: {daily_roi:+.1f}%
• Today's P&L: ${state.daily_pnl:+.2f}
🎯 **Strategy Performance**
"""
        
        for i, strategy in enumerate(top_strategies[:5], 1):
            report += f"\n{i}. {strategy.name}"
            report += f"\n   Win Rate: {strategy.win_rate:.1%} | Sharpe: {strategy.sharpe_ratio:.2f}"
            report += f"\n   P&L: ${strategy.total_pnl:.2f} | Trades: {strategy.total_trades}"
        
        report += f"\n\n📈 **Active Positions**: {len(state.positions)}"
        report += f"\n🧬 **Active Strategies**: {len([s for s in state.strategies.values() if s.status == StrategyStatus.ACTIVE])}"
        
        # Add phase-specific info
        phase = self._determine_growth_phase(state)
        report += f"\n\n🚀 **Growth Phase**: {phase}"
        
        if phase == "Discovery":
            report += "\n*Focus: Finding patterns, testing many strategies*"
        elif phase == "Optimization":
            report += "\n*Focus: Refining winners, increasing position sizes*"
        elif phase == "Scaling":
            report += "\n*Focus: Maximizing proven strategies*"
        else:
            report += "\n*Focus: Maintaining dominance, risk management*"
        
        await self.send_alert(report)
    
    def _determine_growth_phase(self, state: SystemState) -> str:
        """Determines the current growth phase based on equity."""
        equity = state.equity
        initial = Config.INITIAL_CAPITAL
        
        if equity < initial * 5:  # Up to $1K
            return "Discovery"
        elif equity < initial * 50:  # Up to $10K
            return "Optimization"
        elif equity < initial * 500:  # Up to $100K
            return "Scaling"
        else:  # Above $100K
            return "Domination"

# ═══════════════════════════ PATTERN DISCOVERY ENGINE ═════════════════════════

class PatternDiscoveryEngine:
    """
    Autonomously discovers market inefficiencies and behavioral patterns.
    This engine analyzes market data, social feeds, and on-chain events
    to find repeatable, exploitable trading setups.
    """
    def __init__(self, learning_memory: LearningMemory, exchanges: Dict[str, ccxt.Exchange], ai_manager: OpenAIManager):
        self.learning_memory = learning_memory
        self.exchanges = exchanges
        self.ai_manager = ai_manager
        self.cache = {}  # Cache for expensive calculations
        self.pattern_history = {}  # Track pattern performance
        log.info("🧠 Pattern Discovery Engine initialized.")

    async def discover_patterns(self, state: SystemState):
        """
        The core loop for discovering new patterns. This runs periodically.
        """
        log.info("🔬 Starting pattern discovery cycle...")
        
        # Run all discovery methods in parallel for speed
        tasks = [
            self._discover_price_action_patterns(),
            self._discover_volume_patterns(),
            self._discover_time_based_patterns(),
            self._discover_correlation_patterns(),
            self._discover_psychological_patterns(),
            self._discover_volatility_patterns(),
            self._discover_microstructure_patterns()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any errors but don't crash
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error(f"Pattern discovery task {i} failed: {result}")
        
        # Clean up old patterns with low confidence
        await self._cleanup_low_confidence_patterns()
        
        log.info("✅ Pattern discovery cycle complete.")

    async def _discover_price_action_patterns(self):
        """
        Analyzes OHLCV data to find technical patterns like support/resistance,
        round number effects, or time-of-day anomalies.
        """
        log.debug("🔍 Discovering price action patterns...")
        
        # Get all active symbols across exchanges
        all_symbols = set()
        for exchange in self.exchanges.values():
            markets = await exchange.load_markets()
            all_symbols.update([s for s in markets.keys() if is_sane_ticker(s)])
        
        # Analyze top 50 by volume
        for symbol in list(all_symbols)[:50]:
            try:
                # Try each exchange until we get data
                for exchange_name, exchange in self.exchanges.items():
                    try:
                        # Get multiple timeframes for comprehensive analysis
                        ohlcv_5m = await exchange.fetch_ohlcv(symbol, '5m', limit=1000)
                        ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=500)
                        
                        if not ohlcv_5m or not ohlcv_1h:
                            continue
                            
                        df_5m = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        
                        # 1. Support/Resistance Levels
                        await self._find_support_resistance(df_1h, symbol, exchange_name)
                        
                        # 2. Breakout Patterns
                        await self._find_breakout_patterns(df_5m, symbol, exchange_name)
                        
                        # 3. Mean Reversion Opportunities
                        await self._find_mean_reversion(df_1h, symbol, exchange_name)
                        
                        break  # Got data, move to next symbol
                        
                    except Exception as e:
                        continue  # Try next exchange
                        
            except Exception as e:
                log.error(f"Error analyzing {symbol}: {e}")

    async def _find_support_resistance(self, df: pd.DataFrame, symbol: str, exchange: str):
        """Find significant support and resistance levels"""
        prices = df['close'].values
        
        # Find local minima and maxima
        from scipy.signal import find_peaks
        
        # Find resistance levels (peaks)
        peaks, properties = find_peaks(prices, distance=20, prominence=prices.std() * 0.5)
        
        # Find support levels (inverted peaks)
        troughs, _ = find_peaks(-prices, distance=20, prominence=prices.std() * 0.5)
        
        # Cluster nearby levels
        significant_levels = []
        
        for idx in peaks[-10:]:  # Last 10 peaks
            level = prices[idx]
            # Count how many times price reversed near this level
            touches = len(df[(df['high'] >= level * 0.99) & (df['high'] <= level * 1.01)])
            
            if touches >= 3:  # At least 3 touches
                bounces = len(df[(df['high'] >= level * 0.99) & (df['close'] < level * 0.98)])
                bounce_rate = bounces / touches if touches > 0 else 0
                
                if bounce_rate > 0.6:  # 60% bounce rate
                    pattern = {
                        'type': 'resistance_level',
                        'symbol': symbol,
                        'exchange': exchange,
                        'level': float(level),
                        'touches': int(touches),
                        'bounce_rate': float(bounce_rate),
                        'description': f"{symbol} shows strong resistance at ${level:.4f} with {bounce_rate:.1%} rejection rate",
                        'confidence': min(0.95, bounce_rate * (touches / 10)),
                        'avg_bounce': float(df[df['high'] >= level * 0.99]['high'].mean() - df[df['high'] >= level * 0.99]['close'].mean()) / level,
                        'last_test': df['timestamp'].iloc[-1]
                    }
                    
                    await self.learning_memory.save_pattern(pattern)
                    log.info(f"💡 Discovered resistance: {pattern['description']}")

    async def _find_breakout_patterns(self, df: pd.DataFrame, symbol: str, exchange: str):
        """Find consolidation patterns that lead to breakouts"""
        
        # Calculate rolling volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['avg_volume'] = df['volume'].rolling(window=20).mean()
        
        # Find periods of low volatility (consolidation)
        low_vol_threshold = df['volatility'].quantile(0.2)
        consolidation_periods = []
        
        in_consolidation = False
        start_idx = 0
        
        for i in range(20, len(df)):
            if df['volatility'].iloc[i] < low_vol_threshold and not in_consolidation:
                in_consolidation = True
                start_idx = i
            elif df['volatility'].iloc[i] > low_vol_threshold * 2 and in_consolidation:
                # Consolidation ended, check for breakout
                in_consolidation = False
                consolidation_length = i - start_idx
                
                if consolidation_length >= 10:  # At least 10 periods
                    # Check what happened after consolidation
                    if i + 10 < len(df):
                        breakout_move = (df['close'].iloc[i+10] - df['close'].iloc[i]) / df['close'].iloc[i]
                        volume_spike = df['volume'].iloc[i] / df['avg_volume'].iloc[i]
                        
                        if abs(breakout_move) > 0.05 and volume_spike > 2:  # 5% move with 2x volume
                            pattern = {
                                'type': 'consolidation_breakout',
                                'symbol': symbol,
                                'exchange': exchange,
                                'consolidation_length': int(consolidation_length),
                                'breakout_direction': 'up' if breakout_move > 0 else 'down',
                                'breakout_magnitude': float(abs(breakout_move)),
                                'volume_spike': float(volume_spike),
                                'description': f"{symbol} breaks {breakout_move:+.1%} after {consolidation_length} periods of consolidation",
                                'confidence': min(0.9, abs(breakout_move) * 10 * (volume_spike / 3)),
                                'entry_volatility': float(low_vol_threshold),
                                'timestamp': int(df['timestamp'].iloc[i])
                            }
                            
                            consolidation_periods.append(pattern)
        
        # Analyze historical success rate
        if len(consolidation_periods) >= 5:
            successful = [p for p in consolidation_periods if p['breakout_magnitude'] > 0.05]
            success_rate = len(successful) / len(consolidation_periods)
            
            if success_rate > 0.6:  # 60% success rate
                meta_pattern = {
                    'type': 'consolidation_breakout_meta',
                    'symbol': symbol,
                    'exchange': exchange,
                    'success_rate': float(success_rate),
                    'avg_breakout': float(np.mean([p['breakout_magnitude'] for p in successful])),
                    'avg_consolidation': float(np.mean([p['consolidation_length'] for p in consolidation_periods])),
                    'description': f"{symbol} consolidation breakouts have {success_rate:.1%} success rate with {np.mean([p['breakout_magnitude'] for p in successful]):.1%} avg move",
                    'confidence': float(success_rate * 0.8),
                    'sample_size': len(consolidation_periods),
                    'win_rate': float(success_rate)
                }
                
                await self.learning_memory.save_pattern(meta_pattern)
                log.info(f"💡 Discovered breakout pattern: {meta_pattern['description']}")

    async def _discover_volume_patterns(self):
        """Discover patterns related to volume spikes and unusual activity"""
        log.debug("📊 Discovering volume patterns...")
        
        # Focus on top movers
        for exchange_name, exchange in self.exchanges.items():
            try:
                tickers = await exchange.fetch_tickers()
                
                # Sort by volume
                sorted_tickers = sorted(
                    [(s, t) for s, t in tickers.items() if is_sane_ticker(s) and t.get('quoteVolume') is not None and t['quoteVolume'] > 0],
                    key=lambda x: x[1].get('quoteVolume', 0),
                    reverse=True
                )[:30]  # Top 30 by volume
                
                for symbol, ticker in sorted_tickers:
                    try:
                        # Get historical data
                        ohlcv = await exchange.fetch_ohlcv(symbol, '15m', limit=500)
                        if not ohlcv:
                            continue
                            
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        
                        # Volume analysis
                        df['volume_ma'] = df['volume'].rolling(window=20).mean()
                        df['volume_std'] = df['volume'].rolling(window=20).std()
                        df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume_std']
                        df['price_change'] = df['close'].pct_change()
                        
                        # Find extreme volume spikes
                        volume_spikes = df[df['volume_zscore'] > 3].copy()
                        
                        if len(volume_spikes) >= 5:
                            # Analyze what happens after volume spikes
                            results = []
                            
                            for idx in volume_spikes.index:
                                if idx + 4 < len(df):  # Need 4 periods after
                                    subsequent_return = (df['close'].iloc[idx+4] - df['close'].iloc[idx]) / df['close'].iloc[idx]
                                    results.append({
                                        'volume_zscore': df['volume_zscore'].iloc[idx],
                                        'immediate_change': df['price_change'].iloc[idx],
                                        'subsequent_return': subsequent_return
                                    })
                            
                            if len(results) >= 5:
                                # Statistical analysis
                                avg_subsequent = np.mean([r['subsequent_return'] for r in results])
                                win_rate = len([r for r in results if r['subsequent_return'] > 0]) / len(results)
                                
                                if abs(avg_subsequent) > 0.02 and win_rate > 0.65:  # 2% avg move, 65% win rate
                                    pattern = {
                                        'type': 'volume_spike_momentum',
                                        'symbol': symbol,
                                        'exchange': exchange_name,
                                        'avg_zscore': float(np.mean([r['volume_zscore'] for r in results])),
                                        'avg_subsequent_return': float(avg_subsequent),
                                        'win_rate': float(win_rate),
                                        'sample_size': len(results),
                                        'description': f"{symbol} shows {avg_subsequent:+.1%} avg move after 3+ sigma volume spikes ({win_rate:.1%} win rate)",
                                        'confidence': float(win_rate * min(1.0, len(results) / 10)),
                                        'direction': 'bullish' if avg_subsequent > 0 else 'bearish',
                                        'optimal_hold_periods': 4,  # 15min * 4 = 1 hour
                                        'avg_return': float(avg_subsequent)
                                    }
                                    
                                    await self.learning_memory.save_pattern(pattern)
                                    log.info(f"💡 Discovered volume pattern: {pattern['description']}")
                                    
                    except Exception as e:
                        continue
                        
            except Exception as e:
                log.error(f"Error discovering volume patterns on {exchange_name}: {e}")

    async def _discover_time_based_patterns(self):
        """Discover patterns based on time of day, day of week, etc."""
        log.debug("⏰ Discovering time-based patterns...")
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Get a diverse set of symbols
                tickers = await exchange.fetch_tickers()
                symbols = [s for s in tickers.keys() if is_sane_ticker(s)][:20]
                
                for symbol in symbols:
                    try:
                        # Get hourly data for analysis
                        ohlcv = await exchange.fetch_ohlcv(symbol, '1h', limit=24*30)  # 30 days
                        if not ohlcv or len(ohlcv) < 24*7:  # Need at least a week
                            continue
                            
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df['hour'] = df['timestamp'].dt.hour
                        df['day_of_week'] = df['timestamp'].dt.dayofweek
                        df['returns'] = df['close'].pct_change()
                        
                        # Analyze hourly patterns
                        hourly_stats = df.groupby('hour')['returns'].agg(['mean', 'std', 'count'])
                        hourly_stats['sharpe'] = hourly_stats['mean'] / hourly_stats['std'] * np.sqrt(252*24)  # Annualized
                        
                        # Find best and worst hours
                        best_hours = hourly_stats[hourly_stats['sharpe'] > 1.0].sort_values('sharpe', ascending=False)
                        
                        for hour, stats in best_hours.iterrows():
                            if stats['count'] >= 20:  # Enough samples
                                win_rate = len(df[(df['hour'] == hour) & (df['returns'] > 0)]) / stats['count']
                                
                                if win_rate > 0.6 and stats['mean'] > 0.001:  # 60% win rate, 0.1% avg return
                                    pattern = {
                                        'type': 'time_of_day',
                                        'symbol': symbol,
                                        'exchange': exchange_name,
                                        'hour_utc': int(hour),
                                        'avg_return': float(stats['mean']),
                                        'sharpe_ratio': float(stats['sharpe']),
                                        'win_rate': float(win_rate),
                                        'sample_size': int(stats['count']),
                                        'description': f"{symbol} shows {stats['mean']:.2%} avg return at {hour}:00 UTC (Sharpe: {stats['sharpe']:.2f})",
                                        'confidence': float(min(0.9, win_rate * (stats['count'] / 30))),
                                        'time_window': f"{hour}:00-{hour}:59 UTC"
                                    }
                                    
                                    await self.learning_memory.save_pattern(pattern)
                                    log.info(f"💡 Discovered time pattern: {pattern['description']}")
                        
                        # Analyze day of week patterns
                        daily_stats = df.groupby('day_of_week')['returns'].agg(['mean', 'std', 'count'])
                        best_days = daily_stats[daily_stats['mean'] > daily_stats['mean'].mean() * 2]
                        
                        for day, stats in best_days.iterrows():
                            if stats['count'] >= 4:  # At least 4 weeks of data
                                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                                win_rate = len(df[(df['day_of_week'] == day) & (df['returns'] > 0)]) / stats['count']
                                
                                if win_rate > 0.65 and stats['mean'] > 0.002:
                                    pattern = {
                                        'type': 'day_of_week',
                                        'symbol': symbol,
                                        'exchange': exchange_name,
                                        'day': int(day),
                                        'day_name': day_names[day],
                                        'avg_return': float(stats['mean']),
                                        'win_rate': float(win_rate),
                                        'description': f"{symbol} shows {stats['mean']:.2%} avg return on {day_names[day]}s",
                                        'confidence': float(win_rate * 0.8),
                                        'sample_size': int(stats['count'])
                                    }
                                    
                                    await self.learning_memory.save_pattern(pattern)
                                    log.info(f"💡 Discovered weekly pattern: {pattern['description']}")
                                    
                    except Exception as e:
                        continue
                        
            except Exception as e:
                log.error(f"Error discovering time patterns: {e}")

    async def _discover_psychological_patterns(self):
        """Discover patterns around psychological price levels ($0.99, $9.99, etc)"""
        log.debug("🧠 Discovering psychological patterns...")
        
        # Define psychological levels - these are where human psychology affects trading
        psychological_levels = [
            0.01, 0.05, 0.10, 0.25, 0.50, 0.99, 1.00,
            5.00, 9.99, 10.00, 25.00, 50.00, 99.00, 100.00,
            250.00, 500.00, 999.00, 1000.00, 10000.00
        ]
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                tickers = await exchange.fetch_tickers()
                
                for symbol, ticker in list(tickers.items())[:30]:  # Top 30
                    if not is_sane_ticker(symbol):
                        continue
                    
                    current_price = ticker.get('last', 0)
                    if current_price <= 0:
                        continue
                    
                    # Find nearest psychological level
                    nearest_level = min(psychological_levels, key=lambda x: abs(x - current_price))
                    distance_pct = abs(current_price - nearest_level) / nearest_level
                    
                    if distance_pct < 0.02:  # Within 2% of psychological level
                        try:
                            # Get detailed history around this level
                            ohlcv = await exchange.fetch_ohlcv(symbol, '5m', limit=500)
                            if not ohlcv:
                                continue
                                
                            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            
                            # Find all times price was near this level
                            near_level = df[
                                (df['high'] >= nearest_level * 0.98) & 
                                (df['low'] <= nearest_level * 1.02)
                            ].copy()
                            
                            if len(near_level) >= 10:  # At least 10 touches
                                # Analyze behavior around level
                                results = []
                                
                                for idx in near_level.index:
                                    if idx + 6 < len(df):  # Need 30 minutes after
                                        touch_price = df['close'].iloc[idx]
                                        future_price = df['close'].iloc[idx + 6]
                                        
                                        if touch_price < nearest_level:  # Approached from below
                                            if future_price > nearest_level * 1.01:
                                                results.append('breakthrough')
                                            else:
                                                results.append('rejection')
                                        else:  # Approached from above
                                            if future_price < nearest_level * 0.99:
                                                results.append('breakdown')
                                            else:
                                                results.append('support')
                                
                                if len(results) >= 10:
                                    # Calculate probabilities
                                    rejection_rate = results.count('rejection') / len([r for r in results if r in ['breakthrough', 'rejection']]) if 'rejection' in results or 'breakthrough' in results else 0
                                    support_rate = results.count('support') / len([r for r in results if r in ['breakdown', 'support']]) if 'support' in results or 'breakdown' in results else 0

                                    pattern = {
                                        'level': float(nearest_level),
                                        'current_price': float(current_price),
                                        'rejection_rate': float(rejection_rate),
                                        'support_rate': float(support_rate),
                                        'touches': len(near_level),
                                        'behavior': 'resistance' if rejection_rate > support_rate else 'support',
                                        'description': f"{symbol} shows {max(rejection_rate, support_rate):.1%} {('rejection' if rejection_rate > support_rate else 'support')} at ${nearest_level}",
                                        'confidence': float(max(rejection_rate, support_rate) * min(1.0, len(results) / 20)),
                                        'distance_pct': float(distance_pct),
                                        'avg_bounce': 0.02,  # Default 2% bounce
                                        'win_rate': float(max(rejection_rate, support_rate)),
                                        'sample_size': len(results),
                                        'avg_return': 0.02 if rejection_rate > support_rate else -0.02
                                    }

                                    await self.learning_memory.save_pattern(pattern)
                                    log.info(f"💡 Discovered psychological pattern: {pattern['description']}")
                        except Exception as e:
                            continue
                            
            except Exception as e:
                log.error(f"Error discovering psychological patterns: {e}")

    async def _discover_correlation_patterns(self):
        """Discover correlations between different assets (BTC dominance effects, etc)"""
        log.debug("🔗 Discovering correlation patterns...")
        
        # Focus on major pairs - these often move together or opposite
        correlation_pairs = [
            ('BTC/USDT', 'ETH/USDT'),
            ('BTC/USDT', 'BNB/USDT'),
            ('ETH/USDT', 'MATIC/USDT'),
            ('BTC/USDT', 'SOL/USDT'),
            ('BTC/USDT', 'DOGE/USDT')
        ]
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                for symbol1, symbol2 in correlation_pairs:
                    try:
                        # Get data for both symbols
                        ohlcv1 = await exchange.fetch_ohlcv(symbol1, '15m', limit=500)
                        ohlcv2 = await exchange.fetch_ohlcv(symbol2, '15m', limit=500)
                        
                        if not ohlcv1 or not ohlcv2 or len(ohlcv1) != len(ohlcv2):
                            continue
                        
                        df1 = pd.DataFrame(ohlcv1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df2 = pd.DataFrame(ohlcv2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        
                        # Calculate returns
                        df1['returns'] = df1['close'].pct_change()
                        df2['returns'] = df2['close'].pct_change()
                        
                        # Rolling correlation
                        window = 50
                        correlations = []
                        
                        for i in range(window, len(df1)):
                            corr = df1['returns'].iloc[i-window:i].corr(df2['returns'].iloc[i-window:i])
                            correlations.append(corr)
                        
                        correlations = np.array(correlations)
                        avg_correlation = np.mean(correlations)
                        
                        # Look for regime changes (when correlation breaks down)
                        if abs(avg_correlation) > 0.5:  # Significant correlation
                            # Find periods of decorrelation
                            decorrelation_threshold = 0.2
                            decorrelated = correlations < decorrelation_threshold
                            
                            # Analyze what happens after decorrelation
                            opportunities = []
                            
                            for i in range(1, len(decorrelated)-10):
                                if decorrelated[i] and not decorrelated[i-1]:  # Just decorrelated
                                    # Check subsequent performance
                                    idx = i + window
                                    if idx + 10 < len(df1):
                                        future_ret1 = (df1['close'].iloc[idx+10] - df1['close'].iloc[idx]) / df1['close'].iloc[idx]
                                        future_ret2 = (df2['close'].iloc[idx+10] - df2['close'].iloc[idx]) / df2['close'].iloc[idx]
                                        
                                        spread_change = future_ret2 - future_ret1
                                        opportunities.append(spread_change)
                            
                            if len(opportunities) >= 5:
                                avg_opportunity = np.mean(opportunities)
                                win_rate = len([o for o in opportunities if o > 0]) / len(opportunities)
                                
                                if abs(avg_opportunity) > 0.02 and win_rate > 0.65:
                                    pattern = {
                                        'type': 'correlation_breakdown',
                                        'symbol1': symbol1,
                                        'symbol2': symbol2,
                                        'exchange': exchange_name,
                                        'avg_correlation': float(avg_correlation),
                                        'decorrelation_threshold': float(decorrelation_threshold),
                                        'avg_spread_change': float(avg_opportunity),
                                        'win_rate': float(win_rate),
                                        'opportunities': len(opportunities),
                                        'description': f"{symbol2}/{symbol1} spread moves {avg_opportunity:+.1%} after correlation breakdown (normally {avg_correlation:.1f} corr)",
                                        'confidence': float(win_rate * min(1.0, len(opportunities) / 10)),
                                        'trade_direction': 'long_spread' if avg_opportunity > 0 else 'short_spread',
                                        'sample_size': len(opportunities),
                                        'avg_return': float(avg_opportunity),
                                        'symbol': f"{symbol1}_{symbol2}"  # Combined symbol for tracking
                                    }
                                    
                                    await self.learning_memory.save_pattern(pattern)
                                    log.info(f"💡 Discovered correlation pattern: {pattern['description']}")
                                    
                    except Exception as e:
                        continue
                        
            except Exception as e:
                log.error(f"Error discovering correlation patterns: {e}")

    async def _discover_volatility_patterns(self):
        """Discover patterns related to volatility changes and clustering"""
        log.debug("📈 Discovering volatility patterns...")
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                tickers = await exchange.fetch_tickers()
                volatile_symbols = sorted(
                    [(s, t) for s, t in tickers.items() if is_sane_ticker(s) and t.get('percentage') is not None and abs(t.get('percentage', 0)) > 5],
                    key=lambda x: abs(x[1].get('percentage', 0)),
                    reverse=True
                )[:20]  # Top 20 movers
                
                for symbol, _ in volatile_symbols:
                    try:
                        ohlcv = await exchange.fetch_ohlcv(symbol, '5m', limit=500)
                        if not ohlcv:
                            continue
                            
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        
                        # Calculate various volatility metrics
                        df['returns'] = df['close'].pct_change()
                        df['abs_returns'] = abs(df['returns'])
                        df['range_pct'] = (df['high'] - df['low']) / df['close']
                        
                        # Historical volatility
                        df['hist_vol'] = df['returns'].rolling(window=20).std()
                        df['vol_percentile'] = df['hist_vol'].rolling(window=100).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))
                        
                        # GARCH-like patterns: volatility clustering
                        high_vol_periods = df[df['vol_percentile'] > 80].copy()
                        
                        if len(high_vol_periods) >= 10:
                            # Analyze persistence
                            persistence_results = []
                            
                            for idx in high_vol_periods.index:
                                if idx + 10 < len(df):
                                    # Check if volatility remains high
                                    future_vol = df['hist_vol'].iloc[idx:idx+10].mean()
                                    current_vol = df['hist_vol'].iloc[idx]
                                    persistence = future_vol / current_vol if current_vol > 0 else 0
                                    persistence_results.append(persistence)
                            
                            if len(persistence_results) >= 5:
                                avg_persistence = np.mean(persistence_results)
                                
                                if avg_persistence > 0.7:  # Volatility persists
                                    # Check profitability of volatility strategies
                                    straddle_results = []
                                    
                                    for idx in high_vol_periods.index:
                                        if idx + 6 < len(df):  # 30 minutes
                                            entry_price = df['close'].iloc[idx]
                                            max_move = max(
                                                abs(df['high'].iloc[idx:idx+6].max() - entry_price),
                                                abs(entry_price - df['low'].iloc[idx:idx+6].min())
                                            )
                                            straddle_pnl = max_move / entry_price - 0.002  # Assume 0.2% cost
                                            straddle_results.append(straddle_pnl)
                                    
                                    if len(straddle_results) >= 5:
                                        avg_pnl = np.mean(straddle_results)
                                        win_rate = len([r for r in straddle_results if r > 0]) / len(straddle_results)
                                        
                                        if avg_pnl > 0.005 and win_rate > 0.6:  # 0.5% avg profit, 60% win
                                            pattern = {
                                                'type': 'volatility_persistence',
                                                'symbol': symbol,
                                                'exchange': exchange_name,
                                                'avg_persistence': float(avg_persistence),
                                                'avg_straddle_pnl': float(avg_pnl),
                                                'win_rate': float(win_rate),
                                                'sample_size': len(straddle_results),
                                                'description': f"{symbol} volatility persists {avg_persistence:.1f}x with {avg_pnl:.1%} avg straddle profit",
                                                'confidence': float(win_rate * min(1.0, len(straddle_results) / 10)),
                                                'entry_condition': 'vol_percentile > 80',
                                                'hold_periods': 6,  # 30 minutes
                                                'avg_return': float(avg_pnl)
                                            }
                                            
                                            await self.learning_memory.save_pattern(pattern)
                                            log.info(f"💡 Discovered volatility pattern: {pattern['description']}")
                                            
                    except Exception as e:
                        continue
                        
            except Exception as e:
                log.error(f"Error discovering volatility patterns: {e}")

    async def _discover_microstructure_patterns(self):
        """Discover patterns in order book dynamics and market microstructure"""
        log.debug("🔬 Discovering microstructure patterns...")
        
        for exchange_name, exchange in self.exchanges.items():
            if not exchange.has['fetchOrderBook']:
                continue
                
            try:
                # Focus on liquid pairs
                tickers = await exchange.fetch_tickers()
                liquid_symbols = sorted(
                    [(s, t) for s, t in tickers.items() if is_sane_ticker(s) and t.get('quoteVolume') is not None and t.get('quoteVolume', 0) > 100000],
                    key=lambda x: x[1].get('quoteVolume', 0),
                    reverse=True
                )[:10]  # Top 10 most liquid
                
                for symbol, ticker in liquid_symbols:
                    try:
                        # Get order book snapshots over time
                        spread_data = []
                        imbalance_data = []
                        
                        for _ in range(20):  # 20 snapshots
                            order_book = await exchange.fetch_order_book(symbol, limit=20)
                            
                            if order_book['bids'] and order_book['asks']:
                                # Calculate metrics
                                best_bid = order_book['bids'][0][0]
                                best_ask = order_book['asks'][0][0]
                                mid_price = (best_bid + best_ask) / 2
                                spread_bps = (best_ask - best_bid) / mid_price * 10000  # basis points
                                
                                # Order book imbalance
                                bid_volume = sum([bid[1] for bid in order_book['bids'][:5]])
                                ask_volume = sum([ask[1] for ask in order_book['asks'][:5]])
                                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
                                
                                spread_data.append(spread_bps)
                                imbalance_data.append(imbalance)
                                
                            await asyncio.sleep(1)  # Wait between snapshots
                        
                        if len(spread_data) >= 10:
                            avg_spread = np.mean(spread_data)
                            avg_imbalance = np.mean(imbalance_data)
                            
                            # Check for consistently wide spreads (market making opportunity)
                            if avg_spread > 10:  # More than 10 bps
                                pattern = {
                                    'type': 'wide_spread',
                                    'symbol': symbol,
                                    'exchange': exchange_name,
                                    'avg_spread_bps': float(avg_spread),
                                    'spread_std': float(np.std(spread_data)),
                                    'description': f"{symbol} on {exchange_name} has {avg_spread:.1f} bps spread - potential market making opportunity",
                                    'confidence': 0.8 if avg_spread > 20 else 0.6,
                                    'min_spread': float(min(spread_data)),
                                    'max_spread': float(max(spread_data)),
                                    'win_rate': 0.7,  # Estimated for market making
                                    'avg_return': float(avg_spread / 10000),  # Convert bps to return
                                    'sample_size': len(spread_data)
                                }
                                
                                await self.learning_memory.save_pattern(pattern)
                                log.info(f"💡 Discovered microstructure pattern: {pattern['description']}")
                            
                            # Check for persistent order book imbalance
                            if abs(avg_imbalance) > 0.3:  # 30% imbalance
                                direction = 'buy' if avg_imbalance > 0 else 'sell'
                                pattern = {
                                    'type': 'order_book_imbalance',
                                    'symbol': symbol,
                                    'exchange': exchange_name,
                                    'avg_imbalance': float(avg_imbalance),
                                    'imbalance_persistence': float(len([i for i in imbalance_data if i * avg_imbalance > 0]) / len(imbalance_data)),
                                    'pressure_direction': direction,
                                    'description': f"{symbol} shows persistent {direction} pressure with {abs(avg_imbalance):.1%} order book imbalance",
                                    'confidence': float(min(0.9, abs(avg_imbalance) * 2)),
                                    'sample_size': len(imbalance_data),
                                    'win_rate': 0.65,  # Estimated
                                    'avg_return': 0.01 if direction == 'buy' else -0.01
                                }
                                
                                await self.learning_memory.save_pattern(pattern)
                                log.info(f"💡 Discovered order book pattern: {pattern['description']}")
                                
                    except Exception as e:
                        continue
                        
            except Exception as e:
                log.error(f"Error discovering microstructure patterns: {e}")

    async def _find_mean_reversion(self, df: pd.DataFrame, symbol: str, exchange: str):
        """Find mean reversion opportunities when price deviates from moving averages"""
        
        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(window=20).mean()
        df['std'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma'] + (df['std'] * 2)
        df['lower_band'] = df['sma'] - (df['std'] * 2)        # Calculate z-score for extreme deviations
        df['z_score'] = (df['close'] - df['sma']) / df['std']
        
        # Track reversions
        reversion_results = []
        
        for i in range(20, len(df) - 10):
            if abs(df['z_score'].iloc[i]) > 2:  # 2+ standard deviations
                entry_price = df['close'].iloc[i]
                entry_z = df['z_score'].iloc[i]
                
                # Check if price reverted to mean within 10 periods
                for j in range(i+1, min(i+11, len(df))):
                    if entry_z > 0 and df['close'].iloc[j] < df['sma'].iloc[j]:  # Reverted from above
                        exit_price = df['close'].iloc[j]
                        profit = (entry_price - exit_price) / entry_price  # Short profit
                        reversion_results.append({
                            'entry_z': entry_z,
                            'periods_to_revert': j - i,
                            'profit': profit,
                            'type': 'short'
                        })
                        break
                    elif entry_z < 0 and df['close'].iloc[j] > df['sma'].iloc[j]:  # Reverted from below
                        exit_price = df['close'].iloc[j]
                        profit = (exit_price - entry_price) / entry_price  # Long profit
                        reversion_results.append({
                            'entry_z': entry_z,
                            'periods_to_revert': j - i,
                            'profit': profit,
                            'type': 'long'
                        })
                        break
        
        if len(reversion_results) >= 10:
            avg_profit = np.mean([r['profit'] for r in reversion_results])
            win_rate = len([r for r in reversion_results if r['profit'] > 0]) / len(reversion_results)
            avg_periods = np.mean([r['periods_to_revert'] for r in reversion_results])
            
            if avg_profit > 0.005 and win_rate > 0.65:  # 0.5% avg profit, 65% win rate
                pattern = {
                    'type': 'mean_reversion',
                    'symbol': symbol,
                    'exchange': exchange,
                    'entry_z_score': 2.0,
                    'avg_profit': float(avg_profit),
                    'win_rate': float(win_rate),
                    'avg_periods_to_revert': float(avg_periods),
                    'sample_size': len(reversion_results),
                    'description': f"{symbol} shows {avg_profit:.1%} avg profit from 2-sigma mean reversion ({win_rate:.1%} win rate)",
                    'confidence': float(win_rate * min(1.0, len(reversion_results) / 20)),
                    'sma_period': 20,
                    'entry_threshold': 2.0,
                    'avg_return': float(avg_profit)
                }
                
                await self.learning_memory.save_pattern(pattern)
                log.info(f"💡 Discovered mean reversion pattern: {pattern['description']}")

    async def _cleanup_low_confidence_patterns(self):
        """Remove patterns that have proven unreliable over time"""
        
        # Get all patterns
        all_patterns = await self.learning_memory.get_all_patterns()
        
        for pattern_dict in all_patterns:
            # Skip if recently discovered
            if 'discovered_at' in pattern_dict:
                discovered = datetime.fromisoformat(pattern_dict['discovered_at']) if isinstance(pattern_dict['discovered_at'], str) else pattern_dict['discovered_at']
                if (datetime.utcnow() - discovered).days < 7:
                    continue
            
            # Check if pattern has been used in strategies
            if pattern_dict.get('usage_count', 0) > 10:
                # Pattern has been tested enough
                success_rate = pattern_dict.get('success_count', 0) / pattern_dict['usage_count']
                
                if success_rate < 0.4:  # Poor performance
                    # Reduce confidence significantly
                    if pattern_dict['id'] in self.learning_memory.patterns:
                        pattern = self.learning_memory.patterns[pattern_dict['id']]
                        pattern.confidence *= 0.5
                        
                        if pattern.confidence < 0.3:
                            await self.learning_memory.delete_pattern(pattern.id)
                            log.info(f"🗑️ Removed low-confidence pattern: {pattern.description}")
                        else:
                            await self.learning_memory.update_pattern(pattern)
                            log.debug(f"📉 Reduced confidence for pattern: {pattern.description}")

# ═══════════════════════════ ADAPTIVE STRATEGY ENGINE ═════════════════════════

class AdaptiveStrategyEngine:
    """
    Generates, tests, and evolves trading strategies based on discovered patterns.
    Uses genetic algorithm concepts (generation, mutation, crossover) to
    continuously improve the portfolio of active strategies.
    """
    def __init__(self, learning_memory: LearningMemory, ai_manager: OpenAIManager, db: Database):
        self.learning_memory = learning_memory
        self.ai_manager = ai_manager
        self.db = db
        self.notifier = None  # Will be set by AutonomousTrader
        log.info("🧬 Adaptive Strategy Engine initialized.")

    def _determine_growth_phase(self, state: SystemState) -> str:
        """Determines the current growth phase based on equity."""
        equity = state.equity
        initial = Config.INITIAL_CAPITAL
        
        if equity < initial * 5:  # Up to $1K
            return "Discovery"
        elif equity < initial * 50:  # Up to $10K
            return "Optimization"
        elif equity < initial * 500:  # Up to $100K
            return "Scaling"
        else:  # Above $100K
            return "Domination"

    async def evolve_strategies(self, state: SystemState):
        """
        Enhanced evolution with mutation and crossover for 90-day sprint.
        """
        log.info("🧬 Starting aggressive strategy evolution...")
        
        # Generate new strategies from patterns
        await self._generate_new_strategies_from_patterns(state)
        
        # Mutate top performers
        await self._mutate_top_strategies(state)
        
        # Crossover successful strategies
        await self._crossover_strategies(state)
        
        # Evaluate and promote
        await self._evaluate_and_promote_strategies(state)
        
        # Retire underperformers
        await self._retire_underperforming_strategies(state)
        
        # Report evolution stats
        active_count = len([s for s in state.strategies.values() if s.status in [StrategyStatus.ACTIVE, StrategyStatus.MICRO]])
        paper_count = len([s for s in state.strategies.values() if s.status == StrategyStatus.PAPER])
        retired_count = len([s for s in state.strategies.values() if s.status == StrategyStatus.RETIRED])
        
        log.info(f"📊 Evolution complete: {active_count} active, {paper_count} testing, {retired_count} retired")
        
        if self.notifier:
            await self.notifier.send_alert(f"🧬 Strategy evolution: {active_count} active, {paper_count} testing")

    async def _generate_new_strategies_from_patterns(self, state: SystemState):
        """
        Uses high-potential patterns from memory to generate new strategies via the AI.
        """
        log.debug("🌱 Generating new strategies from patterns...")
        
        patterns = await self.learning_memory.get_high_confidence_patterns(limit=5)
        phase = self._determine_growth_phase(state)
        
        for pattern in patterns:
            # Skip if we already have a strategy for this exact pattern
            strategy_name = f"Strat_{pattern['type']}_{pattern.get('symbol', 'multi')}_{int(time.time())}"
            
            # Check for existing similar strategies
            existing_similar = any(
                s.pattern_id == pattern.get('id') and s.status != StrategyStatus.RETIRED 
                for s in state.strategies.values()
            )
            
            if existing_similar:
                log.debug(f"Skipping pattern {pattern['id']}: similar strategy exists")
                continue
            
            # Create enhanced prompt with specific pattern details
            prompt = self._create_prompt_from_pattern(pattern, state)
            
            try:
                generated_code = await self.ai_manager.generate_strategy(prompt)
                
                # Validate generated code
                if "async def execute_strategy" not in generated_code:
                    log.warning(f"Invalid code generated for pattern {pattern.get('id', 'unknown')}")
                    continue
                
                # Create new strategy
                new_strategy = Strategy(
                    id=hashlib.sha256(strategy_name.encode()).hexdigest()[:16],
                    name=strategy_name,
                    description=f"Auto-generated from: {pattern['description']}",
                    code=generated_code,
                    status=StrategyStatus.PAPER,  # Always start in paper mode
                    pattern_id=pattern.get('id'),
                    generation=len([s for s in state.strategies.values() if s.pattern_id == pattern.get('id')]) + 1
                )
                
                # Adjust initial parameters based on phase
                if phase == "Discovery":
                    new_strategy.max_position_pct = 0.02  # 2% max
                elif phase == "Optimization":
                    new_strategy.max_position_pct = 0.05  # 5% max
                else:
                    new_strategy.max_position_pct = 0.10  # 10% max
                
                state.strategies[new_strategy.id] = new_strategy
                await self.db.save_strategy(new_strategy)
                
                log.info(f"🌱 Generated new strategy: {new_strategy.name} (Gen {new_strategy.generation})")
                
            except Exception as e:
                log.error(f"Error generating strategy from pattern {pattern.get('id', 'unknown')}: {e}")

    def _create_prompt_from_pattern(self, pattern: dict, state: SystemState) -> str:
        """Creates an enhanced, detailed prompt for the AI to generate a Python strategy."""
        
        phase = self._determine_growth_phase(state)
        
        # Build pattern-specific implementation details
        pattern_details = []
        entry_logic = []
        exit_logic = []
        risk_params = {}
        
        if pattern['type'] == 'psychological_level':
            level = pattern['level']
            pattern_details.append(f"Key level: ${level:.4f}")
            pattern_details.append(f"Behavior: {pattern.get('behavior', 'resistance')}")
            pattern_details.append(f"Historical rejection rate: {pattern.get('rejection_rate', 0.7):.1%}")
            
            entry_logic.append(f"abs(opp['price'] - {level}) / {level} < 0.005  # Within 0.5% of level")
            entry_logic.append(f"# Entry when price approaches ${level}")
            
            if pattern.get('behavior') == 'resistance':
                entry_logic.append(f"opp['price'] < {level} * 0.995  # Below resistance")
                exit_logic.append(f"price >= {level} * 0.98  # Bounce from resistance")
            else:
                entry_logic.append(f"opp['price'] > {level} * 1.005  # Above support")
                exit_logic.append(f"price <= {level} * 1.02  # Bounce from support")
            
            risk_params['stop_loss'] = 0.01  # 1% stop
            risk_params['take_profit'] = pattern.get('avg_bounce', 0.02)
            
        elif pattern['type'] == 'volume_spike_momentum':
            pattern_details.append(f"Volume Z-score threshold: {pattern['avg_zscore']:.1f}")
            pattern_details.append(f"Expected move: {pattern['avg_subsequent_return']:.1%}")
            pattern_details.append(f"Direction: {pattern['direction']}")
            pattern_details.append(f"Optimal hold: {pattern.get('optimal_hold_periods', 4)} periods")
            
            entry_logic.append(f"# Detect volume spike (would need volume data in real implementation)")
            entry_logic.append(f"volume_spike = True  # Placeholder - check if volume > 3 sigma")
            entry_logic.append(f"momentum_direction = '{pattern['direction']}'")
            
            risk_params['stop_loss'] = 0.02
            risk_params['take_profit'] = abs(pattern['avg_subsequent_return'])
            
        elif pattern['type'] == 'time_of_day':
            hour = pattern['hour_utc']
            pattern_details.append(f"Entry hour: {hour}:00 UTC")
            pattern_details.append(f"Historical return: {pattern['avg_return']:.2%}")
            pattern_details.append(f"Sharpe ratio: {pattern.get('sharpe_ratio', 1.0):.2f}")
            
            entry_logic.append(f"from datetime import datetime")
            entry_logic.append(f"current_hour = datetime.utcnow().hour")
            entry_logic.append(f"time_match = current_hour == {hour}")
            
            risk_params['stop_loss'] = 0.015
            risk_params['take_profit'] = pattern['avg_return'] * 2
        elif pattern['type'] == 'mean_reversion':
            pattern_details.append(f"Entry Z-score: {pattern.get('entry_z_score', 2.0):.1f}")
            pattern_details.append(f"Average profit: {pattern['avg_profit']:.2%}")
            pattern_details.append(f"Periods to revert: {pattern.get('avg_periods_to_revert', 5):.0f}")
            
            entry_logic.append(f"# Calculate Bollinger Bands z-score")
            entry_logic.append(f"z_score_threshold = {pattern.get('entry_z_score', 2.0)}")
            entry_logic.append(f"# Entry when price deviates {pattern.get('entry_z_score', 2.0)} sigma from mean")
            
            risk_params['stop_loss'] = 0.03
            risk_params['take_profit'] = pattern['avg_profit']        # Build the complete prompt
        prompt = f"""
Generate a complete async Python trading strategy function based on this discovered pattern:

**IMPORTANT RULES**:
- DO NOT use any import statements
- numpy is available as 'np'
- pandas is available as 'pd' 
- datetime, math, and random are available
- The function MUST be named 'execute_strategy'
- It MUST be async (use 'async def')
- It MUST return a dict with 'action' and 'conf' keys

**Pattern Type**: {pattern['type']}
**Symbol**: {pattern.get('symbol', 'Multiple')}
**Description**: {pattern['description']}
**Win Rate**: {pattern['win_rate']:.1%}
**Average Return**: {pattern['avg_return']:.2%}
**Confidence**: {pattern['confidence']:.1%}
**Sample Size**: {pattern['sample_size']} observations

**Pattern Details**:
{chr(10).join(f"• {detail}" for detail in pattern_details)}

**Current Growth Phase**: {phase}
- Risk tolerance: {"Low (2% max)" if phase == "Discovery" else "Medium (5% max)" if phase == "Optimization" else "High (10% max)"}
- Focus: {"Pattern validation" if phase == "Discovery" else "Optimizing winners" if phase == "Optimization" else "Maximizing returns"}

Generate a complete async function with this signature:
```python
async def execute_strategy(state, opp):
    '''
    Strategy: {pattern['type']} pattern for {pattern.get('symbol', 'multiple symbols')}
    
    Args:
        state: SystemState object with equity, positions, etc.
        opp: Dict with 'symbol', 'price', 'volume', 'exchange', etc.
    
    Returns:
        Dict with 'action' ('buy'/'sell'/'hold'), 'conf' (0.0-1.0), and optional 'reason'
    '''
```

**Entry Conditions to implement**:
{chr(10).join(f"- {condition}" for condition in entry_logic)}

**Risk Parameters**:
- Stop Loss: {risk_params.get('stop_loss', 0.05):.1%}
- Take Profit: {risk_params.get('take_profit', 0.15):.1%}

**Requirements**:
1. Check if pattern conditions are met
2. Calculate position size based on Kelly Criterion (capped at {Config.KELLY_FRACTION})
3. Verify risk limits (max position size, daily loss limits)
4. Return confidence score based on pattern strength
5. Include detailed reasoning in return dict

Make the strategy aggressive but intelligent - we need to reach $1M in 90 days from $200.
"""
        
        return prompt

    async def _mutate_top_strategies(self, state: SystemState):
        """
        Mutate successful strategies to create variations.
        More aggressive mutations in Discovery phase, fine-tuning in later phases.
        """
        phase = self._determine_growth_phase(state)
        
        # Determine mutation rate based on phase
        if phase == "Discovery":
            mutation_rate = 0.20  # 20% parameter change
            top_count = 3
        elif phase == "Optimization":
            mutation_rate = 0.10  # 10% parameter change
            top_count = 5
        elif phase == "Scaling":
            mutation_rate = 0.05  # 5% parameter change
            top_count = 7
        else:  # Domination
            mutation_rate = 0.02  # 2% fine-tuning
            top_count = 10
        
        # Get top performing strategies
        active_strategies = [s for s in state.strategies.values() 
                           if s.status in [StrategyStatus.ACTIVE, StrategyStatus.MICRO] 
                           and s.total_trades >= 10]
        
        if not active_strategies:
            return
        
        # Sort by performance score (combination of win rate and Sharpe ratio)
        top_strategies = sorted(
            active_strategies,
            key=lambda s: s.win_rate * (1 + s.sharpe_ratio) * (1 + s.total_pnl/100),
            reverse=True
        )[:top_count]
        
        for parent_strategy in top_strategies:
            # Extract numeric parameters from code
            params = extract_params(parent_strategy.code)
            
            if not params:
                continue
            
            # Mutate parameters
            mutated_params = {}
            for param_name, param_value in params.items():
                # Random mutation within range
                mutation = random.uniform(-mutation_rate, mutation_rate)
                mutated_value = param_value * (1 + mutation)
                
                # Keep parameters in reasonable ranges
                if 'stop' in param_name.lower():
                    mutated_value = max(0.005, min(0.1, mutated_value))  # 0.5% to 10%
                elif 'profit' in param_name.lower() or 'target' in param_name.lower():
                    mutated_value = max(0.01, min(0.5, mutated_value))  # 1% to 50%
                elif 'threshold' in param_name.lower():
                    mutated_value = max(0.1, min(5.0, mutated_value))  # Keep thresholds reasonable
                
                mutated_params[param_name] = mutated_value
            
            # Create mutated strategy
            mutated_code = inject_params(parent_strategy.code, mutated_params)
            
            mutated_strategy = Strategy(
                id=hashlib.sha256(f"{parent_strategy.id}_mut_{time.time()}".encode()).hexdigest()[:16],
                name=f"{parent_strategy.name}_mut{parent_strategy.generation + 1}",
                description=f"Mutation of {parent_strategy.name} ({mutation_rate:.0%} change)",
                code=mutated_code,
                status=StrategyStatus.PAPER,  # Test mutations first
                pattern_id=parent_strategy.pattern_id,
                generation=parent_strategy.generation + 1,
                parent_id=parent_strategy.id
            )
            
            # Inherit some characteristics from parent
            mutated_strategy.max_position_pct = parent_strategy.max_position_pct * (1 + random.uniform(-0.2, 0.2))
            mutated_strategy.stop_loss = parent_strategy.stop_loss * (1 + random.uniform(-0.1, 0.1))
            mutated_strategy.take_profit = parent_strategy.take_profit * (1 + random.uniform(-0.1, 0.1))
            
            state.strategies[mutated_strategy.id] = mutated_strategy
            await self.db.save_strategy(mutated_strategy)
            
            log.info(f"🧬 Created mutation: {mutated_strategy.name} from {parent_strategy.name}")

    async def _crossover_strategies(self, state: SystemState):
        """
        Combine elements from two successful strategies to create offspring.
        This implements genetic crossover for strategy evolution.
        """
        # Get pairs of successful strategies
        successful_strategies = [
            s for s in state.strategies.values()
            if s.status in [StrategyStatus.ACTIVE, StrategyStatus.MICRO]
            and s.win_rate > 0.55
            and s.total_trades >= 20
        ]
        
        if len(successful_strategies) < 2:
            return
        
        # Sort by performance
        successful_strategies.sort(
            key=lambda s: s.win_rate * s.sharpe_ratio,
            reverse=True
        )
        
        # Create crossover pairs
        num_crossovers = min(3, len(successful_strategies) // 2)
        
        for i in range(num_crossovers):
            parent1 = successful_strategies[i * 2]
            parent2 = successful_strategies[i * 2 + 1]
            
            # Extract parameters from both parents
            params1 = extract_params(parent1.code)
            params2 = extract_params(parent2.code)
            
            # Crossover: randomly select parameters from each parent
            offspring_params = {}
            all_param_names = set(params1.keys()) | set(params2.keys())
            
            for param_name in all_param_names:
                if param_name in params1 and param_name in params2:
                    # Both have this parameter - randomly choose or average
                    if random.random() < 0.5:
                        offspring_params[param_name] = params1[param_name]
                    else:
                        offspring_params[param_name] = params2[param_name]
                elif param_name in params1:
                    offspring_params[param_name] = params1[param_name]
                else:
                    offspring_params[param_name] = params2[param_name]
            
            # Use the better performing parent's code as base
            base_code = parent1.code if parent1.sharpe_ratio > parent2.sharpe_ratio else parent2.code
            offspring_code = inject_params(base_code, offspring_params)
            
            # Create offspring strategy
            offspring_strategy = Strategy(
                id=hashlib.sha256(f"cross_{parent1.id}_{parent2.id}_{time.time()}".encode()).hexdigest()[:16],
                name=f"Cross_{parent1.name[:10]}_{parent2.name[:10]}",
                description=f"Crossover of {parent1.name} × {parent2.name}",
                code=offspring_code,
                status=StrategyStatus.PAPER,
                generation=max(parent1.generation, parent2.generation) + 1,
                parent_id=f"{parent1.id},{parent2.id}"  # Both parents
            )
            
            # Inherit averaged characteristics
            offspring_strategy.max_position_pct = (parent1.max_position_pct + parent2.max_position_pct) / 2
            offspring_strategy.stop_loss = (parent1.stop_loss + parent2.stop_loss) / 2
            offspring_strategy.take_profit = (parent1.take_profit + parent2.take_profit) / 2
            
            state.strategies[offspring_strategy.id] = offspring_strategy
            await self.db.save_strategy(offspring_strategy)
            
            log.info(f"🧬 Created crossover: {offspring_strategy.name}")

    async def _evaluate_and_promote_strategies(self, state: SystemState):
        """
        Evaluate paper trading strategies and promote successful ones.
        Uses Wilson score for more accurate win rate with small samples.
        """
        phase = self._determine_growth_phase(state)
        
        for strategy in state.strategies.values():
            if strategy.status == StrategyStatus.PAPER:
                # Need minimum trades before evaluation
                min_trades = 50 if phase == "Discovery" else 30
                
                if strategy.total_trades >= min_trades:
                    # Calculate Wilson score for win rate (better for small samples)
                    wins = strategy.winning_trades
                    total = strategy.total_trades
                    z = 1.96  # 95% confidence
                    
                    p_hat = wins / total if total > 0 else 0
                    wilson_score = (p_hat + z*z/(2*total) - z * math.sqrt((p_hat*(1-p_hat) + z*z/(4*total))/total)) / (1 + z*z/total)
                    
                    # Promotion criteria varies by phase
                    if phase == "Discovery":
                        promote_threshold = 0.52  # Just need to be profitable
                        sharpe_threshold = 0.5
                    elif phase == "Optimization":
                        promote_threshold = 0.55
                        sharpe_threshold = 1.0
                    else:  # Scaling/Domination
                        promote_threshold = 0.60
                        sharpe_threshold = 1.5
                    
                    if wilson_score > promote_threshold and strategy.sharpe_ratio > sharpe_threshold:
                        # Promote to micro trading
                        strategy.status = StrategyStatus.MICRO
                        await self.db.save_strategy(strategy)
                        
                        log.info(f"⬆️ Promoted {strategy.name} to MICRO (Wilson: {wilson_score:.2%}, Sharpe: {strategy.sharpe_ratio:.2f})")
                        
                        if self.notifier:
                            await self.notifier.send_alert(
                                f"⬆️ Strategy promoted: {strategy.name}\n"
                                f"Win rate: {wilson_score:.1%} | Sharpe: {strategy.sharpe_ratio:.2f}"
                            )
            
            elif strategy.status == StrategyStatus.MICRO:
                # Evaluate micro strategies for full activation
                if strategy.total_trades >= 100:
                    if strategy.win_rate > 0.58 and strategy.sharpe_ratio > 1.2 and strategy.total_pnl > 0:
                        strategy.status = StrategyStatus.ACTIVE
                        
                        # Increase position size for proven strategies
                        if phase == "Scaling" or phase == "Domination":
                            strategy.position_multiplier = min(2.0, 1.0 + (strategy.sharpe_ratio - 1.0))
                        
                        await self.db.save_strategy(strategy)
                        
                        log.info(f"🚀 Activated {strategy.name} (Win: {strategy.win_rate:.1%}, Sharpe: {strategy.sharpe_ratio:.2f})")
                        
                        if self.notifier:
                            await self.notifier.send_alert(
                                f"🚀 Strategy ACTIVATED: {strategy.name}\n"
                                f"P&L: ${strategy.total_pnl:.2f} | Sharpe: {strategy.sharpe_ratio:.2f}"
                            )

    async def _retire_underperforming_strategies(self, state: SystemState):
        """
        Retire strategies that are consistently underperforming.
        More aggressive retirement in later phases to focus on winners.
        """
        phase = self._determine_growth_phase(state)
        
        for strategy in list(state.strategies.values()):
            should_retire = False
            reason = ""
            
            # Different retirement criteria by phase
            if phase == "Discovery":
                # Very lenient - only retire if terrible
                if strategy.total_trades >= 100 and strategy.win_rate < 0.40:
                    should_retire = True
                    reason = "Win rate < 40% after 100 trades"
                elif strategy.max_drawdown > 0.30:
                    should_retire = True
                    reason = "Drawdown > 30%"
                    
            elif phase == "Optimization":
                # Moderate - remove clear losers
                if strategy.total_trades >= 50 and strategy.win_rate < 0.45:
                    should_retire = True
                    reason = "Win rate < 45%"
                elif strategy.total_trades >= 50 and strategy.sharpe_ratio < 0:
                    should_retire = True
                    reason = "Negative Sharpe ratio"
                elif strategy.max_drawdown > 0.25:
                    should_retire = True
                    reason = "Drawdown > 25%"
                    
            else:  # Scaling/Domination
                # Aggressive - only keep the best
                if strategy.total_trades >= 30 and strategy.win_rate < 0.50:
                    should_retire = True
                    reason = "Win rate < 50%"
                elif strategy.total_trades >= 30 and strategy.sharpe_ratio < 0.8:
                    should_retire = True
                    reason = "Sharpe < 0.8"
                elif strategy.max_drawdown > 0.20:
                    should_retire = True
                    reason = "Drawdown > 20%"
                
                # Also retire if hasn't traded recently
                if strategy.last_trade:
                    days_since_trade = (datetime.utcnow() - strategy.last_trade).days
                    if days_since_trade > 3:
                        should_retire = True
                        reason = "Inactive for 3+ days"
            
            if should_retire:
                strategy.status = StrategyStatus.RETIRED
                await self.db.save_strategy(strategy)
                
                # Update pattern confidence if this strategy failed
                if strategy.pattern_id:
                    await self.learning_memory.record_pattern_usage(
                        strategy.pattern_id,
                        success=False,
                        return_pct=strategy.total_pnl / 100 if strategy.total_trades > 0 else 0
                    )
                
                log.info(f"🗑️ Retired {strategy.name}: {reason}")

# ═══════════════════════════ AUTONOMOUS TRADER ═════════════════════════════

class AutonomousTrader:
    """
    The main orchestrator that brings everything together.
    Manages the trading loop, executes strategies, and handles positions.
    """
    
    def __init__(self):
        self.state = SystemState()
        self.db = Database()
        self.learning_memory = None
        self.pattern_engine = None
        self.strategy_engine = None
        self.exchanges = {}
        self.notifier = None
        self.ai_manager = None
        self.running = False
        self.tasks = []
        
    async def initialize(self):
        """Initialize all components of the trading system."""
        log.info("🚀 Initializing v26meme Autonomous Trading System...")
        
        # Initialize AI manager
        if Config.OPENAI_API_KEY:
            self.ai_manager = OpenAIManager(Config.OPENAI_API_KEY)
        else:
            log.warning("⚠️ No OpenAI API key - strategy generation will be limited")
        
        # Initialize exchanges
        for exchange_name, config in Config.get_exchanges().items():
            if config.get('apiKey'):
                try:
                    exchange_class = getattr(ccxt, exchange_name)
                    self.exchanges[exchange_name] = exchange_class(config)
                    log.info(f"✅ Connected to {exchange_name}")
                except Exception as e:
                    log.error(f"Failed to connect to {exchange_name}: {e}")
        
        if not self.exchanges:
            log.error("❌ No exchanges configured!")
            return False
        
        # Initialize notification system
        if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
            self.notifier = TelegramNotifier(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)
            await self.notifier.send_alert("🚀 v26meme starting up...")
        
        # Initialize learning memory
        self.learning_memory = LearningMemory(self.db)
        await self.learning_memory.initialize()
        
        # Initialize pattern discovery engine
        self.pattern_engine = PatternDiscoveryEngine(
            self.learning_memory,
            self.exchanges,
            self.ai_manager
        )
        
        # Initialize strategy engine
        self.strategy_engine = AdaptiveStrategyEngine(
            self.learning_memory,
            self.ai_manager,
            self.db
        )
        self.strategy_engine.notifier = self.notifier
        
        # Load existing strategies
        strategies = await self.db.load_strategies()
        for strategy in strategies:
            self.state.strategies[strategy.id] = strategy
        
        log.info(f"📚 Loaded {len(self.state.strategies)} strategies")
        
        # Determine initial mode based on configuration
        if os.getenv('TRADING_MODE') == 'LIVE':
            self.state.mode = TradingMode.MICRO  # Start with small positions
            log.warning("⚠️ LIVE TRADING MODE - Real money at risk!")
        else:
            self.state.mode = TradingMode.PAPER
            log.info("📝 Paper trading mode - No real money at risk")
        
        return True
        
    async def run(self):
        """Main trading loop - the heart of the autonomous system."""
        log.info("💚 Starting main trading loop...")
        self.running = True
        
        # Start background tasks
        self.tasks = [
            asyncio.create_task(self._pattern_discovery_loop()),
            asyncio.create_task(self._strategy_evolution_loop()),
            asyncio.create_task(self._trading_loop()),
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._daily_report_loop())
        ]
        
        try:
            # Keep running until stopped
            await asyncio.gather(*self.tasks)
        except KeyboardInterrupt:
            log.info("⛔ Shutdown requested...")
            await self.shutdown()
        except Exception as e:
            log.error(f"Fatal error in main loop: {e}")
            await self.emergency_shutdown()

    async def _pattern_discovery_loop(self):
        """Continuously discover new patterns."""
        while self.running:
            try:
                await self.pattern_engine.discover_patterns(self.state)
                await asyncio.sleep(Config.PATTERN_DISCOVERY_INTERVAL)
            except Exception as e:
                log.error(f"Pattern discovery error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def _strategy_evolution_loop(self):
        """Continuously evolve and improve strategies."""
        await asyncio.sleep(300)  # Initial delay to gather some patterns
        
        while self.running:
            try:
                await self.strategy_engine.evolve_strategies(self.state)
                await asyncio.sleep(Config.STRATEGY_EVOLUTION_INTERVAL)
            except Exception as e:
                log.error(f"Strategy evolution error: {e}")
                await asyncio.sleep(60)

    async def _trading_loop(self):
        """Main trading execution loop."""
        log.info("📈 Trading loop started...")
        
        while self.running:
            try:
                # Get market opportunities
                opportunities = await self._scan_markets()
                
                # Execute strategies on opportunities
                for opp in opportunities:
                    await self._execute_strategies(opp)
                
                # Manage existing positions
                await self._manage_positions()
                
                # Update state
                await self._update_state()
                
                # Risk check
                if await self._check_risk_limits():
                    log.warning("⚠️ Risk limits hit - entering emergency mode")
                    self.state.mode = TradingMode.EMERGENCY
                
                await asyncio.sleep(5)  # Main loop frequency
                
            except Exception as e:
                log.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)

    async def _scan_markets(self) -> List[Dict]:
        """Scan all exchanges for trading opportunities."""
        opportunities = []
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Get current tickers
                tickers = await exchange.fetch_tickers()
                
                # Filter and rank opportunities
                for symbol, ticker in tickers.items():
                    if not is_sane_ticker(symbol):
                        continue
                    
                    # Calculate opportunity score based on volatility and volume
                    volume_usd = ticker.get('quoteVolume') if ticker.get('quoteVolume') is not None else 0
                    change_pct = abs(ticker.get('percentage')) if ticker.get('percentage') is not None else 0
                    
                    if volume_usd > 10000 and change_pct > 1:  # Active markets
                        opportunities.append({
                            'symbol': symbol,
                            'exchange': exchange_name,
                            'price': ticker.get('last', 0),
                            'volume': volume_usd,
                            'change_24h': ticker.get('percentage', 0),
                            'bid': ticker.get('bid', 0),
                            'ask': ticker.get('ask', 0),
                            'spread': (ticker.get('ask', 0) - ticker.get('bid', 0)) / ticker.get('last', 1) if ticker.get('last') else 0,
                            'timestamp': datetime.utcnow()
                        })
                
            except Exception as e:
                log.error(f"Error scanning {exchange_name}: {e}")
        
        # Sort by opportunity score (volume * volatility)
        opportunities.sort(
            key=lambda x: x['volume'] * abs(x['change_24h']),
            reverse=True
        )
        
        return opportunities[:100]  # Top 100 opportunities

    async def _execute_strategies(self, opp: Dict):
        """Execute all active strategies on a market opportunity."""
        
        # Only execute strategies in appropriate modes
        if self.state.mode == TradingMode.EMERGENCY:
            return
        
        # Get strategies to execute based on mode
        if self.state.mode == TradingMode.PAPER:
            strategies_to_run = [s for s in self.state.strategies.values() 
                               if s.status == StrategyStatus.PAPER]
        elif self.state.mode == TradingMode.MICRO:
            strategies_to_run = [s for s in self.state.strategies.values() 
                               if s.status in [StrategyStatus.PAPER, StrategyStatus.MICRO]]
        else:  # ACTIVE mode
            strategies_to_run = [s for s in self.state.strategies.values() 
                               if s.status != StrategyStatus.RETIRED]
        
        for strategy in strategies_to_run:
            try:
                # Execute strategy code dynamically
                decision = await self._run_strategy_code(strategy, opp)
                
                if decision and decision.get('action') != 'hold':
                    # Process the trading decision
                    await self._process_decision(strategy, opp, decision)
                    
            except Exception as e:
                log.error(f"Error executing strategy {strategy.name}: {e}")
                # Track strategy failures
                strategy.max_drawdown = max(strategy.max_drawdown, 0.01)  # Penalty for errors

    async def _run_strategy_code(self, strategy: Strategy, opp: Dict) -> Optional[Dict]:
        """Safely execute strategy code and return decision."""
        try:
            # Import libraries that strategies might need
            import numpy as np
            import pandas as pd
            
            # Create a restricted but functional execution environment
            exec_globals = {
                '__builtins__': {
                    'len': len,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'round': round,
                    'float': float,
                    'int': int,
                    'str': str,
                    'bool': bool,
                    'dict': dict,
                    'list': list,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'any': any,
                    'all': all
                },
                'np': np,  # Provide numpy
                'pd': pd,  # Provide pandas
                'datetime': datetime,
                'math': math,
                'random': random,
                'calculate_kelly_position': calculate_kelly_position
            }
            
            # Execute the strategy code
            exec(strategy.code, exec_globals)
            
            # Call the execute_strategy function
            if 'execute_strategy' in exec_globals:
                result = await exec_globals['execute_strategy'](self.state, opp)
                return result
            else:
                log.error(f"Strategy {strategy.name} missing execute_strategy function")
                return None
                
        except Exception as e:
            log.error(f"Strategy execution error in {strategy.name}: {e}")
            return None

    async def _process_decision(self, strategy: Strategy, opp: Dict, decision: Dict):
        """Process a trading decision from a strategy."""
        
        action = decision.get('action')
        confidence = decision.get('conf', 0.5)
        reason = decision.get('reason', 'No reason provided')
        
        # Confidence threshold based on strategy status
        if strategy.status == StrategyStatus.PAPER:
            min_confidence = 0.3  # Low threshold for testing
            position_size = 100  # Simulated $100
        elif strategy.status == StrategyStatus.MICRO:
            min_confidence = 0.5
            position_size = min(50, self.state.equity * 0.01)  # $50 or 1% of equity
        else:  # ACTIVE
            min_confidence = 0.6
            # Kelly-based position sizing
            if strategy.win_rate > 0 and strategy.avg_profit > 0:
                kelly_size = calculate_kelly_position(
                    strategy.win_rate,
                    strategy.avg_profit,
                    abs(strategy.avg_profit * (1 - strategy.win_rate) / strategy.win_rate) if strategy.win_rate > 0 else 0.05
                )
                position_size = self.state.equity * kelly_size * strategy.position_multiplier
            else:
                position_size = self.state.equity * 0.02  # Default 2%
        
        if confidence < min_confidence:
            return
        
        # Check if we already have a position in this symbol
        existing_position = None
        for pos in self.state.positions.values():
            if pos.symbol == opp['symbol'] and pos.exchange == opp['exchange']:
                existing_position = pos
                break
        
        # Execute the action
        if action == 'buy' and not existing_position:
            await self._open_position(
                strategy,
                opp,
                'buy',
                position_size,
                reason
            )
        elif action == 'sell' and not existing_position:
            # Short selling (if supported)
            if self.state.mode != TradingMode.PAPER:
                log.debug(f"Short selling not implemented for {opp['symbol']}")
            else:
                await self._open_position(
                    strategy,
                    opp,
                    'sell',
                    position_size,
                    reason
                )
        elif action in ['close', 'exit'] and existing_position:
            await self._close_position(existing_position, reason)

    async def _open_position(self, strategy: Strategy, opp: Dict, side: str, size: float, reason: str):
        """Open a new trading position."""
        
        # Risk checks
        if len(self.state.positions) >= 20:  # Max 20 concurrent positions
            log.debug("Maximum positions reached")
            return
        
        if size < Config.MIN_TRADE_SIZE:
            log.debug(f"Position size ${size:.2f} below minimum")
            return
        
        # Calculate position details
        position = Position(
            id=hashlib.sha256(f"{strategy.id}_{opp['symbol']}_{time.time()}".encode()).hexdigest()[:16],
            strategy_id=strategy.id,
            symbol=opp['symbol'],
            exchange=opp['exchange'],
            side=side,
            amount=size / opp['price'] if opp['price'] > 0 else 0,
            entry_price=opp['price'],
            current_price=opp['price'],
            stop_loss=opp['price'] * (1 - strategy.stop_loss) if side == 'buy' else opp['price'] * (1 + strategy.stop_loss),
            take_profit=opp['price'] * (1 + strategy.take_profit) if side == 'buy' else opp['price'] * (1 - strategy.take_profit)
        )
        
        # Execute trade (paper or real)
        if self.state.mode == TradingMode.PAPER or strategy.status == StrategyStatus.PAPER:
            # Simulated execution
            log.info(f"📝 PAPER {side.upper()}: {position.amount:.4f} {opp['symbol']} @ ${opp['price']:.4f} (Strategy: {strategy.name})")
        else:
            # Real execution
            exchange = self.exchanges.get(opp['exchange'])
            if exchange:
                try:
                    order = await exchange.create_market_order(
                        opp['symbol'],
                        side,
                        position.amount
                    )
                    position.entry_price = order['average'] if order.get('average') else opp['price']
                    log.info(f"💰 EXECUTED {side.upper()}: {position.amount:.4f} {opp['symbol']} @ ${position.entry_price:.4f}")
                    
                    if self.notifier:
                        await self.notifier.send_alert(
                            f"💰 Opened {side.upper()} position:\n"
                            f"{opp['symbol']} @ ${position.entry_price:.4f}\n"
                            f"Size: ${size:.2f}\n"
                            f"Strategy: {strategy.name}\n"
                            f"Reason: {reason}"
                        )
                except Exception as e:
                    log.error(f"Failed to execute trade: {e}")
                    return
        
        # Add position to state
        self.state.positions[position.id] = position
        self.state.cash -= size
        
        # Update strategy metrics
        strategy.total_trades += 1
        strategy.last_trade = datetime.utcnow()

    async def _close_position(self, position: Position, reason: str = "Strategy exit"):
        """Close an existing position."""
        
        # Get current price
        exchange = self.exchanges.get(position.exchange)
        if exchange:
            try:
                ticker = await exchange.fetch_ticker(position.symbol)
                exit_price = ticker['last']
            except:
                exit_price = position.current_price
        else:
            exit_price = position.current_price
        
        # Calculate P&L
        if position.side == 'buy':
            pnl = (exit_price - position.entry_price) * position.amount
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl = (position.entry_price - exit_price) * position.amount
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        # Execute close (paper or real)
        if self.state.mode != TradingMode.PAPER:
            try:
                close_side = 'sell' if position.side == 'buy' else 'buy'
                order = await exchange.create_market_order(
                    position.symbol,
                    close_side,
                    position.amount
                )
                exit_price = order['average'] if order.get('average') else exit_price
                
                # Recalculate P&L with actual exit price
                if position.side == 'buy':
                    pnl = (exit_price - position.entry_price) * position.amount
                    pnl_pct = (exit_price - position.entry_price) / position.entry_price
                else:
                    pnl = (position.entry_price - exit_price) * position.amount
                    pnl_pct = (position.entry_price - exit_price) / position.entry_price
                    
            except Exception as e:
                log.error(f"Failed to close position: {e}")
                return
        
        # Update state
        self.state.cash += position.amount * exit_price
        self.state.daily_pnl += pnl
        self.state.total_pnl += pnl
        
        # Update strategy metrics
        strategy = self.state.strategies.get(position.strategy_id)
        if strategy:
            strategy.total_pnl += pnl
            if pnl > 0:
                strategy.winning_trades += 1
            
            # Update win rate
            strategy.win_rate = strategy.winning_trades / strategy.total_trades if strategy.total_trades > 0 else 0
            
            # Add to trade history
            strategy.trade_history.append({
                'symbol': position.symbol,
                'side': position.side,
                'entry': position.entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'opened': position.opened_at.isoformat(),
                'closed': datetime.utcnow().isoformat()
            })
            
            # Calculate Sharpe ratio (simplified)
            if len(strategy.trade_history) >= 10:
                returns = [t['pnl_pct'] for t in strategy.trade_history[-30:]]
                if returns:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    strategy.sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Update max drawdown
            cumulative_returns = []
            cumsum = 0
            for trade in strategy.trade_history:
                cumsum += trade['pnl']
                cumulative_returns.append(cumsum)
            
            if cumulative_returns:
                peak = max(cumulative_returns)
                if peak > 0:
                    current_dd = (peak - cumulative_returns[-1]) / peak
                    strategy.max_drawdown = max(strategy.max_drawdown, current_dd)
            
            # Save updated strategy
            await self.db.save_strategy(strategy)
        
        # Log the close
        emoji = "💚" if pnl > 0 else "💔"
        log.info(f"{emoji} Closed {position.symbol}: {pnl:+.2f} ({pnl_pct:+.1%}) - {reason}")
        
        # Notify if significant
        if abs(pnl) > 10 and self.notifier:
            await self.notifier.send_alert(
                f"{emoji} Position closed:\n"
                f"{position.symbol}: ${pnl:+.2f} ({pnl_pct:+.1%})\n"
                f"Reason: {reason}"
            )
        
        # Save trade to database
        await self.db.save_trade({
            'id': hashlib.sha256(f"{position.id}_close_{time.time()}".encode()).hexdigest()[:16],
            'strategy_id': position.strategy_id,
            'symbol': position.symbol,
            'exchange': position.exchange,
            'side': position.side,
            'amount': position.amount,
            'price': position.entry_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'opened_at': position.opened_at.isoformat(),
            'closed_at': datetime.utcnow().isoformat()
        })
        
        # Remove position from state
        del self.state.positions[position.id]

    async def _manage_positions(self):
        """Manage existing positions - check stops, targets, etc."""
        
        for position in list(self.state.positions.values()):
            try:
                # Get current price
                exchange = self.exchanges.get(position.exchange)
                if exchange:
                    ticker = await exchange.fetch_ticker(position.symbol)
                    current_price = ticker['last']
                else:
                    continue
                
                position.current_price = current_price
                
                # Calculate current P&L
                if position.side == 'buy':
                    position.pnl_pct = (current_price - position.entry_price) / position.entry_price
                else:
                    position.pnl_pct = (position.entry_price - current_price) / position.entry_price
                
                position.pnl = position.pnl_pct * position.amount * position.entry_price
                
                # Check stop loss
                if position.side == 'buy' and current_price <= position.stop_loss:
                    await self._close_position(position, "Stop loss hit")
                elif position.side == 'sell' and current_price >= position.stop_loss:
                    await self._close_position(position, "Stop loss hit")
                
                # Check take profit
                elif position.side == 'buy' and current_price >= position.take_profit:
                    await self._close_position(position, "Take profit hit")
                elif position.side == 'sell' and current_price <= position.take_profit:
                    await self._close_position(position, "Take profit hit")
                
                # Time-based exit (positions older than 24 hours)
                elif (datetime.utcnow() - position.opened_at).total_seconds() > 86400:
                    await self._close_position(position, "Time limit (24h)")
                    
            except Exception as e:
                log.error(f"Error managing position {position.id}: {e}")

    async def _update_state(self):
        """Update system state with current values."""
        
        # Calculate total equity
        positions_value = sum([p.amount * p.current_price for p in self.state.positions.values()])
        self.state.equity = self.state.cash + positions_value
        
        # Update win rate
        total_trades = sum([s.total_trades for s in self.state.strategies.values()])
        total_wins = sum([s.winning_trades for s in self.state.strategies.values()])
        self.state.win_rate = total_wins / total_trades if total_trades > 0 else 0
        
        # Reset daily P&L at midnight UTC
        if datetime.utcnow().hour == 0 and datetime.utcnow().minute < 1:
            self.state.daily_pnl = 0
            self.state.trades_today = 0
        
        self.state.last_update = datetime.utcnow()
        
        # Save state periodically (every 5 minutes)
        if self.state.last_update.minute % 5 == 0:
            await self.db.save_state(self.state)

    async def _check_risk_limits(self) -> bool:
        """Check if risk limits have been exceeded."""
        
        # Daily loss limit
        if self.state.daily_pnl < -self.state.equity * Config.MAX_DAILY_LOSS:
            log.warning(f"⚠️ Daily loss limit hit: ${self.state.daily_pnl:.2f}")
            return True
        
        # Maximum drawdown
        peak_equity = Config.INITIAL_CAPITAL + self.state.total_pnl
        if self.state.total_pnl < 0:
            current_drawdown = abs(self.state.total_pnl) / Config.INITIAL_CAPITAL
        else:
            current_drawdown = 0
        
        if current_drawdown > Config.MAX_DRAWDOWN:
            log.warning(f"⚠️ Max drawdown hit: {current_drawdown:.1%}")
            return True
        
        return False

    async def _monitoring_loop(self):
        """Monitor system health and performance."""
        
        while self.running:
            try:
                # Log current status
                log.info(
                    f"💰 Equity: ${self.state.equity:.2f} | "
                    f"P&L: ${self.state.total_pnl:+.2f} | "
                    f"Positions: {len(self.state.positions)} | "
                    f"Win Rate: {self.state.win_rate:.1%}"
                )
                
                # Check for stale positions
                for position in self.state.positions.values():
                    age = (datetime.utcnow() - position.opened_at).total_seconds() / 3600
                    if age > 48:  # 48 hours
                        log.warning(f"⚠️ Stale position: {position.symbol} open for {age:.1f} hours")
                
                # Check strategy health
                for strategy in self.state.strategies.values():
                    if strategy.status == StrategyStatus.ACTIVE and strategy.last_trade:
                        days_inactive = (datetime.utcnow() - strategy.last_trade).days
                        if days_inactive > 7:
                            log.warning(f"⚠️ Strategy {strategy.name} inactive for {days_inactive} days")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                log.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)

    async def _daily_report_loop(self):
        """Send daily reports."""
        
        while self.running:
            try:
                # Wait until 00:00 UTC
                now = datetime.utcnow()
                tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                wait_seconds = (tomorrow - now).total_seconds()
                
                await asyncio.sleep(wait_seconds)
                
                # Send daily report
                if self.notifier:
                    top_strategies = sorted(
                        self.state.strategies.values(),
                        key=lambda s: s.total_pnl,
                        reverse=True
                    )[:5]
                    
                    await self.notifier.send_daily_report(self.state, top_strategies)
                
                # Archive old trades
                for strategy in self.state.strategies.values():
                    if len(strategy.trade_history) > 1000:
                        strategy.trade_history = strategy.trade_history[-500:]  # Keep last 500
                
            except Exception as e:
                log.error(f"Daily report error: {e}")
                await asyncio.sleep(3600)

    async def shutdown(self):
        """Graceful shutdown of the system."""
        log.info("🔴 Initiating graceful shutdown...")
        
        self.running = False
        
        # Close all positions
        for position in list(self.state.positions.values()):
            await self._close_position(position, "System shutdown")
        
        # Save final state
        await self.db.save_state(self.state)
        
        # Save all strategies
        for strategy in self.state.strategies.values():
            await self.db.save_strategy(strategy)
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        
        # Close exchange connections
        for exchange in self.exchanges.values():
            await exchange.close()
        
        if self.notifier:
            await self.notifier.send_alert(
                f"🔴 System shutdown\n"
                f"Final equity: ${self.state.equity:.2f}\n"
                f"Total P&L: ${self.state.total_pnl:+.2f}\n"
                f"ROI: {(self.state.equity / Config.INITIAL_CAPITAL - 1) * 100:+.1f}%"
            )
        
        log.info("👋 Shutdown complete.")

    async def emergency_shutdown(self):
        """Emergency shutdown - close everything immediately."""
        log.error("🚨 EMERGENCY SHUTDOWN INITIATED")
        
        self.running = False
        self.state.mode = TradingMode.EMERGENCY
        
        # Try to close positions
        try:
            for position in list(self.state.positions.values()):
                await self._close_position(position, "EMERGENCY")
        except:
            pass
        
        # Force save state
        try:
            await self.db.save_state(self.state)
        except:
            pass
        
        # Alert
        if self.notifier:
            try:
                await self.notifier.send_alert("🚨 EMERGENCY SHUTDOWN - Check logs!")
            except:
                pass
        
        # Exit
        sys.exit(1)

# ═══════════════════════════════ MAIN ENTRY POINT ═══════════════════════════════

async def main():
    """Main entry point for the v26meme autonomous trading system."""
    
    # Display startup banner
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold yellow]           v26meme Autonomous Trading System            [/bold yellow]")
    console.print("[bold green]            $200 → $1M in 90 Days Challenge            [/bold green]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
    
    # Create and initialize trader
    trader = AutonomousTrader()
    
    if not await trader.initialize():
        log.error("❌ Failed to initialize system")
        return
    
    # Start trading
    console.print("[bold green]🚀 System initialized successfully![/bold green]")
    console.print(f"[yellow]Mode: {trader.state.mode.value}[/yellow]")
    console.print(f"[yellow]Initial Capital: ${Config.INITIAL_CAPITAL}[/yellow]")
    console.print(f"[yellow]Target: ${Config.TARGET_CAPITAL:,} in {Config.TARGET_DAYS} days[/yellow]\n")
    
    # Run the system
    await trader.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]⛔ Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Fatal error: {e}[/red]")
        traceback.print_exc()