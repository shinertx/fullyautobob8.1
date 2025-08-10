#!/usr/bin/env python3
"""
Lightweight strategy backtester for rapid gating before PAPER.
Fetches OHLCV via public exchange API and simulates decisions.
"""

import asyncio
import math
from typing import Dict, List

import ccxt.async_support as ccxt


def _build_opp_from_candle(symbol: str, candle: List[float], last24: List[List[float]]) -> Dict:
    ts, o, h, low, c, v = candle
    # Approximate 24h stats from last 288 x 5m candles if provided
    prices = [cndl[4] for cndl in last24] or [c]
    high24 = max(prices) if prices else c
    low24 = min(prices) if prices else c
    open24 = prices[0] if prices else c
    change_24h = ((c - open24) / open24 * 100.0) if open24 else 0.0
    volume_quote = sum([cndl[4] * cndl[5] for cndl in last24[-288:]]) if last24 else (c * v)

    return {
        'symbol': symbol,
        'exchange': 'kraken',
        'price': c,
        'current_price': c,
        'volume': volume_quote,
        'volume_24h': volume_quote,
        'high_24h': high24,
        'low_24h': low24,
        'open_24h': open24,
        'change_24h': change_24h,
        'vwap': (o + h + l + c) / 4.0 if all([o, h, l, c]) else c,
    }


async def quick_backtest(strategy_code: str, symbol: str = 'ETH/USDT', timeframe: str = '5m', limit: int = 300) -> Dict[str, float]:
    """
    Run a quick backtest of a generated strategy on public OHLCV.
    Returns metrics dict with pnl, sharpe-like score, win_rate estimate.
    """
    # Prepare sandbox
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    exec_globals = {
        '__builtins__': {
            'len': len, 'abs': abs, 'min': min, 'max': max, 'sum': sum,
            'round': round, 'float': float, 'int': int, 'str': str, 'bool': bool,
            'dict': dict, 'list': list, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'any': any, 'all': all,
            'Exception': Exception, 'ValueError': ValueError, 'KeyError': KeyError,
            'TypeError': TypeError, 'ZeroDivisionError': ZeroDivisionError,
            'AttributeError': AttributeError, 'IndexError': IndexError,
            'print': lambda *a, **k: None,
            '__name__': '__main__', '__build_class__': __build_class__, 'type': type,
            'isinstance': isinstance, 'hasattr': hasattr, 'getattr': getattr,
            'setattr': setattr, 'tuple': tuple, 'set': set, 'sorted': sorted,
            'reversed': reversed, 'filter': filter, 'map': map, 'callable': callable,
            'iter': iter, 'next': next, 'property': property, 'staticmethod': staticmethod,
            'classmethod': classmethod, 'super': super, 'object': object,
            'None': None, 'True': True, 'False': False
        },
        'np': np, 'pd': pd, 'datetime': datetime, 'timedelta': timedelta,
        'math': math,
        'safe_divide': lambda a, b, default=0: a / b if b != 0 else default,
        'safe_get': lambda d, key, default=0: d.get(key, default) if isinstance(d, dict) else default,
    }

    # Execute strategy code
    try:
        exec(strategy_code, exec_globals)
        if 'execute_strategy' not in exec_globals:
            return {'pnl': 0.0, 'sharpe_like': 0.0, 'win_rate': 0.0, 'trades': 0}
    except Exception:
        return {'pnl': 0.0, 'sharpe_like': 0.0, 'win_rate': 0.0, 'trades': 0}

    # Fetch OHLCV
    ex = ccxt.kraken({'enableRateLimit': True})
    try:
        ohlcv = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        await ex.close()
        return {'pnl': 0.0, 'sharpe_like': 0.0, 'win_rate': 0.0, 'trades': 0}

    # Simulate
    cash = 1000.0
    position = None
    equity_curve = []
    wins = 0
    trades = 0
    last24 = []

    for i, candle in enumerate(ohlcv):
        last24.append(candle)
        if len(last24) > 288:
            last24.pop(0)

        opp = _build_opp_from_candle(symbol, candle, last24)
        try:
            decision = await exec_globals['execute_strategy']({'equity': cash}, opp)  # lightweight state
        except Exception:
            decision = {'action': 'hold', 'conf': 0.0}

        action = decision.get('action', 'hold')
        conf = float(decision.get('conf', 0.0) or 0.0)

        price = opp['price']

        if action == 'buy' and conf >= 0.4 and position is None:
            size = cash * 0.05 / price
            if size > 0:
                position = {'size': size, 'entry': price}
                cash -= size * price
                trades += 1
        elif action in ('close', 'sell') and position is not None:
            pnl = (price - position['entry']) * position['size']
            cash += position['size'] * price
            if pnl > 0:
                wins += 1
            position = None

        # Mark-to-market
        equity = cash + (position['size'] * price if position else 0.0)
        equity_curve.append(equity)

    await ex.close()

    if len(equity_curve) < 2:
        return {'pnl': 0.0, 'sharpe_like': 0.0, 'win_rate': 0.0, 'trades': trades}

    pnl = equity_curve[-1] - 1000.0
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe_like = (np.mean(returns) / (np.std(returns) + 1e-9)) * math.sqrt(252*24*12) if len(returns) > 1 else 0.0
    win_rate = (wins / trades) if trades > 0 else 0.0

    return {'pnl': float(pnl), 'sharpe_like': float(sharpe_like), 'win_rate': float(win_rate), 'trades': trades}


async def main():
    # Simple manual test
    code = """
async def execute_strategy(state, opp):
    # naive momentum
    conf = 0.5 if opp.get('change_24h', 0) > 0 else 0.0
    return {'action': 'buy' if conf>0.4 else 'hold', 'conf': conf}
"""
    metrics = await quick_backtest(code)
    print(metrics)


if __name__ == '__main__':
    asyncio.run(main())


