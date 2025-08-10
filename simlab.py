#!/usr/bin/env python3
"""
SimLab: parallel, always-on simulation & replay engine for v26meme

What it does:
- Runs historical replays and synthetic scenario sims in the background
- Evaluates all current strategies on OHLCV bars without touching live loops
- Writes results to SQLite (run metadata, sim trades, metrics)
- Surfaces best configs for the genetic engine to steal from

Zero-risk: it has no side-effects on live/paper trading.
"""

import os
import asyncio
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# ---- Lightweight config (reads env; no dependency on your Config class) ----
SIM_DATA_DIR = os.getenv("SIMLAB_DATA_DIR", "data")   # put csv/parquet here
SIM_MAX_CONCURRENCY = int(os.getenv("SIMLAB_CONCURRENCY", "2"))
SIM_ENABLED = os.getenv("SIMLAB_ENABLED", "true").lower() == "true"
SIM_DEFAULT_EXCHANGE = os.getenv("SIM_DEFAULT_EXCHANGE", "sim")

# ---- Schemas ----------------------------------------------------------------
@dataclass
class SimRun:
    id: str
    created_at: str
    kind: str               # "replay" | "scenario"
    symbol: str
    timeframe: str
    start: str
    end: str
    params: Dict[str, Any]
    status: str             # "queued" | "running" | "done" | "error"
    notes: str = ""

@dataclass
class SimTrade:
    run_id: str
    strategy_id: str
    strategy_name: str
    symbol: str
    side: str
    entry_ts: str
    exit_ts: str
    entry: float
    exit: float
    pnl: float
    pnl_pct: float
    sl_hit: int
    tp_hit: int

@dataclass
class SimMetrics:
    run_id: str
    strategy_id: str
    strategy_name: str
    trades: int
    wins: int
    win_rate: float
    pnl: float
    sharpe_like: float
    max_dd: float
    avg_trade: float

# ---- SimLab engine ----------------------------------------------------------
class SimLab:
    """
    Background simulation orchestrator.
    You pass:
      - db_path: your SQLite file
      - run_strategy_callable(state, strategy, opp) -> decision dict
      - fetch_strategies_callable() -> Dict[str, Strategy]
    """

    def __init__(
        self,
        db_path: str,
        run_strategy_callable,
        fetch_strategies_callable,
        logger=None
    ):
        self.db_path = db_path
        self._run_strategy = run_strategy_callable
        self._fetch_strategies = fetch_strategies_callable
        self._log = logger
        self._running = False
        self._sem = asyncio.Semaphore(SIM_MAX_CONCURRENCY)
        self._queue: asyncio.Queue[SimRun] = asyncio.Queue()
        self._ensure_db()

    # -- Public API -----------------------------------------------------------
    async def start_background(self):
        if not SIM_ENABLED:
            self._log and self._log.info("SimLab disabled (SIMLAB_ENABLED=false)")
            return
        if self._running:
            return
        self._running = True
        self._log and self._log.info("🧪 SimLab starting (background)")
        asyncio.create_task(self._runner_loop())
        asyncio.create_task(self._autodiscover_loop())  # auto-queue replays

    async def enqueue_replay(
        self, symbol: str, timeframe: str, start: str, end: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Manually enqueue a replay job."""
        run = SimRun(
            id=self._new_id("simrun"),
            created_at=datetime.utcnow().isoformat(),
            kind="replay",
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            params=params or {},
            status="queued",
        )
        self._insert_run(run)
        await self._queue.put(run)
        return run.id

    # -- Internals ------------------------------------------------------------
    async def _runner_loop(self):
        while self._running:
            try:
                run: SimRun = await self._queue.get()
                await self._execute_run(run)
            except Exception as e:
                self._log and self._log.error(f"SimLab runner error: {e}")
                await asyncio.sleep(2)

    async def _autodiscover_loop(self):
        """
        Every N minutes, check data dir for new {symbol}_{timeframe}.csv/parquet and enqueue.
        Keeps a small rolling queue going forever.
        """
        seen = set()
        while self._running:
            try:
                for fn in os.listdir(SIM_DATA_DIR):
                    # Handle .csv, .csv.gz, and .parquet files
                    if not (fn.endswith(".csv") or fn.endswith(".csv.gz") or fn.endswith(".parquet")):
                        continue
                    
                    # Extract stem (remove .csv.gz, .csv, or .parquet)
                    if fn.endswith(".csv.gz"):
                        stem = fn[:-7]  # remove .csv.gz
                    elif fn.endswith(".parquet"):
                        stem = fn[:-8]  # remove .parquet
                    else:
                        stem = os.path.splitext(fn)[0]  # remove .csv
                        
                    if stem in seen:
                        continue
                    # Expect filenames like: BTC_USDT_5m.csv  OR  ETH_USDC_15m.parquet
                    parts = stem.split("_")
                    if len(parts) < 3:
                        continue
                    symbol = f"{parts[0]}/{parts[1]}"
                    timeframe = parts[2]
                    run_id = await self.enqueue_replay(
                        symbol=symbol,
                        timeframe=timeframe,
                        start="",
                        end="",
                        params={"file": os.path.join(SIM_DATA_DIR, fn)}
                    )
                    seen.add(stem)
                    self._log and self._log.info(f"🧪 SimLab enqueued {stem} (run {run_id})")
            except FileNotFoundError:
                os.makedirs(SIM_DATA_DIR, exist_ok=True)
            except Exception as e:
                self._log and self._log.error(f"SimLab autodiscover error: {e}")
            await asyncio.sleep(300)  # every 5 minutes

    async def _execute_run(self, run: SimRun):
        self._update_run_status(run.id, "running")
        try:
            df = self._load_bars(run)
            if df.empty:
                self._update_run_status(run.id, "error", notes="empty dataset")
                return

            strategies = self._fetch_strategies()
            if not strategies:
                self._update_run_status(run.id, "error", notes="no strategies")
                return

            # Sim state (local, isolated from live)
            class SimState:
                def __init__(self):
                    self.equity = 200.0
                    self.cash = 200.0
                    self.positions: Dict[str, Dict[str, Any]] = {}
                    self.start_time = datetime.utcnow()

            state = SimState()

            # per-strategy metrics holders
            per_strat_trades: Dict[str, List[SimTrade]] = {sid: [] for sid in strategies.keys()}
            per_strat_returns: Dict[str, List[float]] = {sid: [] for sid in strategies.keys()}
            per_strat_curve: Dict[str, List[float]] = {sid: [0.0] for sid in strategies.keys()}

            # iterate bars
            for i in range(1, len(df)):
                bar = df.iloc[i]
                prev = df.iloc[i - 1]
                opp = self._make_opp(run.symbol, bar, prev)

                # evaluate all strategies
                for sid, strat in strategies.items():
                    # Build a tiny Strategy-like shim for naming
                    strat_name = getattr(strat, "name", sid)

                    # Existing positions?
                    pos_key = f"{sid}|{run.symbol}"
                    pos = state.positions.get(pos_key)

                    # Entry/exit decision
                    try:
                        # Reuse your live runner signature: (strategy, opp) → dict
                        decision = await self._run_strategy(strat, opp)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        continue

                    action = (decision or {}).get("action", "hold")
                    conf = float((decision or {}).get("conf", 0) or 0)

                    # Close logic on TP/SL while holding
                    if pos is not None:
                        exit_reason = None
                        price = float(bar["close"])
                        if pos["side"] == "buy":
                            if price <= pos["sl"]:
                                exit_reason = "sl"
                            elif price >= pos["tp"]:
                                exit_reason = "tp"
                        else:  # sell
                            if price >= pos["sl"]:
                                exit_reason = "sl"
                            elif price <= pos["tp"]:
                                exit_reason = "tp"

                        if action in ("close", "exit"):
                            exit_reason = "manual"

                        if exit_reason:
                            entry = pos["entry"]
                            exitp = float(bar["close"])
                            pnl = (exitp - entry) * pos["qty"] * (1 if pos["side"] == "buy" else -1)
                            pnl_pct = pnl / max(1e-9, pos["notional"])
                            tr = SimTrade(
                                run_id=run.id,
                                strategy_id=sid,
                                strategy_name=strat_name,
                                symbol=run.symbol,
                                side=pos["side"],
                                entry_ts=pos["ts"].isoformat(),
                                exit_ts=str(bar["ts"]),
                                entry=entry,
                                exit=exitp,
                                pnl=pnl,
                                pnl_pct=pnl_pct,
                                sl_hit=1 if exit_reason == "sl" else 0,
                                tp_hit=1 if exit_reason == "tp" else 0,
                            )
                            per_strat_trades[sid].append(tr)
                            per_strat_returns[sid].append(pnl_pct)
                            per_strat_curve[sid].append(per_strat_curve[sid][-1] + pnl)

                            # clear position
                            del state.positions[pos_key]
                            continue  # go next strategy

                    # Entry logic
                    if pos is None and action in ("buy", "sell") and conf >= 0.20:
                        price = float(bar["close"])
                        notional = 15.0  # small probe; you can tune
                        qty = notional / max(1e-9, price)
                        sl = decision.get("sl") or (price * (0.985 if action == "buy" else 1.015))
                        tp = decision.get("tp") or (price * (1.02 if action == "buy" else 0.98))
                        state.positions[pos_key] = {
                            "side": action,
                            "entry": price,
                            "qty": qty,
                            "notional": notional,
                            "sl": float(sl),
                            "tp": float(tp),
                            "ts": bar["ts"],
                        }

            # Close any leftovers at final bar
            last_bar = df.iloc[-1]
            for pos_key, pos in list(state.positions.items()):
                sid = pos_key.split("|")[0]
                strat = strategies.get(sid)
                if not strat:
                    continue
                strat_name = getattr(strat, "name", sid)
                exitp = float(last_bar["close"])
                pnl = (exitp - pos["entry"]) * pos["qty"] * (1 if pos["side"] == "buy" else -1)
                pnl_pct = pnl / max(1e-9, pos["notional"])
                tr = SimTrade(
                    run_id=run.id,
                    strategy_id=sid,
                    strategy_name=strat_name,
                    symbol=run.symbol,
                    side=pos["side"],
                    entry_ts=pos["ts"].isoformat(),
                    exit_ts=str(last_bar["ts"]),
                    entry=pos["entry"],
                    exit=exitp,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    sl_hit=0,
                    tp_hit=0,
                )
                per_strat_trades[sid].append(tr)
                per_strat_returns[sid].append(pnl_pct)
                per_strat_curve[sid].append(per_strat_curve[sid][-1] + pnl)

            # Persist results
            self._store_results(run.id, per_strat_trades, per_strat_returns, per_strat_curve)
            self._update_run_status(run.id, "done")
        except Exception as e:
            self._update_run_status(run.id, "error", notes=str(e))

    # -- Helpers --------------------------------------------------------------
    def _make_opp(self, symbol: str, bar: pd.Series, prev: pd.Series) -> Dict[str, Any]:
        # Build an opp dict that mirrors what your live scanner feeds strategies
        last = float(bar["close"])
        openp = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        v = float(bar["volume"])
        vwap = (high + low + last) / 3.0
        # change_24h is not available on bar; approximate from prev close
        change_approx = ((last - float(prev["close"])) / max(1e-9, float(prev["close"]))) * 100.0

        return {
            "symbol": symbol,
            "exchange": SIM_DEFAULT_EXCHANGE,
            "price": last,
            "current_price": last,
            "volume_24h_usd": v * last,         # rough
            "volume_24h": v * last,
            "change_24h_pct": change_approx,
            "change_24h": change_approx,
            "bid": last * 0.999,
            "ask": last * 1.001,
            "spread_bps": 20.0,
            "timestamp": bar["ts"],
            "high_24h": high,
            "low_24h": low,
            "open_24h": openp,
            "vwap": vwap,
        }

    def _load_bars(self, run: SimRun) -> pd.DataFrame:
        path = run.params.get("file")
        if not path:
            # fall back to conventional filename - check multiple formats
            sym = run.symbol.replace("/", "_")
            for ext in [".parquet", ".csv.gz", ".csv"]:
                candidate = os.path.join(SIM_DATA_DIR, f"{sym}_{run.timeframe}{ext}")
                if os.path.exists(candidate):
                    path = candidate
                    break
            
            if not path:
                return pd.DataFrame()

        # Load based on file extension
        try:
            if path.endswith(".parquet"):
                df = pd.read_parquet(path)
            elif path.endswith(".csv.gz"):
                df = pd.read_csv(path, compression='gzip')
            else:
                df = pd.read_csv(path)
        except Exception as e:
            self._log and self._log.error(f"Failed to load {path}: {e}")
            return pd.DataFrame()

        # Expect columns: timestamp(ms) | open | high | low | close | volume
        # Normalize
        cols = {c.lower(): c for c in df.columns}
        def pick(name): return cols.get(name, name)
        if "timestamp" in [c.lower() for c in df.columns]:
            ts = df[pick("timestamp")]
            if ts.max() > 10_000_000_000:  # likely ms
                ts = pd.to_datetime(ts, unit="ms")
            else:
                ts = pd.to_datetime(ts, unit="s")
        elif "ts" in [c.lower() for c in df.columns]:
            ts = pd.to_datetime(df[pick("ts")])
        else:
            raise ValueError("dataset missing timestamp column")

        out = pd.DataFrame({
            "ts": ts,
            "open": df[pick("open")].astype(float),
            "high": df[pick("high")].astype(float),
            "low": df[pick("low")].astype(float),
            "close": df[pick("close")].astype(float),
            "volume": df[pick("volume")].astype(float),
        }).sort_values("ts").reset_index(drop=True)

        # clip to start/end if provided
        if run.start:
            out = out[out["ts"] >= pd.to_datetime(run.start)]
        if run.end:
            out = out[out["ts"] <= pd.to_datetime(run.end)]
        return out

    def _ensure_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sim_runs (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            kind TEXT,
            symbol TEXT,
            timeframe TEXT,
            start TEXT,
            end TEXT,
            params TEXT,
            status TEXT,
            notes TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sim_trades (
            run_id TEXT,
            strategy_id TEXT,
            strategy_name TEXT,
            symbol TEXT,
            side TEXT,
            entry_ts TEXT,
            exit_ts TEXT,
            entry REAL,
            exit REAL,
            pnl REAL,
            pnl_pct REAL,
            sl_hit INTEGER,
            tp_hit INTEGER
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sim_metrics (
            run_id TEXT,
            strategy_id TEXT,
            strategy_name TEXT,
            trades INTEGER,
            wins INTEGER,
            win_rate REAL,
            pnl REAL,
            sharpe_like REAL,
            max_dd REAL,
            avg_trade REAL
        )""")
        conn.commit()
        conn.close()

    def _insert_run(self, run: SimRun):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO sim_runs VALUES (?,?,?,?,?,?,?,?,?,?)", (
            run.id, run.created_at, run.kind, run.symbol, run.timeframe,
            run.start, run.end, json.dumps(run.params), run.status, run.notes
        ))
        conn.commit()
        conn.close()

    def _update_run_status(self, run_id: str, status: str, notes: str = ""):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("UPDATE sim_runs SET status=?, notes=? WHERE id=?", (status, notes, run_id))
        conn.commit()
        conn.close()

    def _store_results(
        self,
        run_id: str,
        per_strat_trades: Dict[str, List[SimTrade]],
        per_strat_returns: Dict[str, List[float]],
        per_strat_curve: Dict[str, List[float]]
    ):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # trades
        for sid, trades in per_strat_trades.items():
            for t in trades:
                cur.execute("""INSERT INTO sim_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    t.run_id, t.strategy_id, t.strategy_name, t.symbol, t.side,
                    t.entry_ts, t.exit_ts, t.entry, t.exit, t.pnl, t.pnl_pct, t.sl_hit, t.tp_hit
                ))

        # metrics
        for sid, returns in per_strat_returns.items():
            curve = per_strat_curve[sid]
            pnl_total = float(sum([x for x in (returns or [])]))  # approx (% sum)
            wins = int(sum([1 for r in returns if r > 0]))
            trades = len(returns)
            win_rate = (wins / trades) if trades else 0.0
            avg = (pnl_total / trades) if trades else 0.0
            std = float(np.std(returns)) if trades else 0.0
            sharpe_like = (avg / std * math.sqrt(252)) if std > 0 else 0.0
            # max drawdown on cumulative PnL (absolute)
            dd = 0.0
            peak = -1e9
            for v in curve:
                if v > peak:
                    peak = v
                dd = max(dd, peak - v)
            # get a representative name
            strat_name = None
            if per_strat_trades[sid]:
                strat_name = per_strat_trades[sid][0].strategy_name
            else:
                strat_name = sid
            cur.execute("""INSERT INTO sim_metrics VALUES (?,?,?,?,?,?,?,?,?,?)""", (
                run_id, sid, strat_name, trades, wins, win_rate, pnl_total, sharpe_like, dd, avg
            ))

        conn.commit()
        conn.close()

    # -- utils ----------------------------------------------------------------
    def _new_id(self, prefix: str) -> str:
        import hashlib
        import time
        import random
        return hashlib.sha256(f"{prefix}_{time.time()}_{random.random()}".encode()).hexdigest()[:16]
