# v26meme – Autonomous Crypto‑Trading Black‑Box 🛰️

> **Goal:** compound **\$200 → \$1 000 000** on U.S‑compliant spot pairs (Coinbase Pro) using an LLM‑generated, self‑evolving ensemble of systematic strategies. Drop the code + this README in a repo, add API keys to `.env`, run `python v26meme.py`, and the bot bootstraps itself.

---

## 1 · Features

| Capability                  | Detail                                                                                                    |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Fully‑autonomous loop**   | One process pulls data, generates signals, sizes trades, executes (paper → real), logs, and self‑evolves. |
| **LLM‑seeded edge**         | GPT‑4o‑mini creates starter strategies; hourly evolution hook ready for RL / genetic refactors.           |
| **Capital‑adaptive sizing** | Fractional‑Kelly sizing capped at 25 % of bankroll, dynamic per‑strategy win‑rate CI.                     |
| **Hard risk rails**         | 20 % daily draw‑down circuit‑breaker, liquidity filter (\$10k+) and slippage cap.                         |
| **Zero‑touch deploy**       | Runtime installs missing Python deps, fetches data directly from exchange REST, no DB required.           |
| **Budget‑fit infra**        | Runs fine on a \$5‑10/mo VPS or a free GitHub Codespace.                                                  |

---

## 2 · Directory layout

```
repo/
├── v26meme.py        # single‑file bot (already in canvas)
├── README.md         # ← you are here
├── .env.example      # template for secrets
└── state.json        # persist file (auto‑created)
```

---

## 3 · Quick Start

```bash
# 1 · clone & enter
$ git clone https://github.com/<you>/v26meme.git
$ cd v26meme

# 2 · set secrets (Coinbase keys + OpenAI)
$ cp .env.example .env && nano .env

# 3 · run
$ python v26meme.py
$ tail -f v26meme.log   # follow logs
```

`.env.example` template:

```
OPENAI_API_KEY=sk‑...
COINBASE_API_KEY=...
COINBASE_SECRET=...
COINBASE_PASSPHRASE=...
```

> **Tip:** if you also have Kraken keys, export `KRAKEN_API_KEY` / `KRAKEN_SECRET` – the bot auto‑detects.

---

## 4 · What happens on boot?

1. **Dependency bootstrap** – installs `openai`, `ccxt`, `pandas`, `numpy`, etc.
2. **LLM seeding** – `gpt‑4o‑mini` is asked (via structured function‑call) for three starter strategies (`momentum_tracker`, `mean_reversion`, `social_sentiment`).
3. **Back‑test sanity‑check** – each strategy must hit **Sharpe > 1.2** and **Max DD < 20 %** on the last 1 000 × 5‑min BTC/USD candles.
4. **Main loop** (every minute):

   * pulls fresh Coinbase tickers;
   * each strategy emits a JSON signal;
   * fractional‑Kelly position sizing;
   * paper trade (graduating to real once win‑rate ≥ 40 % on 50 trades & 7 days).
5. **Evolution hook** – hourly placeholder ready to call LLM/RL to refactor or spawn new strategies.

ASCII diagram:

```
┌──────── GPT‑4o ────────┐
│  new_strategy() fn‑call│
└───┬───────────┬───────┘
    │           │
┌───▼───┐   ┌───▼───┐ ...  ↻ hourly evolve
│ strat │   │ strat │
└─┬─────┘   └─────┬─┘
  │ Kelly‑sized orders
┌─▼───────────────┐
│ Coinbase Pro API│
└─────────────────┘
```

---

## 5 · Graduation logic (`PAPER` → `REAL`)

| Condition             | Threshold               |
| --------------------- | ----------------------- |
| Trading days in PAPER | ≥ 7 days                |
| Total trades          | ≥ 50                    |
| Win rate              | ≥ 40 % (Wilson 95 % CI) |

Meet all three? The bot flips `self.state.mode` to `REAL` and begins live orders.

---

## 6 · Safety rails

* **Daily Circuit‑Breaker:** pause if equity ≤ 80 % of starting‑day equity.
* **Liquidity filter:** skip symbols with < \$10 000 depth at top‑of‑book.
* **Kelly cap:** never risk > 25 % bankroll on any single strategy suggestion.
* **Runtime sandbox:** strategy code compiled & exec’d in isolated namespace; banned keywords (`eval`, `exec`, `os.system`, etc.).

---

## 7 · Extending the bot

| Task                    | Where / How                                                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Add RL evolution**    | Implement `FullyAutonomousTrader.evolve()` – feed bad performers into `stable_baselines3` or ask the LLM for refactor. |
| **Plug real execution** | Fill in `execute_real_trade()` (placeholder in code) – map actions to `ccxt.create_market_buy_order` / `sell`.         |
| **More markets**        | Append APIs (Kraken, Binance US) in `setup_exchanges()` with proper credential env vars.                               |
| **Observability**       | Pipe logs to Grafana Cloud or add Prometheus client calls for metrics.                                                 |

---

## 8 · Troubleshooting

| Symptom                       | Fix                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError` on boot | The runtime installer should catch it; if not, `pip install -r <(python v26meme.py --print-req)` |
| LLM quota errors              | Check `OPENAI_API_KEY` credit; fallback to `gpt‑4o-mini` instead of larger models.               |
| Bot stops after drawdown      | Inspect `v26meme.log`, wait 24 h or edit `self.safety['max_daily_dd']` cautiously.               |

---


