README.md

markdown
Copy
Edit
# v26meme_full

**100 % AI-Driven Solana/Memecoin Trading Bot**

A single-file, self-bootstrapping Python bot that autonomously compounds \$200 → \$1 000 000 (default 60-day horizon) via high-conviction, AI-generated strategies.  

## Features

- **Market Scanning**  
  - Coinbase top-10 24 h movers (>20 %)  
  - CoinGecko trending memecoins  
  - GPT-4 venue selection for best liquidity  

- **Opportunity Research**  
  - GPT-4 bull vs. bear scoring → conviction ∈ [0,1]  

- **Strategy Engineering**  
  - GPT-4 generates full `async execute_strategy(state, opp)` code  
  - Back-tests on 2 days of 5 min OHLC  
  - Enforces Sharpe ≥ 1.2 & MaxDD ≥ –50 %  

- **Position Sizing & Execution**  
  - Fractional-Kelly (Wilson lower bound) sizing, cap tiers (3 %→30 %)  
  - Simulated “paper” orders by default; real market orders when run with `REAL` flag  
  - Trailing stops + hard stops (–30 %) & take-profit (+200 %)  

- **Risk Management**  
  - 50 % daily drawdown pause (6 h)  
  - 90 % global kill-switch  

- **Evolution & Self-Healing**  
  - Hourly promote/retire strategies based on win-rate & Wilson score  
  - Console, WebSocket & webhook alerts for key events  
  - Built-in web dashboard:  
    - JSON state → `http://<ip>:8000/state`  
    - Live logs → `ws://<ip>:8000/ws`  

## Getting Started

### 1. Install Dependencies

```bash
pip install -r <(python v26meme_full.py --deps)
2. First-Time Setup
bash
Copy
Edit
python v26meme_full.py
You’ll be prompted to enter any missing keys:

OPENAI_API_KEY

COINBASE_API_KEY

COINBASE_SECRET

Values are stored in .env for future runs.

3. Paper-Trade
bash
Copy
Edit
python v26meme_full.py
Runs in PAPER mode by default.

Simulates all orders, logs PnL, evolves strategies.

4. Go Live
bash
Copy
Edit
python v26meme_full.py REAL
Switches to REAL mode.

Places live market orders on configured exchanges.

5. Monitor
Dashboard: http://<server-ip>:8000/state

Logs: Connect a WebSocket client to ws://<server-ip>:8000/ws

Configuration
.env keys:

ini
Copy
Edit
OPENAI_API_KEY=sk-...
COINBASE_API_KEY=...
COINBASE_SECRET=...
KRAKEN_API_KEY=...      # optional
KRAKEN_SECRET=...       # optional
ALERT_WEBHOOK=...       # optional Discord/Slack webhook URL
State is persisted in state.json.

License & Disclaimer
High-risk experimental code.
Use at your own peril. Strategies may incur large drawdowns or losses.