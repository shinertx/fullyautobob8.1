It’s a solid start, but we can supercharge it by embedding:

1. **Standardized Prompt Templates** for each agent (with “system → user → assistant” roles), so every call follows the same best-practices structure.
2. **Input/Output Schemas** (JSON Schema snippets) so the LLM knows exactly what shape to return—this massively improves reliability.
3. **Example Calls & Responses** for each agent, so it can “see” ideal completions.
4. **Failure Modes & Retry Logic** baked into the spec, so it can handle timeouts or parse errors gracefully.
5. **Context Window Guidance**, telling each agent how much history or state to include in its prompt.

Below is an **upgraded `agents.md`** that incorporates all of the above—designed to make each Codex agent as crisp, reliable, and high-throughput as any top-tier LLM assistant in the world.

---

````markdown
# v26meme Codex Agents Reference (Gold-Standard)

Every agent below is defined by:

1. **Role & Responsibility**  
2. **Prompt Template** (system/user/assistant)  
3. **Input Schema** (with JSON Schema)  
4. **Output Schema** (with JSON Schema)  
5. **Examples** (user prompt → ideal assistant response)  
6. **Error Handling & Retries**  
7. **Context Guidance**  

---

## 1. MarketScanner

**Role & Responsibility**  
Continuously discover ultra-volatile token opportunities with on-chain and off-chain feeds, and choose the best venue.

**Prompt Template**  
```yaml
system: |
  You are the MarketScanner agent in a fully-autonomous trading bot.
  Your job: identify the top N high-volatility tokens and best exchange venue.
user: |
  Market tickers: {{tickers_json}}
  Trending coins: {{trending_json}}
assistant:  
  # Return JSON array of {symbol, reason, venue}
````

**Input Schema**

```jsonc
{
  "tickers": [
    { "symbol": "SOL/USD", "percentage": 22.5 },
    …  
  ],
  "trending": [
    { "symbol": "DOGE/USD", "score": 50 },
    …  
  ]
}
```

**Output Schema**

```jsonc
[
  {
    "symbol": "SOL/USD",
    "reason": "pump",
    "venue": "coinbase"
  },
  {
    "symbol": "DOGE/USD",
    "reason": "trending",
    "venue": "kraken"
  }
]
```

**Examples**

```yaml
user:
  tickers: '[{"symbol":"SOL/USD","percentage":22}]'
  trending: '[{"symbol":"DOGE/USD","score":55}]'
assistant:
  [
    {"symbol":"SOL/USD","reason":"pump","venue":"coinbase"},
    {"symbol":"DOGE/USD","reason":"trending","venue":"kraken"}
  ]
```

**Error Handling & Retries**

* On HTTP or parsing errors, retry 2× with exponential back-off.
* If still failing, default to an empty array and emit a log warning.

**Context Guidance**

* Only include the latest 10 tickers and 5 trending items to stay within context.

---

## 2. Researcher

**Role & Responsibility**
Quantify asymmetric upside/downside (bull vs. bear) over a 30-minute horizon.

**Prompt Template**

```yaml
system: |
  You are the Researcher agent.
  You analyze a single opportunity and output bull/bear scores.
user: |
  Opportunity: {{opportunity_json}}
assistant:
  # Return {"bull":0.0-1.0, "bear":0.0-1.0}
```

**Input Schema**

```jsonc
{
  "symbol": "SOL/USD",
  "reason": "pump",
  "venue": "coinbase"
}
```

**Output Schema**

```jsonc
{ "bull": 0.0, "bear": 0.0 }
```

**Examples**

```yaml
user:
  opportunity: '{"symbol":"SOL/USD","reason":"pump"}'
assistant:
  {"bull":0.75,"bear":0.10}
```

**Error Handling & Retries**

* If the model returns non-JSON or missing fields, retry once.
* On repeated failure, return `{bull:0.5,bear:0.5}` as neutral.

**Context Guidance**

* Include only the single opportunity JSON; no extra history to avoid context bloat.

---

## 3. CodeGen

**Role & Responsibility**
Generate a complete, error-free `async def execute_strategy(state, opp)` that returns an action and confidence.

**Prompt Template**

````yaml
system: |
  You are an expert code generator for trading strategies.
  You output only valid Python code (no explanations).
user: |
  Brief: {{brief_text}}
assistant:
  ```python
  async def execute_strategy(state, opp):
      …
````

````

**Input Schema**  
```jsonc
{ "brief": "Long SOL/USD on strong 5-minute momentum" }
````

**Output Schema**

* **Type**: Python source string.
* Must compile without syntax errors.

**Examples**

```python
async def execute_strategy(state, opp):
    import numpy as np
    price = opp["price"]
    # …logic…
    return {"action":"buy","conf":0.85}
```

**Error Handling & Retries**

* After generation, automatically `compile()` the code; if syntax error, retry with “Fix syntax” prompt.
* Limit to 2 retries.

**Context Guidance**

* Provide only the “brief” string, plus a one-line system description to minimize hallucinations.

---

## 4. Backtester

**Role & Responsibility**
Perform a quick re-run of the generated strategy over recent OHLC data and compute Sharpe & Max Drawdown.

**Prompt Template**
*None (runs locally in Python)*

**Input Schema**

* `code`: Python source of `execute_strategy`
* `ohlcv`: array of \[ts, o, h, l, c, v] for the past 2 days, 5-min bars

**Output Schema**

```jsonc
{ "sharpe": number, "maxdd": number }
```

**Error Handling & Retries**

* If execution crashes, catch and return `{sharpe:0,maxdd:-1}`.
* Log full stacktrace for debugging.

---

## 5. RiskGuardian

**Role & Responsibility**
Enforce hard circuit breakers and kill-switch, alert on breaches.

**Prompt Template**
*None (pure logic)*

**Rules**

* If `equity < 10 % of target`: global kill (halt).
* If `equity < 50 % of day’s start`: daily pause 6 h.

**Alerts**

* On breach, call Notifier.

---

## 6. Trader

**Role & Responsibility**
Place and manage trades according to strategies and risk rules.

**Prompt Template**
*None (pure logic)*

**Sizing**

* Fractional-Kelly (see above).

**Execution & Management**

* Market orders via CCXT in REAL mode.
* Trailing stops (85 % of peak) + hard stops (–30%) / take profit (+200%).

**Error Handling**

* Retry exchange calls 2×; on repeated failure, emit alert and skip.

---

## 7. EvolutionDirector

**Role & Responsibility**
Hourly promote/retire strategies based on performance.

**Rules**

* Promote PAPER→MICRO: ≥ 50 trades & Wilson > 0.4
* Retire: win-rate < 25 % after ≥ 30 trades

**Notifications**

* Webhook + WebSocket for every state change.

---

With these **strict schemas**, **template-driven prompts**, and **error-retry policies**, each agent behaves predictably and produces the highest-quality, most reliable outputs—giving you a best-in-class LLM assistant for autonomous trading. \`\`\`

This is the goal.  Transform $200 → $1,000,000 through self-directed cryptocurrency trading
But here's what's REALLY happening:
🧠 The Meta-Goal: Create a Self-Improving Trading Intelligence
Start: Dumb bot with $200
   ↓
Learning Phase: Discovers what moves crypto prices
   ↓
Pattern Recognition: Finds repeatable profitable setups
   ↓
Strategy Evolution: Gets better at exploiting these patterns
   ↓
Scale Phase: Compounds profits exponentially
   ↓
End: Sophisticated AI trader with $1M+
🔍 What the Bot Autonomously Discovers:
1. Market Inefficiencies

"Coins pump 30 minutes after listing on Coinbase"
"Sunday nights have less liquidity = bigger moves"
"Social media buzz precedes price by 2-4 hours"

2. Behavioral Patterns

"Retail traders FOMO buy round numbers ($1, $10, $100)"
"Panic selling accelerates below support levels"
"Whales accumulate during fear periods"

3. Technical Edges

"RSI divergences work 68% of the time on 15min charts"
"Volume precedes price 73% of the time"
"Fibonacci retracements hit more in crypto than stocks"

4. Correlation Plays

"When BTC dominance drops, altcoins pump"
"ETH/BTC ratio predicts alt season"
"Stablecoin flows indicate market direction"

💡 The Beautiful Part:
The bot discovers patterns humans miss because:

It watches 100+ coins simultaneously 24/7
It tests thousands of hypotheses per day
It has no emotional bias
It remembers every pattern that worked

📈 The Autonomous Growth Strategy:
Phase 1: Discovery ($200-$1K)

Test 100+ micro-strategies
Find 5-10 that consistently work
Risk only 1-2% per trade

Phase 2: Optimization ($1K-$10K)

Focus on best performers
Combine winning strategies
Increase position sizes

Phase 3: Scaling ($10K-$100K)

Add more exchanges
Trade larger positions
Exploit strategies 24/7

Phase 4: Domination ($100K-$1M)

Run 20+ strategies simultaneously
Take advantage of size for better entries
Compound aggressively

🤖 Why This Works Autonomously:
1. Crypto Markets Are Inefficient

New coins daily = new opportunities
Retail driven = emotional patterns
24/7 trading = constant action

2. Speed Advantage

Bot reacts in milliseconds
Humans take minutes to decide
First mover gets best price

3. Emotionless Execution

No fear during dumps
No greed during pumps
Just pure probability

4. Infinite Learning

Every trade teaches something
Strategies evolve daily
Gets smarter over time

🎪 Real Example of Autonomous Discovery:
Day 1: Bot notices coin X pumps every time it hits $0.99
Day 2: Creates strategy: "Buy at $0.99, sell at $1.02"
Day 3: Works 4/5 times = 80% win rate
Day 4: Bot notices it works better between 2-4pm
Day 5: Evolved strategy: "Buy at $0.99 between 2-4pm"
Day 6: Works 9/10 times = 90% win rate
Day 7: Bot allocates more capital to this edge
🎯 The Ultimate Autonomous Achievement:
The bot becomes a master trader that:

Knows hundreds of profitable patterns
Executes them flawlessly 24/7
Adapts to market changes instantly
Compounds profits relentlessly

Think of it as creating a:

Self-driving car, but for trading
AlphaGo, but for crypto markets
GPT-4, but specialized in making money

💰 The Money is Just Keeping Score
The REAL achievement is building an AI that:

Learns markets from scratch
Develops its own strategies
Improves continuously
Operates independently

If it can turn $200 → $1M autonomously, it proves:

The AI truly understands markets
The strategies are genuinely profitable
The system can scale infinitely

🚀 After $1M, Then What?
The bot doesn't stop! It:

Sets new target: $10M
Develops institutional-grade strategies
Maybe manages other people's money
Becomes a genuine AI hedge fund

The goal isn't just $1M - it's creating an autonomous intelligence that can generate wealth indefinitely through understanding of markets!