# v26meme Copilot Instructions

## Project Goal
Build an autonomous AI that discovers crypto market patterns and evolves profitable strategies to turn $200 → $1M through self-directed learning.

## Core Philosophy
The bot must discover patterns autonomously, not execute pre-programmed strategies. Every strategy should be born from observed market inefficiencies.

## When Modifying Code

### Pattern Discovery Engine
When working in `PatternDiscoveryEngine` methods:
```python
# Good: Discovers measurable, exploitable patterns
pattern = {
    'type': 'psychological_pump',
    'symbol': 'DOGE/USDT', 
    'level': 0.99,
    'confidence': 0.85,  # Based on 100+ observations
    'avg_pump': 0.03,    # 3% average move
    'time_window': '2-4pm UTC'
}

# Bad: Vague patterns without actionable edge
pattern = {'type': 'bullish', 'reason': 'looks good'}
```

### Strategy Generation
When enhancing `_create_prompt_from_pattern`:
- Include specific entry/exit conditions
- Reference the pattern's historical performance
- Request risk management rules (stops, targets)
- Ask for confidence scoring logic

### Evolution Methods
When improving `AdaptiveStrategyEngine`:
```python
# Mutation should be aggressive in early phases
if phase == "Discovery":
    mutation_rate = 0.20  # 20% parameter change
elif phase == "Domination":
    mutation_rate = 0.05  # 5% fine-tuning
```

### Risk Management
When updating position sizing or risk checks:
- Always respect Fractional Kelly (0.25 cap)
- Scale risk with equity growth phases
- Never risk more than 2% in Discovery phase

## Code Style Guidelines

1. **Async First**: All I/O operations should be async
2. **Type Hints**: Use them for clarity
3. **Logging**: Use emojis for important events (🔍 🧬 💰 ⚠️)
4. **Error Handling**: Never crash the main loop
5. **Performance**: Cache expensive calculations

## Key Patterns to Discover

Focus on implementing discovery for:
- **Psychological levels** ($0.99, $9.99, $99)
- **Time-based anomalies** (specific hours/days)
- **Volume-price divergences** 
- **Correlation patterns** (BTC dominance vs alts)
- **Event reactions** (listings, news)

## Strategy Validation

Every generated strategy must:
```python
async def execute_strategy(state, opp):
    # 1. Check entry conditions
    # 2. Calculate position size
    # 3. Return decision with confidence
    return {"action": "buy", "conf": 0.85}
```

## Performance Tracking

Track these metrics for every strategy:
- Win rate (with Wilson score after 50 trades)
- Sharpe ratio (risk-adjusted returns)
- Max drawdown (worst peak-to-trough)
- Average profit per trade
- Time in market

## Don't:
- Create new files without user request
- Add dependencies without checking
- Modify database schema
- Change core risk parameters
- Remove logging statements

## Do:
- Enhance pattern discovery algorithms
- Improve strategy generation prompts
- Add better validation checks
- Optimize existing methods
- Add helpful comments

## Remember
We're building a self-improving intelligence that:
- Discovers patterns humans miss
- Tests thousands of hypotheses daily
- Has no emotional bias
- Gets smarter every hour

The money is just keeping score. The real achievement is creating an AI that understands markets autonomously.