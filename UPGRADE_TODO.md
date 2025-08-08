# v26meme Upgrade Checklist

Status: In progress

- [x] Env wiring in `Config`: read INITIAL_CAPITAL, TARGET_CAPITAL, TARGET_DAYS
- [x] Map env: `MAX_POSITION_PCT` → `Config.MAX_POSITION_SIZE`, `MAX_DAILY_DRAWDOWN` → `Config.MAX_DAILY_LOSS`, `FRACTIONAL_KELLY_CAP` → `Config.KELLY_FRACTION`, `MIN_ORDER_NOTIONAL` → `Config.MIN_TRADE_SIZE`, `LOG_LEVEL`
- [x] Add `FEE_RATE` to `Config`
- [x] Add feature flags: `FEED_LISTING`, `FEED_TWITTER`, `FEED_WHALE`, `FEED_WEEKEND`, `GENETIC_V2`, `RISK_GUARDIAN`
- [x] Add `MODE` default from env

- [x] Kelly function: cap by provided `max_position_pct`; use `Config.KELLY_FRACTION` by default
- [x] `_process_decision`: clamp by min(Config.MAX_POSITION_SIZE, strategy.max_position_pct)
- [x] `_process_decision`: forward decision `sl`/`tp` to `_open_position`
- [x] `_open_position`: enforce cash and caps; prefer decision sl/tp

- [x] Daily risk based on start-of-day equity; track `equity_start_of_day`
- [x] Remove unused `peak_equity` var; tighten `_check_risk_limits`

- [x] DB init: stop dropping tables; add `current_positions` table

- [x] Save cadence: swap modulo time for timestamp delta (~30s)

- [x] Parallelize `_scan_markets` across exchanges with gather

- [x] Harden sandbox: remove `__import__`, reject symbol-specific logic

- [x] start_system.sh: guard dashboard start if file missing; TRADING_MODE passthrough

- [x] README: CLI `--mode`, env overrides section

- [ ] Strategy lifecycle: auto-promote/demote (PAPER→MICRO→ACTIVE) with thresholds
- [ ] CLI: argparse for `--mode` and `--install`; set `SystemState.mode` via CLI/env
- [ ] Fees applied on open/close P&L
- [ ] Backfill dashboard file or note it as optional target

After these remaining items, re-run tests and a PAPER dry run for 5-10 minutes to validate loops, sizing, and risk guards.


## Post-bootstrap hardening (promotion thresholds)

- [ ] Create branch `hardening/promotion-thresholds`
- [ ] Revert bootstrap env defaults to strict mode (disable BOOTSTRAP_PROMOTION by default)
  - [ ] Set `BOOTSTRAP_PROMOTION=false`
  - [ ] Restore PAPER→MICRO defaults: min trades 20, Wilson ≥ 0.52, Sharpe ≥ 0.5 (Discovery)
  - [ ] Restore MICRO→ACTIVE defaults: min trades 100, win rate ≥ 0.58, Sharpe ≥ 1.2, PnL > 0
- [ ] Make thresholds configurable only via `.env` in production, not baked-in runtime overrides
- [ ] Add unit tests for promotion/demotion boundaries
- [ ] Update README with recommended production thresholds
- [ ] Open PR and merge after 24-48h observation window

