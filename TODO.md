# v26meme Project TODO

This file tracks the development roadmap for enhancing the autonomous trading system.

## 🎯 High Priority (Edge & Performance)

-   [ ] **Implement Backtesting Framework**: Create a `backtester.py` to simulate strategy performance on historical data. This is critical for faster validation before risking capital.
-   [ ] **Add Event-Based Pattern Discovery**: Implement `_discover_event_patterns` to track exchange listings, protocol upgrades, and major announcements. These are major alpha sources in crypto.
-   [ ] **Add Social Sentiment Patterns**: Implement `_discover_social_patterns` to analyze Twitter/Reddit volume and sentiment spikes, which often precede price movements.

## 🟡 Medium Priority (Optimization & Risk)

-   [ ] **Implement Multi-Timeframe Analysis**: Enhance strategy generation to require confirmation across multiple timeframes (e.g., Daily trend + 4H structure + 1H entry) for higher conviction trades.
-   [ ] **Implement Portfolio Correlation Management**: Add a module to `AutonomousTrader` to check correlation between active strategies and prevent over-concentration on a single market factor.
-   [ ] **Track Slippage and Fees Accurately**: Modify the `_close_position` method to account for actual execution price vs. expected, and include trading fees for more precise P&L calculation.

## 🟢 Low Priority (Enhancements & Nice-to-Haves)

-   [ ] **Build Real-Time Web Dashboard**: Create a simple web interface (e.g., using FastAPI) to serve a `dashboard.html` for live monitoring of equity, positions, and strategy performance.
-   [ ] **Integrate News API**: Implement a news fetching service to feed into the `OpenAIManager` for real-time sentiment analysis.
-   [ ] **Add On-Chain Pattern Discovery**: Implement `_discover_onchain_patterns` to track whale movements and exchange flows from sources like Glassnode or Santiment.
-   [ ] **Diversify Order Types**: Add support for Limit and Stop-Limit orders in the `_open_position` and `_close_position` methods to reduce slippage and improve execution.
