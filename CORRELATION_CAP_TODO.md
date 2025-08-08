# Correlation Cap & Limits TODO

## Current Status
- ✅ **FIXED**: Increased correlation cap from 5 to 20 positions per quote currency
- ✅ **FIXED**: Closed all existing positions to start fresh

## Limits to Review & Potentially Adjust

### 1. Correlation Caps
- **Quote Currency Cap**: Currently 20 positions per quote (USD, EUR, USDT, etc.)
- **Base Currency Cap**: Currently 3 positions per base currency (BTC, ETH, etc.)
- **Status**: ✅ Increased quote cap to 20

### 2. Position Size Limits
- **MAX_POSITION_SIZE**: 10% of equity per position
- **MIN_TRADE_SIZE_PAPER**: $10 minimum trade size
- **PROBE_SIZE_PAPER**: $15 probe size
- **Status**: ⏳ Review if these are appropriate for aggressive trading

### 3. Volume & Spread Filters
- **MIN_VOLUME_USD_PAPER**: $5,000 minimum volume
- **MAX_SPREAD_BPS**: 75 bps (0.75%) maximum spread
- **Status**: ⏳ Consider relaxing for more opportunities

### 4. Confidence Thresholds
- **MIN_CONF_PAPER**: 5% minimum confidence
- **Status**: ⏳ Consider lowering for more exploration

### 5. Risk Management
- **MAX_DAILY_LOSS**: 10% daily drawdown limit
- **MAX_DRAWDOWN**: 20% total drawdown limit
- **Status**: ⏳ Review for aggressive 90-day sprint

## Next Steps
1. **Monitor trading activity** with new 20-position cap
2. **Review position size limits** if bot is still too conservative
3. **Consider relaxing volume/spread filters** for more opportunities
4. **Adjust confidence thresholds** if needed for more exploration

## Notes
- The bot was working correctly but was being too conservative
- Correlation cap was the main blocker (5 positions per quote)
- Increased to 20 positions per quote for more aggressive trading
- All positions closed to start fresh
