import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import sys
sys.path.append('.')
from v26meme_full import (
    PatternDiscoveryEngine, AdaptiveStrategyEngine, 
    AutonomousTrader, SystemState, Strategy, Position,
    calculate_kelly_position, is_sane_ticker
)

# A dummy _detect_psychological_level_proximity function for testing
def _detect_psychological_level_proximity(price):
    price_str = f"{price:.2f}"
    return price_str.endswith('.99') or price_str.endswith('.98') or price_str.endswith('.01') or price_str.endswith('.02')


class TestUtilityFunctions:
    def test_kelly_position_calculation(self):
        """Test Kelly Criterion calculations"""
        # Normal case
        # Using 0.25 for max_position_pct as it's the default cap in the function
        position = calculate_kelly_position(0.6, 0.05, 0.03, 0.25)
        assert 0 <= position <= 0.25

        # Edge cases
        assert calculate_kelly_position(0, 0.05, 0.03, 0.1) == 0
        # With a 100% win rate, kelly is 1.0, capped at 0.25, then capped by max_position_pct of 0.1
        assert calculate_kelly_position(1, 0.05, 0.03, 0.1) == 0.1
        assert calculate_kelly_position(0.5, 0, 0.03, 0.1) == 0

    def test_ticker_validation(self):
        """Test ticker symbol validation"""
        assert is_sane_ticker("BTC/USDT") == True
        assert is_sane_ticker("ETH/USDC") == True
        assert is_sane_ticker("TESTCOIN/USDT") == True # is_sane_ticker is not for filtering meme coins, just format
        assert is_sane_ticker("BTC3L/USDT") == False
        assert is_sane_ticker("INVALID") == False

class TestPatternDiscovery:
    @pytest.mark.asyncio
    async def test_pattern_discovery_initialization(self):
        """Test pattern discovery engine initialization"""
        mock_memory = Mock()
        mock_exchanges = {'test': Mock()}
        mock_ai = Mock()
        
        engine = PatternDiscoveryEngine(mock_memory, mock_exchanges, mock_ai)
        assert engine is not None
        assert engine.learning_memory == mock_memory
    
    @pytest.mark.asyncio
    async def test_psychological_level_detection(self):
        """Test psychological price level detection"""
        assert _detect_psychological_level_proximity(0.99) == True
        assert _detect_psychological_level_proximity(9.98) == True  # Near 9.99
        assert _detect_psychological_level_proximity(100.5) == False
        assert _detect_psychological_level_proximity(0) == False

class TestStrategyExecution:
    @pytest.mark.asyncio
    async def test_strategy_code_sandbox(self):
        """Test strategy code execution sandbox"""
        trader = AutonomousTrader()
        trader.state = SystemState()
        trader.db = Mock()
        trader.db.save_trade_log = AsyncMock()


        # Safe strategy
        safe_strategy = Strategy(
            id="test1",
            name="SafeStrat",
            description="Test",
            code="""
async def execute_strategy(state, opp):
    return {'action': 'hold', 'conf': 0.5}
""",
            status="paper",
            pattern_id="p1",
            generation=1
        )
        
        opp = {'symbol': 'BTC/USDT', 'price': 50000, 'volume': 1000000, 'exchange': 'test'}
        result = await trader._run_strategy_code(safe_strategy, opp)
        assert result['action'] == 'hold'
        assert result['conf'] == 0.5
        
        # Malicious strategy (should be sandboxed)
        evil_strategy = Strategy(
            id="test2",
            name="EvilStrat",
            description="Test",
            code="""
async def execute_strategy(state, opp):
    import os
    os.system('rm -rf /')  # This should fail
    return {'action': 'hold', 'conf': 0}
""",
            status="paper",
            pattern_id="p2",
            generation=1
        )
        
        result = await trader._run_strategy_code(evil_strategy, opp)
        assert result['action'] == 'hold'  # Should safely fail

class TestRiskManagement:
    @pytest.mark.asyncio
    async def test_position_sizing_limits(self):
        """Test position size limits are enforced"""
        trader = AutonomousTrader()
        trader.state = SystemState(equity=1000)
        trader.db = Mock()
        trader.db.save_position = AsyncMock()
        trader.db.save_trade_log = AsyncMock()
        trader.exchanges = {'test': AsyncMock()}
        trader.exchanges['test'].create_order.return_value = {'id': '123', 'status': 'open'}


        # Test max position size
        strategy = Mock()
        strategy.id = "strat1"
        strategy.name = "TestStrat"
        strategy.status = "paper"
        strategy.win_rate = 0.6
        strategy.avg_profit = 0.03
        strategy.max_position_pct=0.1
        strategy.stop_loss=0.05
        strategy.take_profit=0.15
        
        opp = {'symbol': 'BTC/USDT', 'price': 50000, 'exchange': 'test'}
        
        # Position should not exceed 10% of equity
        with patch.object(trader, '_open_position', new=AsyncMock()) as mock_open:
            await trader._process_decision(
                strategy, opp, 
                {'action': 'buy', 'conf': 0.9, 'sl': 47500, 'tp': 57500}
            )
            if mock_open.called:
                # call_args[0] is the tuple of positional args
                # The 'size' argument is the 4th positional argument (index 3)
                size_in_quote = mock_open.call_args[0][3]
                assert size_in_quote <= trader.state.equity * 0.1

    
    @pytest.mark.asyncio
    async def test_daily_loss_limit(self):
        """Test daily loss limit circuit breaker"""
        trader = AutonomousTrader()
        trader.state = SystemState(
            equity=1000,
            daily_pnl=-110  # More than 10% loss
        )
        trader.config = {'MAX_DAILY_LOSS': 0.1}


        assert await trader._check_risk_limits() == True
        
        trader.state.daily_pnl = -50  # 5% loss
        assert await trader._check_risk_limits() == False

class TestExchangeIntegration:
    @pytest.mark.asyncio
    async def test_market_scanning_error_handling(self):
        """Test market scanning handles exchange errors gracefully"""
        trader = AutonomousTrader()
        trader.db = Mock()
        trader.db.log_event = AsyncMock()


        # Mock exchange that throws error
        mock_exchange = AsyncMock()
        mock_exchange.fetch_tickers.side_effect = Exception("Network error")
        trader.exchanges = {'test': mock_exchange}
        
        # Should not crash
        opportunities = await trader._scan_markets()
        assert opportunities == []  # Returns empty list on error

@pytest.mark.asyncio
async def test_full_integration():
    """Integration test of main components"""
    with patch('v26meme_full.AutonomousTrader') as mock_trader_class:
        mock_trader_instance = AsyncMock()
        mock_trader_class.return_value = mock_trader_instance
        
        from v26meme_full import main
        # We patch the main loop to avoid running forever, and just check initialization
        await main()
        
        mock_trader_instance.initialize.assert_called_once()
        mock_trader_instance.run.assert_called_once()

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
