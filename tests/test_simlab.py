import pytest
import os
import pandas as pd
from simlab import SimLab

# Mock strategy and opportunity for testing
class MockStrategy:
    def __init__(self, id, name):
        self.id = id
        self.name = name

@pytest.fixture
def mock_simlab(tmp_path):
    """Fixture to create a SimLab instance with a temporary database."""
    db_path = tmp_path / "simlab_test.db"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create a dummy data file
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 12:00', '2023-01-01 12:05', '2023-01-01 12:10']),
        'open': [100, 101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [101, 102, 101],
        'volume': [10, 12, 15]
    })
    df.to_csv(data_dir / "BTC_USDT_5m.csv", index=False)

    async def mock_run_strategy(strategy, opp):
        if opp['price'] > 101:
            return {"action": "buy", "conf": 0.8}
        return {"action": "hold"}

    def mock_fetch_strategies():
        return {"strat1": MockStrategy("strat1", "Test Strategy 1")}

    # Patch environment variables for SimLab
    os.environ["SIMLAB_DATA_DIR"] = str(data_dir)
    os.environ["SIMLAB_ENABLED"] = "true"

    simlab_instance = SimLab(
        db_path=str(db_path),
        run_strategy_callable=mock_run_strategy,
        fetch_strategies_callable=mock_fetch_strategies,
    )
    return simlab_instance

@pytest.mark.asyncio
async def test_simlab_initialization(mock_simlab):
    assert mock_simlab is not None
    assert os.path.exists(mock_simlab.db_path)

@pytest.mark.asyncio
async def test_simlab_autodiscover_and_run(mock_simlab):
    # This test will check if SimLab can discover and run a simulation
    # from a data file.
    
    # Manually trigger autodiscovery
    await mock_simlab._autodiscover_loop()
    
    # Get the run from the queue
    run = await mock_simlab._queue.get()
    
    assert run.symbol == "BTC/USDT"
    assert run.timeframe == "5m"
    
    # Execute the run
    await mock_simlab._execute_run(run)
    
    # Check if results are stored in the database
    conn = pd.read_sql_query("SELECT * FROM sim_metrics", f"sqlite:///{mock_simlab.db_path}")
    assert len(conn) > 0
    assert conn.iloc[0]['strategy_id'] == 'strat1'
    assert conn.iloc[0]['trades'] > 0
