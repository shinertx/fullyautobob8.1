import pytest
import os
import sqlite3
import sys
from unittest.mock import patch
from v26meme_full import Config, is_sane_ticker, calculate_kelly_position, Database, extract_params, inject_params

@patch.dict(os.environ, {"INITIAL_CAPITAL": "200.0", "TARGET_CAPITAL": "1000000.0", "TARGET_DAYS": "90"}, clear=True)
def test_config_initialization():
    # We need to reload the module to make sure the new env vars are picked up
    import importlib
    # It's better to reload the specific module where Config is defined
    if 'v26meme_full' in sys.modules:
        importlib.reload(sys.modules['v26meme_full'])

    assert Config.INITIAL_CAPITAL == 200.0
    assert Config.TARGET_CAPITAL == 1_000_000.0
    assert Config.TARGET_DAYS == 90
    assert Config.DB_PATH == "v26meme.db"

def test_is_sane_ticker():
    assert is_sane_ticker("BTC/USDT")
    assert is_sane_ticker("ETH/USD")
    assert is_sane_ticker("PEPE/USDC")
    assert not is_sane_ticker("BTCUP/USDT")
    assert not is_sane_ticker("ETHDOWN/USDT")
    assert not is_sane_ticker("INVALID")
    assert not is_sane_ticker("BTC/FOO")

def test_calculate_kelly_position():
    # Test case 1: Profitable scenario
    assert calculate_kelly_position(win_rate=0.6, avg_win=0.1, avg_loss=0.05) == pytest.approx(0.1) # (0.6 * 2 - 0.4) / 2 * 0.25
    # Test case 2: Losing scenario
    assert calculate_kelly_position(win_rate=0.4, avg_win=0.1, avg_loss=0.05) == 0.0
    # Test case 3: High win rate
    assert calculate_kelly_position(win_rate=0.9, avg_win=0.2, avg_loss=0.1) > 0
    # Test case 4: Edge cases
    assert calculate_kelly_position(win_rate=0.5, avg_win=0.1, avg_loss=0.1) == 0.0
    assert calculate_kelly_position(win_rate=0, avg_win=0.1, avg_loss=0.1) == 0.0
    assert calculate_kelly_position(win_rate=1, avg_win=0.1, avg_loss=0.1) == pytest.approx(0.25) # Capped at max

def test_database_initialization(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path=str(db_path))
    assert os.path.exists(db_path)
    # Check if tables are created
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "patterns" in tables
    assert "strategies" in tables
    assert "trades" in tables
    assert "system_state" in tables
    conn.close()

def test_param_extraction_and_injection():
    sample_code = """
# Strategy to test params
param1 = 123
some_other_code = "hello"
my_param_2 = 0.5  # A comment
another_var=42.0
    """
    params = extract_params(sample_code)
    assert params == {"param1": 123.0, "my_param_2": 0.5, "another_var": 42.0}

    new_params = {"param1": 456, "my_param_2": 0.75}
    new_code = inject_params(sample_code, new_params)

    assert "param1 = 456" in new_code
    assert "my_param_2 = 0.75" in new_code
    assert "another_var=42.0" in new_code
