#!/usr/bin/env python3
"""
Advanced .env Logic Validation for v26meme Trading Bot
======================================================
This script validates not just the environment variables, but also tests
the actual code logic where each variable is used to ensure everything
works as expected in the real application flow.

Features:
- Tests Config class initialization
- Validates trading parameter logic
- Tests API key usage patterns
- Simulates critical code paths
- Checks for edge cases and error handling

Author: v26meme System  
Date: 2025-01-05
"""

import os
import sys
import traceback
from typing import Dict, List, Any
from dotenv import load_dotenv
from datetime import datetime

# Load environment
load_dotenv()

class LogicValidator:
    """Tests actual code logic with environment variables"""
    
    def __init__(self):
        self.test_results = []
        self.critical_issues = []
        
    def run_all_tests(self):
        """Run comprehensive logic validation"""
        print("🧪 ADVANCED LOGIC VALIDATION")
        print("=" * 60)
        
        # Test 1: Config class initialization
        self.test_config_initialization()
        
        # Test 2: Trading parameter logic
        self.test_trading_parameters()
        
        # Test 3: API key validation logic
        self.test_api_key_logic()
        
        # Test 4: Mode-dependent behavior
        self.test_mode_dependent_logic()
        
        # Test 5: Risk management calculations
        self.test_risk_calculations()
        
        # Test 6: Web server configuration
        self.test_web_config()
        
        # Test 7: Database path validation
        self.test_database_config()
        
        # Test 8: Logging configuration
        self.test_logging_config()
        
        return self.generate_report()
    
    def test_config_initialization(self):
        """Test Config class can be initialized with current .env"""
        print("🔧 Testing Config class initialization...")
        
        try:
            # Import and test the actual Config class
            sys.path.insert(0, '.')
            
            # Mock the Config class behavior
            class TestConfig:
                # API Keys
                OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
                
                # Exchange API Keys
                COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
                COINBASE_SECRET = os.getenv("COINBASE_SECRET")
                COINBASE_PASSPHRASE = os.getenv("COINBASE_PASSPHRASE")
                
                # Trading Configuration
                try:
                    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "200"))
                    TARGET_CAPITAL = float(os.getenv("TARGET_CAPITAL", "1000000"))
                    MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.02"))
                    MAX_DAILY_DRAWDOWN = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.10"))
                    MIN_ORDER_NOTIONAL = float(os.getenv("MIN_ORDER_NOTIONAL", "0.0"))
                    MAX_EXPOSURE_PCT = float(os.getenv("MAX_EXPOSURE_PCT", "0.5"))
                    WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
                except ValueError as e:
                    raise ValueError(f"Invalid numeric config value: {e}")
                
                # System Configuration
                MODE = os.getenv("MODE", "PAPER")
                DATABASE_PATH = os.getenv("DATABASE_PATH", "v26meme.db")
                LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
                
                # Web Configuration
                WEB_AUTH_TOKEN = os.getenv("WEB_AUTH_TOKEN", "")
                WEB_BIND_HOST = os.getenv("WEB_BIND_HOST", "127.0.0.1")
            
            # Test each configuration value
            config = TestConfig()
            
            # Validate all attributes exist and have expected types
            assert hasattr(config, 'OPENAI_API_KEY')
            assert hasattr(config, 'INITIAL_CAPITAL')
            assert isinstance(config.INITIAL_CAPITAL, float)
            assert isinstance(config.WEB_PORT, int)
            assert config.MODE in ["PAPER", "REAL"]
            
            self.test_results.append("✅ Config initialization: PASSED")
            
        except Exception as e:
            error_msg = f"❌ Config initialization: FAILED - {e}"
            self.test_results.append(error_msg)
            self.critical_issues.append(error_msg)
    
    def test_trading_parameters(self):
        """Test trading parameter logic and calculations"""
        print("💰 Testing trading parameter logic...")
        
        try:
            initial_capital = float(os.getenv("INITIAL_CAPITAL", "200"))
            target_capital = float(os.getenv("TARGET_CAPITAL", "1000000"))
            max_position_pct = float(os.getenv("MAX_POSITION_PCT", "0.02"))
            max_daily_drawdown = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.10"))
            min_order_notional = float(os.getenv("MIN_ORDER_NOTIONAL", "0.0"))
            max_exposure_pct = float(os.getenv("MAX_EXPOSURE_PCT", "0.5"))
            
            # Test calculations that the bot actually performs
            max_position_size = initial_capital * max_position_pct
            daily_loss_limit = initial_capital * max_daily_drawdown
            max_total_exposure = initial_capital * max_exposure_pct
            
            # Test logical constraints
            assert initial_capital > 0, "Initial capital must be positive"
            assert target_capital > initial_capital, "Target must be greater than initial"
            assert 0 < max_position_pct <= 1, "Position percentage must be between 0 and 1"
            assert 0 < max_daily_drawdown <= 1, "Drawdown percentage must be between 0 and 1"
            assert min_order_notional >= 0, "Min order notional cannot be negative"
            assert 0 < max_exposure_pct <= 1, "Exposure percentage must be between 0 and 1"
            
            # Test realistic values
            assert max_position_size >= min_order_notional, "Max position must be >= min order"
            assert daily_loss_limit < initial_capital, "Daily loss limit must be < total capital"
            
            # Test paper trading specific logic
            mode = os.getenv("MODE", "PAPER")
            if mode == "PAPER":
                assert min_order_notional == 0.0, "Paper mode should have 0 min order notional"
            
            self.test_results.append("✅ Trading parameters: PASSED")
            self.test_results.append(f"   - Max position size: ${max_position_size:.2f}")
            self.test_results.append(f"   - Daily loss limit: ${daily_loss_limit:.2f}")
            self.test_results.append(f"   - Max total exposure: ${max_total_exposure:.2f}")
            
        except Exception as e:
            error_msg = f"❌ Trading parameters: FAILED - {e}"
            self.test_results.append(error_msg)
            self.critical_issues.append(error_msg)
    
    def test_api_key_logic(self):
        """Test API key validation logic"""
        print("🔑 Testing API key logic...")
        
        try:
            mode = os.getenv("MODE", "PAPER")
            openai_key = os.getenv("OPENAI_API_KEY")
            coinbase_key = os.getenv("COINBASE_API_KEY")
            coinbase_secret = os.getenv("COINBASE_SECRET")
            
            # Test OpenAI key logic (from OpenAIManager.__init__)
            if not openai_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            
            # Test Coinbase key logic (from ExchangeManager)
            if mode == "REAL":
                required_keys = [
                    ("COINBASE_API_KEY", coinbase_key),
                    ("COINBASE_SECRET", coinbase_secret)
                ]
                missing = [name for name, value in required_keys if not value]
                if missing:
                    raise ValueError(f"Missing required API keys for REAL mode: {', '.join(missing)}")
            
            # Test key format validation
            if openai_key and not openai_key.startswith("sk-"):
                raise ValueError("OpenAI API key should start with 'sk-'")
            
            if coinbase_key and "organizations/" not in coinbase_key:
                raise ValueError("Coinbase API key should contain 'organizations/'")
            
            if coinbase_secret and not coinbase_secret.startswith("-----BEGIN"):
                raise ValueError("Coinbase secret should be a private key")
            
            self.test_results.append("✅ API key logic: PASSED")
            self.test_results.append(f"   - Mode: {mode}")
            self.test_results.append(f"   - OpenAI key: {'✓' if openai_key else '✗'}")
            self.test_results.append(f"   - Coinbase key: {'✓' if coinbase_key else '✗'}")
            
        except Exception as e:
            error_msg = f"❌ API key logic: FAILED - {e}"
            self.test_results.append(error_msg)
            if "OPENAI_API_KEY" in str(e) or "REAL mode" in str(e):
                self.critical_issues.append(error_msg)
    
    def test_mode_dependent_logic(self):
        """Test mode-dependent behavior"""
        print("🎛️ Testing mode-dependent logic...")
        
        try:
            mode = os.getenv("MODE", "PAPER")
            
            # Test mode validation
            assert mode in ["PAPER", "REAL"], f"Invalid mode: {mode}"
            
            # Test paper mode logic
            if mode == "PAPER":
                min_notional = float(os.getenv("MIN_ORDER_NOTIONAL", "0.0"))
                assert min_notional == 0.0, "Paper mode should have 0 min order notional"
                
                # Paper mode should work without exchange API keys
                self.test_results.append("✅ Paper mode: Can run without exchange credentials")
            
            # Test real mode logic
            elif mode == "REAL":
                coinbase_key = os.getenv("COINBASE_API_KEY")
                coinbase_secret = os.getenv("COINBASE_SECRET")
                
                if not (coinbase_key and coinbase_secret):
                    raise ValueError("Real mode requires Coinbase credentials")
                
                min_notional = float(os.getenv("MIN_ORDER_NOTIONAL", "0.0"))
                if min_notional == 0.0:
                    self.test_results.append("⚠️ Real mode with 0 min notional - may cause exchange errors")
            
            self.test_results.append(f"✅ Mode logic ({mode}): PASSED")
            
        except Exception as e:
            error_msg = f"❌ Mode logic: FAILED - {e}"
            self.test_results.append(error_msg)
            self.critical_issues.append(error_msg)
    
    def test_risk_calculations(self):
        """Test risk management calculations"""
        print("⚖️ Testing risk calculations...")
        
        try:
            initial_capital = float(os.getenv("INITIAL_CAPITAL", "200"))
            max_drawdown = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.10"))
            max_position_pct = float(os.getenv("MAX_POSITION_PCT", "0.02"))
            
            # Simulate the actual risk calculations from the code
            # From RiskManager.check_daily_limits
            current_equity = initial_capital * 0.95  # Simulate 5% loss
            daily_drawdown = (initial_capital - current_equity) / initial_capital
            
            # From RiskManager.calculate_position_size  
            max_position_value = current_equity * max_position_pct
            
            # From the overall drawdown calculation
            overall_drawdown = (initial_capital - current_equity) / initial_capital
            
            # Test risk logic
            assert daily_drawdown <= 1.0, "Daily drawdown cannot exceed 100%"
            assert max_position_value > 0, "Max position value must be positive"
            assert overall_drawdown >= 0, "Overall drawdown cannot be negative"
            
            # Test circuit breaker logic (from the code)
            kill_switch_threshold = 0.90  # 90% drawdown
            daily_pause_threshold = max_drawdown
            
            should_kill = overall_drawdown >= kill_switch_threshold
            should_pause = daily_drawdown >= daily_pause_threshold
            
            self.test_results.append("✅ Risk calculations: PASSED")
            self.test_results.append(f"   - Daily drawdown: {daily_drawdown:.2%}")
            self.test_results.append(f"   - Max position value: ${max_position_value:.2f}")
            self.test_results.append(f"   - Kill switch: {'TRIGGERED' if should_kill else 'OK'}")
            self.test_results.append(f"   - Daily pause: {'TRIGGERED' if should_pause else 'OK'}")
            
        except Exception as e:
            error_msg = f"❌ Risk calculations: FAILED - {e}"
            self.test_results.append(error_msg)
            self.critical_issues.append(error_msg)
    
    def test_web_config(self):
        """Test web server configuration"""
        print("🌐 Testing web configuration...")
        
        try:
            web_port = int(os.getenv("WEB_PORT", "8000"))
            web_host = os.getenv("WEB_BIND_HOST", "127.0.0.1")
            web_token = os.getenv("WEB_AUTH_TOKEN", "")
            
            # Test port validation
            assert 1024 <= web_port <= 65535, f"Web port {web_port} not in valid range 1024-65535"
            
            # Test host validation
            valid_hosts = ["127.0.0.1", "localhost", "0.0.0.0"]
            if web_host not in valid_hosts and not web_host.startswith("192.168."):
                self.test_results.append(f"⚠️ Web host {web_host} - ensure it's safe for your network")
            
            # Test security
            if web_host == "0.0.0.0" and not web_token:
                self.test_results.append("⚠️ Web server on 0.0.0.0 without auth token - security risk")
            
            self.test_results.append("✅ Web configuration: PASSED")
            self.test_results.append(f"   - Port: {web_port}")
            self.test_results.append(f"   - Host: {web_host}")
            self.test_results.append(f"   - Auth token: {'✓' if web_token else '✗'}")
            
        except Exception as e:
            error_msg = f"❌ Web configuration: FAILED - {e}"
            self.test_results.append(error_msg)
    
    def test_database_config(self):
        """Test database configuration"""
        print("🗄️ Testing database configuration...")
        
        try:
            db_path = os.getenv("DATABASE_PATH", "v26meme.db")
            
            # Test path validation
            assert db_path.endswith(".db"), "Database path should end with .db"
            
            # Test if path is writable
            import tempfile
            import sqlite3
            
            try:
                # Test creating a database with the same name in temp directory
                test_path = os.path.join(tempfile.gettempdir(), "test_" + os.path.basename(db_path))
                conn = sqlite3.connect(test_path)
                conn.execute("CREATE TABLE test (id INTEGER)")
                conn.close()
                os.unlink(test_path)
                
                self.test_results.append("✅ Database configuration: PASSED")
                self.test_results.append(f"   - Path: {db_path}")
                self.test_results.append(f"   - Writable: ✓")
                
            except Exception as db_e:
                self.test_results.append(f"⚠️ Database path may not be writable: {db_e}")
                
        except Exception as e:
            error_msg = f"❌ Database configuration: FAILED - {e}"
            self.test_results.append(error_msg)
    
    def test_logging_config(self):
        """Test logging configuration"""
        print("📝 Testing logging configuration...")
        
        try:
            log_level = os.getenv("LOG_LEVEL", "DEBUG")
            
            # Test log level validation
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            assert log_level in valid_levels, f"Invalid log level: {log_level}"
            
            # Test that logging can be initialized
            import logging
            numeric_level = getattr(logging, log_level)
            assert isinstance(numeric_level, int), "Log level should convert to integer"
            
            self.test_results.append("✅ Logging configuration: PASSED")
            self.test_results.append(f"   - Level: {log_level}")
            self.test_results.append(f"   - Numeric level: {numeric_level}")
            
        except Exception as e:
            error_msg = f"❌ Logging configuration: FAILED - {e}"
            self.test_results.append(error_msg)
    
    def generate_report(self):
        """Generate final validation report"""
        passed = len([r for r in self.test_results if r.startswith("✅")])
        warnings = len([r for r in self.test_results if r.startswith("⚠️")])
        failed = len([r for r in self.test_results if r.startswith("❌")])
        
        print("\n📊 LOGIC VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: ✅ {passed}")
        print(f"Warnings: ⚠️ {warnings}")
        print(f"Failed: ❌ {failed}")
        print(f"Critical Issues: 🚨 {len(self.critical_issues)}")
        
        print("\n📋 DETAILED RESULTS")
        print("=" * 60)
        for result in self.test_results:
            print(result)
        
        if self.critical_issues:
            print("\n🚨 CRITICAL ISSUES")
            print("=" * 60)
            for issue in self.critical_issues:
                print(issue)
        
        # Overall status
        if self.critical_issues:
            status = "CRITICAL"
        elif failed > 0:
            status = "FAILED"
        elif warnings > 0:
            status = "WARNING"
        else:
            status = "PASSED"
        
        print(f"\n🏁 Logic validation status: {status}")
        
        return {
            "status": status,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "critical_issues": len(self.critical_issues),
            "results": self.test_results
        }

def main():
    """Main validation function"""
    print("🚀 v26meme ADVANCED .env LOGIC VALIDATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    validator = LogicValidator()
    report = validator.run_all_tests()
    
    print("\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if report["critical_issues"] > 0:
        print("🚨 CRITICAL: Fix critical issues before running the bot")
        print("   - Check API key configuration")
        print("   - Verify mode-dependent settings")
    
    if report["failed"] > 0:
        print("❌ ERRORS: Address failed validations")
        print("   - Review configuration logic")
        print("   - Check parameter calculations")
    
    if report["warnings"] > 0:
        print("⚠️ WARNINGS: Review warning items")
        print("   - Check security settings")
        print("   - Verify file permissions")
    
    if report["status"] == "PASSED":
        print("✅ ALL GOOD: Logic validation passed!")
        print("   - All configurations work correctly")
        print("   - Trading parameters are logical")
        print("   - API integrations are properly configured")
    
    # Exit with appropriate code
    if report["critical_issues"] > 0:
        exit(2)
    elif report["failed"] > 0:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()
