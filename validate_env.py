#!/usr/bin/env python3
"""
Complete .env Validation Script for v26meme Trading Bot
=======================================================
This script validates ALL environment variables used in the codebase:
- Checks if required variables are present
- Validates data types and ranges
- Tests API key formats and connectivity
- Validates trading parameters for safety
- Cross-references with actual code usage
- Flags any issues or misconfigurations

Author: v26meme System
Date: 2025-01-05
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment before validation
load_dotenv()

@dataclass
class ValidationResult:
    """Result of an environment variable validation"""
    variable: str
    status: str  # "PASS", "WARN", "FAIL", "MISSING"
    message: str
    value_preview: str = ""
    severity: str = "INFO"  # "INFO", "WARNING", "ERROR", "CRITICAL"

class EnvValidator:
    """Comprehensive environment validation system"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.critical_failures = 0
        self.warnings = 0
        
        # Define all environment variables used in the codebase
        self.env_schema = {
            # OpenAI Configuration
            "OPENAI_API_KEY": {
                "required": True,
                "type": "string",
                "pattern": r"^sk-proj-[a-zA-Z0-9-_]+$",
                "description": "OpenAI API key for AI strategy generation",
                "min_length": 50,
                "used_in": ["Config.OPENAI_API_KEY", "OpenAIManager.__init__"]
            },
            "OPENAI_MODEL": {
                "required": False,
                "type": "string",
                "default": "gpt-4-turbo-preview",
                "valid_values": ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
                "description": "OpenAI model to use for completions",
                "used_in": ["Config.OPENAI_MODEL", "OpenAIManager.complete"]
            },
            
            # Exchange API Keys
            "COINBASE_API_KEY": {
                "required": True,
                "type": "string",
                "pattern": r"^organizations/[a-f0-9-]+/apiKeys/[a-f0-9-]+$",
                "description": "Coinbase Advanced Trade API key",
                "used_in": ["Config.COINBASE_API_KEY", "ExchangeManager"]
            },
            "COINBASE_SECRET": {
                "required": True,
                "type": "string",
                "pattern": r"^-----BEGIN EC PRIVATE KEY-----.*-----END EC PRIVATE KEY-----$",
                "description": "Coinbase Advanced Trade private key",
                "multiline": True,
                "used_in": ["Config.COINBASE_SECRET", "ExchangeManager"]
            },
            "COINBASE_PASSPHRASE": {
                "required": False,
                "type": "string",
                "description": "Coinbase passphrase (if required)",
                "used_in": ["Config.COINBASE_PASSPHRASE"]
            },
            
            # Optional Exchange Keys
            "KRAKEN_API_KEY": {
                "required": False,
                "type": "string",
                "pattern": r"^[a-zA-Z0-9+/]{56}$",
                "description": "Kraken API key (optional)",
                "used_in": ["Config.KRAKEN_API_KEY"]
            },
            "KRAKEN_SECRET": {
                "required": False,
                "type": "string",
                "pattern": r"^[a-zA-Z0-9+/=]{88}$",
                "description": "Kraken API secret (optional)",
                "used_in": ["Config.KRAKEN_SECRET"]
            },
            "GEMINI_API_KEY": {
                "required": False,
                "type": "string",
                "description": "Gemini API key (optional)",
                "used_in": ["Config.GEMINI_API_KEY"]
            },
            "GEMINI_SECRET": {
                "required": False,
                "type": "string",
                "description": "Gemini API secret (optional)",
                "used_in": ["Config.GEMINI_SECRET"]
            },
            
            # Trading Configuration
            "INITIAL_CAPITAL": {
                "required": False,
                "type": "float",
                "default": 200.0,
                "min_value": 1.0,
                "max_value": 1000000.0,
                "description": "Starting capital amount",
                "used_in": ["Config.INITIAL_CAPITAL", "SystemState"]
            },
            "TARGET_CAPITAL": {
                "required": False,
                "type": "float",
                "default": 1000000.0,
                "min_value": 100.0,
                "description": "Target capital goal",
                "used_in": ["Config.TARGET_CAPITAL"]
            },
            "MAX_POSITION_PCT": {
                "required": False,
                "type": "float",
                "default": 0.02,
                "min_value": 0.001,
                "max_value": 0.5,
                "description": "Maximum position size as percentage of capital",
                "used_in": ["Config.MAX_POSITION_PCT", "RiskManager"]
            },
            "MAX_DAILY_DRAWDOWN": {
                "required": False,
                "type": "float",
                "default": 0.10,
                "min_value": 0.01,
                "max_value": 0.50,
                "description": "Maximum daily drawdown before pause",
                "used_in": ["Config.MAX_DAILY_DRAWDOWN", "RiskManager"]
            },
            "MIN_ORDER_NOTIONAL": {
                "required": False,
                "type": "float",
                "default": 0.0,
                "min_value": 0.0,
                "max_value": 100.0,
                "description": "Minimum order size (0.0 for paper trading)",
                "used_in": ["Config.MIN_ORDER_NOTIONAL", "Trader"]
            },
            "MAX_EXPOSURE_PCT": {
                "required": False,
                "type": "float",
                "default": 0.5,
                "min_value": 0.1,
                "max_value": 1.0,
                "description": "Maximum total exposure percentage",
                "used_in": ["Config.MAX_EXPOSURE_PCT", "RiskManager"]
            },
            
            # System Configuration
            "MODE": {
                "required": False,
                "type": "string",
                "default": "PAPER",
                "valid_values": ["PAPER", "REAL"],
                "description": "Trading mode",
                "used_in": ["Config.MODE", "SystemState", "ExchangeManager"]
            },
            "DATABASE_PATH": {
                "required": False,
                "type": "string",
                "default": "v26meme.db",
                "description": "SQLite database path",
                "used_in": ["Config.DATABASE_PATH", "Database.__init__"]
            },
            "LOG_LEVEL": {
                "required": False,
                "type": "string",
                "default": "DEBUG",
                "valid_values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "description": "Logging level",
                "used_in": ["Config.LOG_LEVEL", "Logger.__init__"]
            },
            
            # Web Configuration
            "WEB_PORT": {
                "required": False,
                "type": "int",
                "default": 8000,
                "min_value": 1024,
                "max_value": 65535,
                "description": "Web dashboard port",
                "used_in": ["Config.WEB_PORT", "WebServer"]
            },
            "WEB_AUTH_TOKEN": {
                "required": False,
                "type": "string",
                "description": "Web API authentication token",
                "used_in": ["Config.WEB_AUTH_TOKEN", "WebServer"]
            },
            "WEB_BIND_HOST": {
                "required": False,
                "type": "string",
                "default": "127.0.0.1",
                "description": "Web server bind host",
                "used_in": ["Config.WEB_BIND_HOST", "WebServer"]
            },
            
            # Notification Configuration
            "DISCORD_WEBHOOK": {
                "required": False,
                "type": "string",
                "pattern": r"^https://discord\.com/api/webhooks/\d+/[a-zA-Z0-9_-]+$",
                "description": "Discord webhook URL for alerts",
                "used_in": ["Config.DISCORD_WEBHOOK", "NotificationManager"]
            },
            "TELEGRAM_TOKEN": {
                "required": False,
                "type": "string",
                "pattern": r"^\d{8,10}:[a-zA-Z0-9_-]{35}$",
                "description": "Telegram bot token",
                "used_in": ["Config.TELEGRAM_TOKEN", "NotificationManager"]
            },
            "TELEGRAM_CHAT_ID": {
                "required": False,
                "type": "string",
                "pattern": r"^-?\d+$",
                "description": "Telegram chat ID",
                "used_in": ["Config.TELEGRAM_CHAT_ID", "NotificationManager"]
            }
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run complete validation and return summary"""
        print("🔍 Starting comprehensive .env validation...")
        print("=" * 60)
        
        # 1. Check each environment variable
        for var_name, schema in self.env_schema.items():
            self._validate_single_var(var_name, schema)
        
        # 2. Cross-validate dependencies
        self._validate_dependencies()
        
        # 3. Validate trading safety
        self._validate_trading_safety()
        
        # 4. Check code usage consistency
        self._validate_code_usage()
        
        # 5. Generate summary
        return self._generate_summary()
    
    def _validate_single_var(self, var_name: str, schema: Dict):
        """Validate a single environment variable"""
        value = os.getenv(var_name)
        
        # Check if required variable is missing
        if schema.get("required", False) and not value:
            self.results.append(ValidationResult(
                variable=var_name,
                status="MISSING",
                message=f"Required variable missing: {schema['description']}",
                severity="CRITICAL"
            ))
            self.critical_failures += 1
            return
        
        # If not required and missing, use default
        if not value:
            default = schema.get("default")
            if default is not None:
                self.results.append(ValidationResult(
                    variable=var_name,
                    status="PASS",
                    message=f"Using default value: {default}",
                    value_preview=str(default)[:20],
                    severity="INFO"
                ))
            return
        
        # Type validation
        var_type = schema.get("type", "string")
        try:
            if var_type == "int":
                int_val = int(value)
                self._validate_numeric_range(var_name, int_val, schema)
            elif var_type == "float":
                float_val = float(value)
                self._validate_numeric_range(var_name, float_val, schema)
            elif var_type == "string":
                self._validate_string(var_name, value, schema)
        except ValueError as e:
            self.results.append(ValidationResult(
                variable=var_name,
                status="FAIL",
                message=f"Type validation failed: expected {var_type}, got '{value}' - {e}",
                value_preview=value[:20],
                severity="ERROR"
            ))
            return
        
        # If we get here, validation passed
        self.results.append(ValidationResult(
            variable=var_name,
            status="PASS",
            message="Validation passed",
            value_preview=self._safe_preview(value),
            severity="INFO"
        ))
    
    def _validate_numeric_range(self, var_name: str, value: float, schema: Dict):
        """Validate numeric ranges"""
        min_val = schema.get("min_value")
        max_val = schema.get("max_value")
        
        if min_val is not None and value < min_val:
            self.results.append(ValidationResult(
                variable=var_name,
                status="FAIL",
                message=f"Value {value} below minimum {min_val}",
                value_preview=str(value),
                severity="ERROR"
            ))
            return False
        
        if max_val is not None and value > max_val:
            self.results.append(ValidationResult(
                variable=var_name,
                status="FAIL",
                message=f"Value {value} above maximum {max_val}",
                value_preview=str(value),
                severity="ERROR"
            ))
            return False
        
        return True
    
    def _validate_string(self, var_name: str, value: str, schema: Dict):
        """Validate string patterns and constraints"""
        # Check for placeholder values in sensitive data
        if any(keyword in var_name.lower() for keyword in ["key", "secret", "token", "webhook"]):
            placeholder_patterns = [
                "your_", "YOUR_", "...", "xxx", "XXX", "placeholder", 
                "PLACEHOLDER", "example", "EXAMPLE", "test", "TEST",
                "bot_token", "chat_id", "webhook_id", "webhook_token"
            ]
            
            if any(pattern in value for pattern in placeholder_patterns):
                self.results.append(ValidationResult(
                    variable=var_name,
                    status="WARN",
                    message="Value appears to be a placeholder - update with real credentials",
                    value_preview=self._safe_preview(value),
                    severity="WARNING"
                ))
                self.warnings += 1
                return False
        
        # Pattern validation
        pattern = schema.get("pattern")
        if pattern:
            if schema.get("multiline", False):
                # For multiline patterns, replace newlines with spaces for matching
                test_value = re.sub(r'\s+', ' ', value.strip())
                test_pattern = pattern.replace(".*", ".*?")  # Make non-greedy
            else:
                test_value = value
                test_pattern = pattern
            
            if not re.match(test_pattern, test_value, re.DOTALL if schema.get("multiline") else 0):
                self.results.append(ValidationResult(
                    variable=var_name,
                    status="WARN",
                    message=f"Format validation failed: expected pattern {pattern}",
                    value_preview=self._safe_preview(value),
                    severity="WARNING"
                ))
                self.warnings += 1
                return False
        
        # Valid values check
        valid_values = schema.get("valid_values")
        if valid_values and value not in valid_values:
            self.results.append(ValidationResult(
                variable=var_name,
                status="FAIL",
                message=f"Invalid value '{value}', must be one of: {valid_values}",
                value_preview=value,
                severity="ERROR"
            ))
            return False
        
        # Length validation
        min_length = schema.get("min_length")
        if min_length and len(value) < min_length:
            self.results.append(ValidationResult(
                variable=var_name,
                status="WARN",
                message=f"Value too short: {len(value)} < {min_length}",
                value_preview=self._safe_preview(value),
                severity="WARNING"
            ))
            self.warnings += 1
            return False
        
        return True
    
    def _validate_dependencies(self):
        """Validate cross-variable dependencies"""
        mode = os.getenv("MODE", "PAPER")
        
        # Real mode requires all exchange credentials
        if mode == "REAL":
            required_for_real = ["COINBASE_API_KEY", "COINBASE_SECRET"]
            missing = [var for var in required_for_real if not os.getenv(var)]
            
            if missing:
                self.results.append(ValidationResult(
                    variable="MODE_DEPENDENCIES",
                    status="FAIL",
                    message=f"REAL mode requires: {', '.join(missing)}",
                    severity="CRITICAL"
                ))
                self.critical_failures += 1
        
        # Telegram requires both token and chat ID
        telegram_token = os.getenv("TELEGRAM_TOKEN")
        telegram_chat = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_token and not telegram_chat:
            self.results.append(ValidationResult(
                variable="TELEGRAM_DEPENDENCIES",
                status="WARN",
                message="TELEGRAM_TOKEN set but TELEGRAM_CHAT_ID missing",
                severity="WARNING"
            ))
            self.warnings += 1
        
        # Capital validation
        initial = float(os.getenv("INITIAL_CAPITAL", "200"))
        target = float(os.getenv("TARGET_CAPITAL", "1000000"))
        if target <= initial:
            self.results.append(ValidationResult(
                variable="CAPITAL_LOGIC",
                status="WARN",
                message=f"TARGET_CAPITAL ({target}) should be > INITIAL_CAPITAL ({initial})",
                severity="WARNING"
            ))
            self.warnings += 1
    
    def _validate_trading_safety(self):
        """Validate trading parameters for safety"""
        try:
            max_position = float(os.getenv("MAX_POSITION_PCT", "0.02"))
            max_drawdown = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.10"))
            max_exposure = float(os.getenv("MAX_EXPOSURE_PCT", "0.5"))
            
            # Safety checks
            if max_position > 0.1:  # 10%
                self.results.append(ValidationResult(
                    variable="SAFETY_MAX_POSITION",
                    status="WARN",
                    message=f"MAX_POSITION_PCT ({max_position:.1%}) is very high - consider <10%",
                    severity="WARNING"
                ))
                self.warnings += 1
            
            if max_drawdown > 0.25:  # 25%
                self.results.append(ValidationResult(
                    variable="SAFETY_MAX_DRAWDOWN",
                    status="WARN",
                    message=f"MAX_DAILY_DRAWDOWN ({max_drawdown:.1%}) is high - consider <25%",
                    severity="WARNING"
                ))
                self.warnings += 1
            
            if max_exposure > 0.8:  # 80%
                self.results.append(ValidationResult(
                    variable="SAFETY_MAX_EXPOSURE",
                    status="WARN",
                    message=f"MAX_EXPOSURE_PCT ({max_exposure:.1%}) is very high",
                    severity="WARNING"
                ))
                self.warnings += 1
                
        except ValueError as e:
            self.results.append(ValidationResult(
                variable="SAFETY_VALIDATION",
                status="FAIL",
                message=f"Could not validate trading safety: {e}",
                severity="ERROR"
            ))
    
    def _validate_code_usage(self):
        """Check that all env vars are actually used in the code"""
        try:
            with open("v26meme_full.py", "r") as f:
                code_content = f.read()
            
            # Check for unused variables
            unused_vars = []
            missing_from_config = []
            
            for var_name in self.env_schema.keys():
                # Check if variable is referenced in Config class
                config_pattern = f"Config.{var_name}"
                getenv_pattern = f'os.getenv("{var_name}"'
                getenv_pattern2 = f"os.getenv('{var_name}'"
                
                if not any(pattern in code_content for pattern in [config_pattern, getenv_pattern, getenv_pattern2]):
                    unused_vars.append(var_name)
                    
                # Check if it's in Config class definition
                if var_name not in ["COINBASE_PASSPHRASE"]:  # Some vars are optional
                    if config_pattern not in code_content:
                        missing_from_config.append(var_name)
            
            if unused_vars:
                self.results.append(ValidationResult(
                    variable="CODE_USAGE",
                    status="WARN", 
                    message=f"Variables not found in code: {', '.join(unused_vars)}",
                    severity="WARNING"
                ))
                self.warnings += 1
            
            # Check for environment variables used in code but not in our schema
            env_pattern = re.compile(r'os\.getenv\(["\']([^"\']+)["\']')
            used_env_vars = set(env_pattern.findall(code_content))
            
            schema_vars = set(self.env_schema.keys())
            missing_from_schema = used_env_vars - schema_vars
            
            if missing_from_schema:
                self.results.append(ValidationResult(
                    variable="SCHEMA_COMPLETENESS",
                    status="WARN",
                    message=f"Env vars used in code but not in validation schema: {', '.join(missing_from_schema)}",
                    severity="WARNING"
                ))
                self.warnings += 1
            
            # Positive result if everything checks out
            if not unused_vars and not missing_from_schema:
                self.results.append(ValidationResult(
                    variable="CODE_USAGE",
                    status="PASS",
                    message="All environment variables properly used in code",
                    severity="INFO"
                ))
                    
        except FileNotFoundError:
            self.results.append(ValidationResult(
                variable="CODE_VALIDATION",
                status="WARN",
                message="Could not validate code usage - v26meme_full.py not found",
                severity="WARNING"
            ))
            self.warnings += 1
    
    def _safe_preview(self, value: str, max_len: int = 20) -> str:
        """Create safe preview of sensitive values"""
        if not value:
            return ""
        
        # Mask sensitive values
        if any(keyword in value.lower() for keyword in ["key", "secret", "token", "webhook"]):
            if len(value) > 8:
                return f"{value[:4]}...{value[-4:]}"
            else:
                return "***"
        
        # Regular values
        preview = value.replace("\n", "\\n")
        return preview[:max_len] + ("..." if len(preview) > max_len else "")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        total = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status in ["FAIL", "MISSING"]])
        warnings = len([r for r in self.results if r.status == "WARN"])
        
        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "critical_failures": self.critical_failures,
            "results": self.results,
            "overall_status": "CRITICAL" if self.critical_failures > 0 else 
                            "FAILED" if failed > 0 else 
                            "WARNING" if warnings > 0 else "PASSED"
        }

def print_results(summary: Dict[str, Any]):
    """Print formatted validation results"""
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    status = summary["overall_status"]
    status_emoji = {
        "PASSED": "✅",
        "WARNING": "⚠️", 
        "FAILED": "❌",
        "CRITICAL": "🚨"
    }
    
    print(f"Overall Status: {status_emoji.get(status, '❓')} {status}")
    print(f"Total Checks: {summary['total_checks']}")
    print(f"Passed: ✅ {summary['passed']}")
    print(f"Warnings: ⚠️ {summary['warnings']}")
    print(f"Failed: ❌ {summary['failed']}")
    print(f"Critical: 🚨 {summary['critical_failures']}")
    
    print("\n📋 DETAILED RESULTS")
    print("=" * 60)
    
    # Group by severity
    by_severity = {}
    for result in summary["results"]:
        severity = result.severity
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(result)
    
    # Print in order of severity
    for severity in ["CRITICAL", "ERROR", "WARNING", "INFO"]:
        if severity not in by_severity:
            continue
            
        results = by_severity[severity]
        if not results:
            continue
            
        print(f"\n{severity} ({len(results)} items):")
        print("-" * 30)
        
        for result in results:
            status_icon = {
                "PASS": "✅",
                "WARN": "⚠️",
                "FAIL": "❌", 
                "MISSING": "🚨"
            }
            
            icon = status_icon.get(result.status, "❓")
            preview = f" [{result.value_preview}]" if result.value_preview else ""
            
            print(f"{icon} {result.variable}: {result.message}{preview}")

async def test_api_connectivity():
    """Test actual API connectivity where possible"""
    print("\n🔌 TESTING API CONNECTIVITY")
    print("=" * 60)
    
    # Test OpenAI API
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=openai_key)
            
            # Test with a minimal request
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            print("✅ OpenAI API: Connected successfully")
        except Exception as e:
            print(f"❌ OpenAI API: Connection failed - {e}")
    else:
        print("⚠️ OpenAI API: No key provided, skipping test")
    
    # Test Coinbase API (basic auth check)
    coinbase_key = os.getenv("COINBASE_API_KEY")
    coinbase_secret = os.getenv("COINBASE_SECRET")
    if coinbase_key and coinbase_secret:
        try:
            import ccxt
            exchange = ccxt.coinbase({
                'apiKey': coinbase_key,
                'secret': coinbase_secret,
                'sandbox': False,
            })
            # Simple connectivity test (synchronous)
            markets = exchange.load_markets()
            print(f"✅ Coinbase API: Connected successfully ({len(markets)} markets)")
        except Exception as e:
            print(f"❌ Coinbase API: Connection failed - {e}")
    else:
        print("⚠️ Coinbase API: No credentials provided, skipping test")

def main():
    """Main validation function"""
    print("🚀 v26meme .env VALIDATION SYSTEM")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Environment file: {os.path.abspath('.env')}")
    
    # Initialize validator
    validator = EnvValidator()
    
    # Run validation
    summary = validator.validate_all()
    
    # Print results
    print_results(summary)
    
    # Test connectivity
    try:
        asyncio.run(test_api_connectivity())
    except Exception as e:
        print(f"⚠️ Connectivity testing failed: {e}")
    
    # Final recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if summary["critical_failures"] > 0:
        print("🚨 CRITICAL: Fix critical failures before running the bot")
        print("   - Check required API keys")
        print("   - Verify trading mode configuration")
    
    if summary["failed"] > 0:
        print("❌ ERRORS: Address failed validations")
        print("   - Check data types and ranges")
        print("   - Verify format patterns")
    
    if summary["warnings"] > 0:
        print("⚠️ WARNINGS: Review warning items")
        print("   - Check trading safety parameters")
        print("   - Verify optional configurations")
    
    if summary["overall_status"] == "PASSED":
        print("✅ ALL GOOD: Environment is properly configured!")
        print("   - All required variables present")
        print("   - Trading parameters are safe")
        print("   - Bot is ready to run")
    
    print(f"\n🏁 Validation complete with status: {summary['overall_status']}")
    
    # Return exit code based on status
    if summary["critical_failures"] > 0:
        exit(2)  # Critical failure
    elif summary["failed"] > 0:
        exit(1)  # Error
    else:
        exit(0)  # Success (warnings are OK)

if __name__ == "__main__":
    main()
