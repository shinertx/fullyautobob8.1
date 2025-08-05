#!/usr/bin/env python3
"""
MASTER .env Validation Suite for v26meme Trading Bot
====================================================
This is the complete validation suite that combines:
1. Environment variable format validation
2. Logic flow validation  
3. API connectivity testing
4. Runtime simulation testing
5. Security validation
6. Performance impact assessment

Run this before deploying to production!

Author: v26meme System
Date: 2025-01-05
"""

import os
import sys
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

# Load environment
load_dotenv()

class MasterValidator:
    """Complete validation suite"""
    
    def __init__(self):
        self.results = {
            "format": [],
            "logic": [],
            "connectivity": [],
            "runtime": [],
            "security": [],
            "performance": []
        }
        self.critical_count = 0
        self.error_count = 0
        self.warning_count = 0
        
    async def run_complete_validation(self):
        """Run all validation suites"""
        print("🚀 MASTER .env VALIDATION SUITE")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Working Directory: {os.getcwd()}")
        print(f"Environment File: {os.path.abspath('.env')}")
        
        # 1. Format Validation
        await self.validate_format()
        
        # 2. Logic Validation
        await self.validate_logic()
        
        # 3. Connectivity Testing
        await self.validate_connectivity()
        
        # 4. Runtime Simulation
        await self.validate_runtime()
        
        # 5. Security Assessment
        await self.validate_security()
        
        # 6. Performance Impact
        await self.validate_performance()
        
        # Generate final report
        return self.generate_master_report()
    
    async def validate_format(self):
        """Validate environment variable formats"""
        print("\n📋 FORMAT VALIDATION")
        print("-" * 40)
        
        env_vars = {
            # Required variables
            "OPENAI_API_KEY": {"required": True, "pattern": r"^sk-proj-[a-zA-Z0-9-_]+$"},
            "COINBASE_API_KEY": {"required": True, "pattern": r"^organizations/[a-f0-9-]+/apiKeys/[a-f0-9-]+$"},
            "COINBASE_SECRET": {"required": True, "multiline": True},
            
            # Trading parameters
            "INITIAL_CAPITAL": {"type": "float", "min": 1, "max": 1000000},
            "TARGET_CAPITAL": {"type": "float", "min": 100},
            "MAX_POSITION_PCT": {"type": "float", "min": 0.001, "max": 0.5},
            "MAX_DAILY_DRAWDOWN": {"type": "float", "min": 0.01, "max": 0.5},
            "MIN_ORDER_NOTIONAL": {"type": "float", "min": 0, "max": 100},
            "MAX_EXPOSURE_PCT": {"type": "float", "min": 0.1, "max": 1.0},
            
            # System configuration
            "MODE": {"valid_values": ["PAPER", "REAL"]},
            "LOG_LEVEL": {"valid_values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
            "WEB_PORT": {"type": "int", "min": 1024, "max": 65535}
        }
        
        for var, schema in env_vars.items():
            result = self.check_env_var(var, schema)
            self.results["format"].append(result)
            print(f"{result['icon']} {var}: {result['message']}")
    
    def check_env_var(self, var_name: str, schema: Dict) -> Dict:
        """Check a single environment variable"""
        value = os.getenv(var_name)
        
        # Check if required
        if schema.get("required", False) and not value:
            self.critical_count += 1
            return {
                "variable": var_name,
                "status": "CRITICAL",
                "icon": "🚨",
                "message": "Required variable missing"
            }
        
        if not value:
            return {
                "variable": var_name,
                "status": "OK",
                "icon": "ℹ️",
                "message": "Optional variable not set"
            }
        
        # Type checking
        var_type = schema.get("type")
        if var_type:
            try:
                if var_type == "int":
                    typed_value = int(value)
                elif var_type == "float":
                    typed_value = float(value)
                else:
                    typed_value = value
                
                # Range checking
                min_val = schema.get("min")
                max_val = schema.get("max")
                
                if min_val is not None and typed_value < min_val:
                    self.error_count += 1
                    return {
                        "variable": var_name,
                        "status": "ERROR",
                        "icon": "❌",
                        "message": f"Value {typed_value} below minimum {min_val}"
                    }
                
                if max_val is not None and typed_value > max_val:
                    self.error_count += 1
                    return {
                        "variable": var_name,
                        "status": "ERROR", 
                        "icon": "❌",
                        "message": f"Value {typed_value} above maximum {max_val}"
                    }
                
            except ValueError:
                self.error_count += 1
                return {
                    "variable": var_name,
                    "status": "ERROR",
                    "icon": "❌",
                    "message": f"Invalid {var_type} value: {value}"
                }
        
        # Valid values check
        valid_values = schema.get("valid_values")
        if valid_values and value not in valid_values:
            self.error_count += 1
            return {
                "variable": var_name,
                "status": "ERROR",
                "icon": "❌",
                "message": f"Invalid value, must be one of: {valid_values}"
            }
        
        # Pattern check
        pattern = schema.get("pattern")
        if pattern:
            import re
            if not re.match(pattern, value, re.DOTALL if schema.get("multiline") else 0):
                self.warning_count += 1
                return {
                    "variable": var_name,
                    "status": "WARNING",
                    "icon": "⚠️",
                    "message": "Format may be incorrect"
                }
        
        return {
            "variable": var_name,
            "status": "OK",
            "icon": "✅",
            "message": "Valid"
        }
    
    async def validate_logic(self):
        """Validate configuration logic"""
        print("\n🧠 LOGIC VALIDATION")
        print("-" * 40)
        
        try:
            # Test trading parameter relationships
            initial = float(os.getenv("INITIAL_CAPITAL", "200"))
            target = float(os.getenv("TARGET_CAPITAL", "1000000"))
            max_pos = float(os.getenv("MAX_POSITION_PCT", "0.02"))
            max_dd = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.10"))
            
            if target <= initial:
                self.warning_count += 1
                self.results["logic"].append({
                    "test": "Capital Logic",
                    "status": "WARNING",
                    "message": f"Target capital ({target}) should be > initial ({initial})"
                })
                print("⚠️ Capital Logic: Target should be greater than initial")
            else:
                self.results["logic"].append({
                    "test": "Capital Logic",
                    "status": "OK",
                    "message": "Target > Initial ✓"
                })
                print("✅ Capital Logic: Valid relationship")
            
            # Test risk parameters
            max_position_value = initial * max_pos
            daily_loss_limit = initial * max_dd
            
            if max_position_value > daily_loss_limit:
                self.warning_count += 1
                self.results["logic"].append({
                    "test": "Risk Logic",
                    "status": "WARNING", 
                    "message": f"Max position (${max_position_value:.2f}) > daily loss limit (${daily_loss_limit:.2f})"
                })
                print(f"⚠️ Risk Logic: Max position exceeds daily loss limit")
            else:
                self.results["logic"].append({
                    "test": "Risk Logic",
                    "status": "OK",
                    "message": "Position sizing within risk limits ✓"
                })
                print("✅ Risk Logic: Position sizing appropriate")
            
            # Test mode consistency
            mode = os.getenv("MODE", "PAPER")
            min_notional = float(os.getenv("MIN_ORDER_NOTIONAL", "0.0"))
            
            if mode == "PAPER" and min_notional != 0.0:
                self.warning_count += 1
                self.results["logic"].append({
                    "test": "Mode Logic",
                    "status": "WARNING",
                    "message": "Paper mode should have MIN_ORDER_NOTIONAL=0.0"
                })
                print("⚠️ Mode Logic: Paper mode should have 0 min notional")
            else:
                self.results["logic"].append({
                    "test": "Mode Logic", 
                    "status": "OK",
                    "message": f"Mode {mode} configuration consistent ✓"
                })
                print(f"✅ Mode Logic: {mode} mode properly configured")
                
        except Exception as e:
            self.error_count += 1
            self.results["logic"].append({
                "test": "Logic Validation",
                "status": "ERROR",
                "message": f"Logic validation failed: {e}"
            })
            print(f"❌ Logic Validation: Failed - {e}")
    
    async def validate_connectivity(self):
        """Test API connectivity"""
        print("\n🔌 CONNECTIVITY VALIDATION")
        print("-" * 40)
        
        # Test OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=openai_key)
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                self.results["connectivity"].append({
                    "service": "OpenAI",
                    "status": "OK",
                    "message": "Connected successfully"
                })
                print("✅ OpenAI API: Connected")
            except Exception as e:
                self.error_count += 1
                self.results["connectivity"].append({
                    "service": "OpenAI",
                    "status": "ERROR",
                    "message": f"Connection failed: {e}"
                })
                print(f"❌ OpenAI API: Failed - {e}")
        else:
            print("⚠️ OpenAI API: No key provided")
        
        # Test Coinbase (if in REAL mode)
        mode = os.getenv("MODE", "PAPER")
        coinbase_key = os.getenv("COINBASE_API_KEY")
        coinbase_secret = os.getenv("COINBASE_SECRET")
        
        if mode == "REAL" and coinbase_key and coinbase_secret:
            try:
                import ccxt
                exchange = ccxt.coinbase({
                    'apiKey': coinbase_key,
                    'secret': coinbase_secret,
                    'sandbox': False
                })
                # Test connection with a simple call
                markets = exchange.load_markets()
                self.results["connectivity"].append({
                    "service": "Coinbase",
                    "status": "OK", 
                    "message": f"Connected ({len(markets)} markets)"
                })
                print(f"✅ Coinbase API: Connected ({len(markets)} markets)")
            except Exception as e:
                self.error_count += 1
                self.results["connectivity"].append({
                    "service": "Coinbase",
                    "status": "ERROR",
                    "message": f"Connection failed: {e}"
                })
                print(f"❌ Coinbase API: Failed - {e}")
        else:
            print(f"ℹ️ Coinbase API: Skipped (mode: {mode})")
    
    async def validate_runtime(self):
        """Simulate runtime scenarios"""
        print("\n⚡ RUNTIME VALIDATION")
        print("-" * 40)
        
        try:
            # Simulate Config initialization
            start_time = time.time()
            
            # Test numeric conversions
            initial_capital = float(os.getenv("INITIAL_CAPITAL", "200"))
            target_capital = float(os.getenv("TARGET_CAPITAL", "1000000"))
            max_position_pct = float(os.getenv("MAX_POSITION_PCT", "0.02"))
            max_daily_drawdown = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.10"))
            min_order_notional = float(os.getenv("MIN_ORDER_NOTIONAL", "0.0"))
            web_port = int(os.getenv("WEB_PORT", "8000"))
            
            config_time = time.time() - start_time
            
            # Simulate position size calculations
            start_time = time.time()
            current_equity = initial_capital * 0.95  # Simulate some trading
            max_position_size = current_equity * max_position_pct
            daily_loss_limit = initial_capital * max_daily_drawdown
            calc_time = time.time() - start_time
            
            self.results["runtime"].append({
                "test": "Config Load Time",
                "status": "OK",
                "message": f"{config_time*1000:.2f}ms"
            })
            
            self.results["runtime"].append({
                "test": "Calculation Performance",
                "status": "OK",
                "message": f"{calc_time*1000:.3f}ms"
            })
            
            print(f"✅ Config Load: {config_time*1000:.2f}ms")
            print(f"✅ Calculations: {calc_time*1000:.3f}ms")
            print(f"✅ Simulated Position Size: ${max_position_size:.2f}")
            print(f"✅ Simulated Daily Limit: ${daily_loss_limit:.2f}")
            
        except Exception as e:
            self.error_count += 1
            self.results["runtime"].append({
                "test": "Runtime Simulation",
                "status": "ERROR", 
                "message": f"Failed: {e}"
            })
            print(f"❌ Runtime Simulation: Failed - {e}")
    
    async def validate_security(self):
        """Assess security configuration"""
        print("\n🔒 SECURITY VALIDATION")
        print("-" * 40)
        
        # Check web security
        web_host = os.getenv("WEB_BIND_HOST", "127.0.0.1")
        web_token = os.getenv("WEB_AUTH_TOKEN", "")
        
        if web_host == "0.0.0.0":
            if not web_token:
                self.warning_count += 1
                self.results["security"].append({
                    "check": "Web Security",
                    "status": "WARNING",
                    "message": "Web server exposed without auth token"
                })
                print("⚠️ Web Security: Server exposed without authentication")
            else:
                self.results["security"].append({
                    "check": "Web Security",
                    "status": "OK",
                    "message": "Exposed but protected with auth token"
                })
                print("✅ Web Security: Protected with auth token")
        else:
            self.results["security"].append({
                "check": "Web Security",
                "status": "OK", 
                "message": f"Bound to {web_host} (localhost)"
            })
            print(f"✅ Web Security: Safely bound to {web_host}")
        
        # Check file permissions
        env_path = ".env"
        if os.path.exists(env_path):
            import stat
            file_stat = os.stat(env_path)
            file_mode = stat.filemode(file_stat.st_mode)
            
            # Check if readable by others
            if file_stat.st_mode & stat.S_IROTH:
                self.warning_count += 1
                self.results["security"].append({
                    "check": "File Permissions",
                    "status": "WARNING",
                    "message": f".env readable by others ({file_mode})"
                })
                print(f"⚠️ File Permissions: .env readable by others ({file_mode})")
            else:
                self.results["security"].append({
                    "check": "File Permissions",
                    "status": "OK",
                    "message": f"Secure permissions ({file_mode})"
                })
                print(f"✅ File Permissions: Secure ({file_mode})")
        
        # Check for sensitive data exposure
        mode = os.getenv("MODE", "PAPER")
        if mode == "REAL":
            self.results["security"].append({
                "check": "Trading Mode",
                "status": "INFO",
                "message": "REAL mode - ensure production security"
            })
            print("ℹ️ Trading Mode: REAL mode - ensure production security")
        else:
            self.results["security"].append({
                "check": "Trading Mode", 
                "status": "OK",
                "message": "PAPER mode - safe for testing"
            })
            print("✅ Trading Mode: PAPER mode - safe for testing")
    
    async def validate_performance(self):
        """Assess performance implications"""
        print("\n🚀 PERFORMANCE VALIDATION")
        print("-" * 40)
        
        # Check logging level impact
        log_level = os.getenv("LOG_LEVEL", "DEBUG")
        if log_level == "DEBUG":
            self.results["performance"].append({
                "check": "Logging Level",
                "status": "INFO",
                "message": "DEBUG logging may impact performance"
            })
            print("ℹ️ Logging Level: DEBUG may impact performance in production")
        else:
            self.results["performance"].append({
                "check": "Logging Level",
                "status": "OK",
                "message": f"{log_level} level appropriate"
            })
            print(f"✅ Logging Level: {log_level} appropriate for production")
        
        # Check capital scaling implications
        initial_capital = float(os.getenv("INITIAL_CAPITAL", "200"))
        target_capital = float(os.getenv("TARGET_CAPITAL", "1000000"))
        growth_factor = target_capital / initial_capital
        
        if growth_factor > 1000:  # More than 1000x growth
            self.results["performance"].append({
                "check": "Capital Scaling",
                "status": "INFO",
                "message": f"Aggressive {growth_factor:.0f}x growth target"
            })
            print(f"ℹ️ Capital Scaling: Aggressive {growth_factor:.0f}x growth target")
        else:
            self.results["performance"].append({
                "check": "Capital Scaling",
                "status": "OK",
                "message": f"Reasonable {growth_factor:.0f}x growth target"
            })
            print(f"✅ Capital Scaling: Reasonable {growth_factor:.0f}x growth target")
        
        # Check position sizing efficiency
        max_position_pct = float(os.getenv("MAX_POSITION_PCT", "0.02"))
        if max_position_pct < 0.01:  # Less than 1%
            self.results["performance"].append({
                "check": "Position Sizing",
                "status": "INFO",
                "message": f"Very conservative {max_position_pct:.1%} position size"
            })
            print(f"ℹ️ Position Sizing: Very conservative {max_position_pct:.1%}")
        else:
            self.results["performance"].append({
                "check": "Position Sizing",
                "status": "OK",
                "message": f"Balanced {max_position_pct:.1%} position size"
            })
            print(f"✅ Position Sizing: Balanced {max_position_pct:.1%}")
    
    def generate_master_report(self):
        """Generate comprehensive final report"""
        total_tests = sum(len(results) for results in self.results.values())
        
        print("\n" + "=" * 80)
        print("📊 MASTER VALIDATION REPORT")
        print("=" * 80)
        
        print(f"Total Tests Run: {total_tests}")
        print(f"Critical Issues: 🚨 {self.critical_count}")
        print(f"Errors: ❌ {self.error_count}")
        print(f"Warnings: ⚠️ {self.warning_count}")
        print(f"Passed: ✅ {total_tests - self.critical_count - self.error_count - self.warning_count}")
        
        # Overall status
        if self.critical_count > 0:
            overall_status = "CRITICAL"
            status_icon = "🚨"
        elif self.error_count > 0:
            overall_status = "FAILED"
            status_icon = "❌"
        elif self.warning_count > 0:
            overall_status = "WARNING"
            status_icon = "⚠️"
        else:
            overall_status = "PASSED"
            status_icon = "✅"
        
        print(f"\nOverall Status: {status_icon} {overall_status}")
        
        # Detailed breakdown
        print(f"\n📋 CATEGORY BREAKDOWN")
        print("-" * 40)
        for category, results in self.results.items():
            category_issues = len([r for r in results if r.get('status') in ['ERROR', 'WARNING', 'CRITICAL']])
            category_icon = "✅" if category_issues == 0 else "⚠️"
            print(f"{category_icon} {category.title()}: {len(results)} tests, {category_issues} issues")
        
        # Final recommendations
        print(f"\n💡 FINAL RECOMMENDATIONS")
        print("-" * 40)
        
        if self.critical_count > 0:
            print("🚨 CRITICAL: Do not run the bot until critical issues are fixed!")
            print("   - Missing required API keys")
            print("   - Invalid configuration values")
        
        if self.error_count > 0:
            print("❌ ERRORS: Fix these issues before production deployment")
            print("   - Configuration logic errors")
            print("   - API connectivity problems")
        
        if self.warning_count > 0:
            print("⚠️ WARNINGS: Review these items for optimal operation")
            print("   - Security considerations")
            print("   - Performance optimizations")
            print("   - Best practice recommendations")
        
        if overall_status == "PASSED":
            print("✅ EXCELLENT: Your .env configuration is production-ready!")
            print("   - All validations passed")
            print("   - APIs are accessible")
            print("   - Configuration is logical and secure")
            print("   - Ready to run v26meme trading bot")
        
        return {
            "overall_status": overall_status,
            "total_tests": total_tests,
            "critical": self.critical_count,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "results": self.results
        }

async def main():
    """Main validation entry point"""
    validator = MasterValidator()
    report = await validator.run_complete_validation()
    
    # Save detailed report
    with open("env_validation_report.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "report": report
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: env_validation_report.json")
    
    # Exit with appropriate code
    if report["critical"] > 0:
        sys.exit(2)  # Critical failure
    elif report["errors"] > 0:
        sys.exit(1)  # Errors
    else:
        sys.exit(0)  # Success

if __name__ == "__main__":
    asyncio.run(main())
