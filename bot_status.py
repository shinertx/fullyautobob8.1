#!/usr/bin/env python3
"""
v26meme Bot Status & Environment Summary
========================================
Quick status check for the running bot and validated environment.
"""

import os
import subprocess
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def check_bot_status():
    """Check if the bot is running"""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        )
        for line in result.stdout.split('\n'):
            if 'v26meme_full.py' in line and 'python3' in line:
                parts = line.split()
                pid = parts[1]
                cpu = parts[2]
                memory = parts[3]
                return {
                    "running": True,
                    "pid": pid,
                    "cpu_percent": cpu,
                    "memory_percent": memory,
                    "command": ' '.join(parts[10:])
                }
        return {"running": False}
    except Exception as e:
        return {"running": False, "error": str(e)}

def get_env_summary():
    """Get environment configuration summary"""
    return {
        "mode": os.getenv("MODE", "PAPER"),
        "initial_capital": os.getenv("INITIAL_CAPITAL", "200"),
        "max_position_pct": os.getenv("MAX_POSITION_PCT", "0.02"),
        "log_level": os.getenv("LOG_LEVEL", "DEBUG"),
        "web_port": os.getenv("WEB_PORT", "8000"),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "coinbase_configured": bool(os.getenv("COINBASE_API_KEY"))
    }

def get_validation_status():
    """Get latest validation results"""
    try:
        if os.path.exists("env_validation_report.json"):
            with open("env_validation_report.json", "r") as f:
                data = json.load(f)
                return data["report"]
        return {"status": "No validation report found"}
    except Exception:
        return {"status": "Error reading validation report"}

def main():
    print("🤖 v26meme Bot Status Report")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Bot Status
    print("\n🔍 Bot Status:")
    status = check_bot_status()
    if status["running"]:
        print(f"✅ RUNNING (PID: {status['pid']})")
        print(f"   CPU: {status['cpu_percent']}%")
        print(f"   Memory: {status['memory_percent']}%")
    else:
        print("❌ NOT RUNNING")
        if "error" in status:
            print(f"   Error: {status['error']}")
    
    # Environment Summary
    print("\n⚙️ Environment Configuration:")
    env = get_env_summary()
    print(f"   Mode: {env['mode']}")
    print(f"   Initial Capital: ${env['initial_capital']}")
    print(f"   Max Position: {float(env['max_position_pct'])*100:.1f}%")
    print(f"   Log Level: {env['log_level']}")
    print(f"   Web Port: {env['web_port']}")
    print(f"   OpenAI: {'✓' if env['openai_configured'] else '✗'}")
    print(f"   Coinbase: {'✓' if env['coinbase_configured'] else '✗'}")
    
    # Validation Status
    print("\n✅ Validation Status:")
    validation = get_validation_status()
    if "overall_status" in validation:
        status_icon = {
            "PASSED": "✅",
            "WARNING": "⚠️",
            "FAILED": "❌",
            "CRITICAL": "🚨"
        }.get(validation["overall_status"], "❓")
        
        print(f"   Overall: {status_icon} {validation['overall_status']}")
        print(f"   Tests: {validation.get('total_tests', 0)}")
        print(f"   Critical: {validation.get('critical', 0)}")
        print(f"   Errors: {validation.get('errors', 0)}")
        print(f"   Warnings: {validation.get('warnings', 0)}")
    else:
        print(f"   Status: {validation.get('status', 'Unknown')}")
    
    # Quick Actions
    print("\n🚀 Quick Actions:")
    if status["running"]:
        print("   • View logs: tail -f v26meme.log")
        print("   • Web dashboard: http://localhost:8000")
        print("   • Stop bot: kill <PID>")
    else:
        print("   • Start bot: python3 v26meme_full.py")
        print("   • Validate env: python3 validate_env_master.py")
    
    print("   • Full validation: python3 validate_env_master.py")
    print("   • Basic validation: python3 validate_env.py")
    print("   • Logic validation: python3 validate_env_logic.py")

if __name__ == "__main__":
    main()
