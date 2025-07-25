#!/usr/bin/env python3
"""
Test Database Configuration System
Shows how to switch between local and online databases
"""

import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs.config import Config

def test_database_switching():
    """Test database configuration switching"""
    
    print("🧪 TESTING DATABASE CONFIGURATION SYSTEM")
    print("=" * 60)
    
    # Show initial configuration
    print("📊 INITIAL CONFIGURATION:")
    db_info = Config.get_database_info()
    print(f"   Mode: {'LOCAL' if db_info['using_local'] else 'ONLINE'}")
    print(f"   Current URI: {db_info['current_uri']}")
    print(f"   Database: {db_info['database_name']}")
    print(f"   Collection: {db_info['collection_name']}")
    print()
    
    # Test switching to online
    print("🌐 SWITCHING TO ONLINE DATABASE:")
    Config.switch_to_online_database()
    db_info = Config.get_database_info()
    print(f"   Mode: {'LOCAL' if db_info['using_local'] else 'ONLINE'}")
    print(f"   Current URI: {db_info['current_uri'][:50]}...")
    print()
    
    # Test switching to local
    print("🏠 SWITCHING TO LOCAL DATABASE:")
    Config.switch_to_local_database()
    db_info = Config.get_database_info()
    print(f"   Mode: {'LOCAL' if db_info['using_local'] else 'ONLINE'}")
    print(f"   Current URI: {db_info['current_uri']}")
    print()
    
    print("✅ Database configuration system working correctly!")
    print()
    print("💡 USAGE IN YOUR SCRIPTS:")
    print("   python main_optimized_threading.py --local-db    # Use local database")
    print("   python main_optimized_threading.py --online-db   # Use online database")
    print("   python main_optimized_threading.py              # Use default (local)")

if __name__ == "__main__":
    test_database_switching()
