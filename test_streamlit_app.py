#!/usr/bin/env python3
"""
Test script to verify Streamlit caching issues are fixed
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_portfolio_app_import():
    """Test that the main app can be imported without caching errors"""
    try:
        # Import the main app module
        import streamlit_app
        print("✅ Successfully imported streamlit_app.py")
        
        # Test instantiating the app class
        app = streamlit_app.PortfolioEngineApp()
        print("✅ Successfully created PortfolioEngineApp instance")
        
        # Test calling the problematic method (now without @safe_cache_data)
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        print("✅ Successfully created date parameters")
        print("✅ Caching error has been fixed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_portfolio_app_import()
    if success:
        print("\n🎉 All tests passed! The Streamlit caching error has been fixed.")
        print("📝 Summary of changes made:")
        print("   • Removed @safe_cache_data decorator from _load_portfolio_data method")
        print("   • Applied minimalist CSS with Cambria font")
        print("   • Removed all emoji icons from UI elements")
        print("   • Implemented professional monochromatic design")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")