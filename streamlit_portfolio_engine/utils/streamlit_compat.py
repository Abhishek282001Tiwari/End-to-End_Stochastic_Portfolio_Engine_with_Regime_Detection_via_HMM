#!/usr/bin/env python3
"""
Streamlit Version Compatibility Utilities

Provides version-safe wrappers for Streamlit functions that have changed
between versions to ensure compatibility across different Streamlit versions.
"""

import streamlit as st


def safe_rerun():
    """Version-safe rerun function for page refresh"""
    try:
        st.rerun()  # New syntax (Streamlit >= 1.27.0)
    except AttributeError:
        st.experimental_rerun()  # Old syntax fallback


def safe_cache_data(func=None, **kwargs):
    """Version-safe cache_data decorator for caching data"""
    def decorator(f):
        try:
            return st.cache_data(**kwargs)(f)
        except AttributeError:
            return st.experimental_memo(**kwargs)(f)
    
    if func is None:
        return decorator
    return decorator(func)


def safe_cache_resource(func=None, **kwargs):
    """Version-safe cache_resource decorator for caching resources"""
    def decorator(f):
        try:
            return st.cache_resource(**kwargs)(f)
        except AttributeError:
            return st.experimental_singleton(**kwargs)(f)
    
    if func is None:
        return decorator
    return decorator(func)


def safe_get_query_params():
    """Version-safe query parameter getter"""
    try:
        return st.query_params  # New syntax
    except AttributeError:
        return st.experimental_get_query_params()  # Old syntax


def safe_set_query_params(**params):
    """Version-safe query parameter setter"""
    try:
        st.query_params.update(params)  # New syntax
    except AttributeError:
        st.experimental_set_query_params(**params)  # Old syntax


# Version detection utilities
def get_streamlit_version():
    """Get the current Streamlit version"""
    try:
        return st.__version__
    except AttributeError:
        return "unknown"


def is_streamlit_version_at_least(version_str):
    """Check if Streamlit version is at least the specified version"""
    try:
        import packaging.version
        current_version = packaging.version.parse(st.__version__)
        target_version = packaging.version.parse(version_str)
        return current_version >= target_version
    except (ImportError, AttributeError):
        # If we can't determine version, assume older version
        return False


# Compatibility constants
HAS_NEW_RERUN = hasattr(st, 'rerun')
HAS_NEW_CACHE = hasattr(st, 'cache_data')
HAS_NEW_QUERY_PARAMS = hasattr(st, 'query_params')