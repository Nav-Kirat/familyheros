import pandas as pd
import streamlit as st

def convert_to_timestamp(date_obj, default=None):
    """Convert various date formats to pandas Timestamp"""
    if date_obj is None:
        return default

    try:
        # Direct conversion instead of using isinstance
        return pd.Timestamp(date_obj)
    except Exception as e:
        st.warning(f"Could not convert date: {date_obj}. Error: {e}")
        return default