import sys
import os
import pandas as pd

# Add the project root to sys.path so src can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import get_data

def test_get_data_returns_dataframe():
    df = get_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Agent" in df.columns 
