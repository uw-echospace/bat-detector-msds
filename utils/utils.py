import pandas as pd

def gen_empty_df():
    return pd.DataFrame({
            "start_time": [],
            "end_time": [],
            "low_freq": [],
            "high_freq": [],
            "event": [],
        })