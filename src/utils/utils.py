import pandas as pd

# Publically accessible dumping ground for stuff that doesn't fit anywhere else

def gen_empty_df():
    """
    Generates an empty dataframe with the correct columns for the output csv
    """
    return pd.DataFrame({
            "start_time": [],
            "end_time": [],
            "low_freq": [],
            "high_freq": [],
            "detection_confidence":[],
            "event": [],
        })

def convert_df_ravenpro(df: pd.DataFrame):
    """
    Converts a dataframe to the format used by RavenPro
    """

    ravenpro_df = df.copy()

    ravenpro_df.rename(columns={
        "start_time": "Begin Time (s)",
        "end_time": "End Time (s)",
        "low_freq": "Low Freq (Hz)",
        "high_freq": "High Freq (Hz)",
        "event": "Annotation",
    }, inplace=True)

    ravenpro_df["Selection"] = "Waveform 1"
    ravenpro_df["View"] = "1"
    ravenpro_df["Channel"] = "1"

    return ravenpro_df