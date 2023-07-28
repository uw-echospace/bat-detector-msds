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
            "event": [],
            "class": [], 
            "class_prob": [],
            "det_prob": [],
            "individual": []
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

    ravenpro_df["Selection"] = pd.Series(range(1, df.shape[0]))
    ravenpro_df["View"] = "Waveform 1"
    ravenpro_df["Channel"] = "1"

    return ravenpro_df