import argparse
import glob
from pathlib import Path

import pandas as pd
import datetime as dt
import numpy as np
import exiftool

import batdt2_pipeline as batdetect2_pipeline

def get_recover_folder_from_filepath(filepath):
    if "recover" in str(filepath.parents[1]):
        return filepath.parents[1].name
    else:
        return filepath.parent.name

def get_recover_DATE_from_filepath(filepath):
    recover_folder = get_recover_folder_from_filepath(filepath)
    return recover_folder.split('-')[-1]

def get_SD_card_from_filepath(filepath):
    dir_name = filepath.parent.name
    if "recover" in str(dir_name):
        return str(dir_name).split('_')[-1]
    else:
        return dir_name

def get_SD_unit_from_filepath(filepath):
    sd_card = get_SD_card_from_filepath(filepath)
    if "UBNA" in sd_card:
        sd_unit = sd_card.split('_')[-1]
    else:
        sd_unit = sd_card[-1]
    return sd_unit

def get_file_temperature(comment):
    if "temperature" in comment:
        before_temperature = comment.split('C.')[0][-4:]
        temperature = before_temperature + "C"
        return temperature
    else:
        return "File does not have metadata!"

def get_file_battery(comment):
    if "battery" in comment:
        before_battery = comment.split('V')[0][-3:]
        battery = before_battery + "V"
        return battery
    else:
        return "File does not have metadata!"

def get_file_status(comment):
    if ("microphone" in comment):
        return "Not usable; microphone issues"
    if ("voltage" in comment):
        return "Not usable; battery issues"
    if (comment == "Is empty!" or comment == "Does not exist!"):
        return "Not usable; no metadata"
    else:
        return "Usable for detection"
    
def get_file_comment(filepath):
    if not(filepath.exists()):
        return "Does not exist!"
    if filepath.stat().st_size == 0:
        return "Is empty!"
    
    return "good!"
    

def generate_files_df(cfg):

    raw_files = Path(cfg['input_dir']).glob(pattern='recover-*/**/*.WAV')
    clean_files = []
    for filepath in raw_files:
        if not('trash' in str(filepath).lower()):
            clean_files.append(filepath)

    all_wav_files = sorted(clean_files)
    file_path_column_name = "file_path"
    files_df = pd.DataFrame((all_wav_files), columns=[file_path_column_name])
    print(f"Created file paths column!")
    file_metadata_column_name = "file_metadata"
    files_df[file_metadata_column_name] = files_df[file_path_column_name].apply(lambda x : get_file_comment(x))
    print(f"Created file metadata column!")
    files_df["sample_rate"] = files_df[file_path_column_name].apply(lambda x : get_file_comment(x))
    print(f"Created sample rate column!")
    files_df["audiomoth_artist_ID"] = files_df[file_path_column_name].apply(lambda x : get_file_comment(x))
    print(f"Created Audiomoth artist ID column!")
    files_df["file_duration"] = files_df[file_path_column_name].apply(lambda x : get_file_comment(x))
    print(f"Created file duration column!")

    with exiftool.ExifToolHelper() as et:
        good_paths = list(files_df.loc[files_df[file_metadata_column_name] == "good!"][file_path_column_name].values)
        comments = []
        sample_rates = []
        artists = []
        durations = []
        for path in good_paths:
            try:
                raw_data = et.get_metadata(path)
                datapoint = raw_data[0]
                no_audiomoth_comment = "File has no Audiomoth-related comment"

                if "Composite:Duration" in datapoint.keys():
                    durations += [(datapoint["Composite:Duration"])]
                else:
                    durations += [no_audiomoth_comment]

                if "RIFF:SampleRate" in datapoint.keys():
                    sample_rates += [datapoint["RIFF:SampleRate"]]
                else:
                    sample_rates += [no_audiomoth_comment]

                if "RIFF:Comment" in datapoint.keys():
                    comments += [datapoint["RIFF:Comment"]]
                else:
                    comments += [no_audiomoth_comment]

                if "RIFF:Artist" in datapoint.keys():
                    artists += [datapoint["RIFF:Artist"]]
                else:
                    artists += [no_audiomoth_comment]
            except exiftool.exceptions.ExifToolExecuteError:
                error_comment = "File has no comment due to error!"
                artists += [error_comment]
                sample_rates += [error_comment]
                comments += [error_comment]
                durations += [error_comment]

        files_df[file_metadata_column_name].loc[files_df[file_metadata_column_name] == "good!"] = comments
        files_df["sample_rate"].loc[files_df["sample_rate"] == "good!"] = sample_rates
        files_df["audiomoth_artist_ID"].loc[files_df["audiomoth_artist_ID"] == "good!"] = artists
        files_df["file_duration"].loc[files_df["file_duration"] == "good!"] = durations
    
    print(f"Updated file metadata info using Audiomoth metadata comments!")
    files_df.insert(2, "audiomoth_battery", files_df[file_metadata_column_name].apply(lambda x : get_file_battery(x)))
    print(f"Created Audiomoth battery column!")
    files_df.insert(2, "audiomoth_temperature", files_df[file_metadata_column_name].apply(lambda x : get_file_temperature(x)))
    print(f"Created Audiomoth temperature column!")
    files_df.insert(2, "file_status", files_df[file_metadata_column_name].apply(lambda x : get_file_status(x)))
    print(f"Created file status column!")
    files_df.insert(0, "recover_folder", files_df[file_path_column_name].apply(lambda x : get_recover_folder_from_filepath(x)))
    print(f"Created recover folder column!")

    filepaths = list(files_df[file_path_column_name].values)
    sd_cards = []
    site_names = []
    audiomoth_names = []
    audiomoth_notes = []
    for path in filepaths:
        print(path)
        date = get_recover_DATE_from_filepath(path)
        sd_unit = get_SD_unit_from_filepath(path)
        site_name = get_site_name(date, sd_unit)
        audiomoth_name = get_audiomoth_name(date, sd_unit)
        audiomoth_note = get_audiomoth_notes(date, sd_unit)
        sd_card = get_audiomoth_sd_card(date, sd_unit)
        sd_cards += [sd_card]
        site_names += [site_name]
        audiomoth_names += [audiomoth_name]
        audiomoth_notes += [audiomoth_note]
        print(f'File at {path} recovered from {date} inside UBNA_{sd_unit} and Audiomoth {audiomoth_name} at {site_name}')

    files_df.insert(1, "sd_card_num", sd_cards)
    print(f"Created SD card column!")
    files_df["Deployment notes"] = audiomoth_notes
    print(f"Created audiomoth # column!")
    files_df.insert(1, "audiomoth_num", audiomoth_names)
    print(f"Created audiomoth # column!")
    files_df.insert(0, "site_name", site_names)
    print(f"Created site name column!")
    files_df.insert(0, "datetime_UTC", pd.to_datetime(files_df[file_path_column_name], format="%Y%m%d_%H%M%S", exact=False))
    print(f"Created datetime column!")

    files_df.to_csv(cfg['output_dir'] / cfg["csv_name"])

    return files_df

def get_related_field_records(recover_date):
    """Gets the related field records that stores information of the provided recover-DATE folder

    Parameters
    ----------
    recover_date : `str`
        The date in the recover-DATE folder that corresponds to the date that the Audiomoth was recovered

    Returns
    ----------
    df_fr : `pd.Dataframe`
        The pandas Dataframe object that contains the field records information
    """

    datetime_of_recovery = dt.datetime.strptime(recover_date, "%Y%m%d")
    if str(datetime_of_recovery.year) == "2021":
        df_fr = get_field_records(Path(f"{Path(__file__).parent}/../field_records/ubna_2021.csv"))
    if str(datetime_of_recovery.year) == "2022":
        if datetime_of_recovery < dt.datetime.strptime('20220715', "%Y%m%d"):
            df_fr = get_field_records(Path(f"{Path(__file__).parent}/../field_records/ubna_2022a.csv"))
        else:
            df_fr = get_field_records(Path(f"{Path(__file__).parent}/../field_records/ubna_2022b.csv"))
    if str(datetime_of_recovery.year) == "2023":
        if (datetime_of_recovery.month) >= 5:
            df_fr = get_field_records(Path(f"{Path(__file__).parent}/../field_records/ubna_2023.csv"))
        else:
            df_fr = get_field_records(Path(f"{Path(__file__).parent}/../field_records/ubna_2022b.csv"))
    if str(datetime_of_recovery.year) == "2024":
        if (datetime_of_recovery.month) <= 5:
            df_fr = get_field_records(Path(f"{Path(__file__).parent}/../field_records/ubna_2023.csv"))
        else:
            df_fr = get_field_records(Path(f"{Path(__file__).parent}/../field_records/ubna_2024.csv"))

    return df_fr

def get_audiomoth_sd_card(DATE, SD_CARD_NUM):
    """Gets the location where an AudioMoth was deployed at a certain date using the deployment field records.
    Will be used to plot activity with the right location label so user can tell location of activity by plots.

    Parameters
    ------------
    DATE : `str`
        The date when an AudioMoth was deployed
    SD_CARD_NUM : `str`
        The SD card inside the AudioMoth to identify which AudioMoth the user wants.

    Returns
    ------------
    site_name : `str`
        - Name of the location where the Audiomoth was deployed at that date
        according to the field records.
        - If the deployment is not recorded, site_name will be "(Site not found in Field Records)"
    """


    recover_date = DATE.split('_')[0]
    datetime_of_recovery = dt.datetime.strptime(recover_date, "%Y%m%d")

    df_fr = get_related_field_records(recover_date)

    cond1 = df_fr["Upload folder name"]==f"recover-{DATE}"
    if datetime_of_recovery < dt.datetime.strptime('20220715', "%Y%m%d"):
        notes = df_fr.loc[cond1, "SD card #"]
        if (notes.empty):
            notes = "(Audiomoth SD card not found in Field Records)"
        else:
            notes = notes.item()
    else:
        notes = f"{SD_CARD_NUM}"
    
    return notes

def get_audiomoth_notes(DATE, SD_CARD_NUM):
    """Gets the location where an AudioMoth was deployed at a certain date using the deployment field records.
    Will be used to plot activity with the right location label so user can tell location of activity by plots.

    Parameters
    ------------
    DATE : `str`
        The date when an AudioMoth was deployed
    SD_CARD_NUM : `str`
        The SD card inside the AudioMoth to identify which AudioMoth the user wants.

    Returns
    ------------
    site_name : `str`
        - Name of the location where the Audiomoth was deployed at that date
        according to the field records.
        - If the deployment is not recorded, site_name will be "(Site not found in Field Records)"
    """


    recover_date = DATE.split('_')[0]
    datetime_of_recovery = dt.datetime.strptime(recover_date, "%Y%m%d")

    df_fr = get_related_field_records(recover_date)

    cond1 = df_fr["Upload folder name"]==f"recover-{DATE}"
    if datetime_of_recovery < dt.datetime.strptime('20220715', "%Y%m%d"):
        notes = df_fr.loc[cond1, "Notes"]
    else:
        cond2 =  df_fr["SD card #"]==f"{SD_CARD_NUM}"
        notes = df_fr.loc[cond1&cond2, "Notes"]
    
    if (notes.empty):
        notes = "(Audiomoth notes not found in Field Records)"
    else:
        notes = notes.item()
    
    return notes

def get_audiomoth_name(DATE, SD_CARD_NUM):
    """Gets the location where an AudioMoth was deployed at a certain date using the deployment field records.
    Will be used to plot activity with the right location label so user can tell location of activity by plots.

    Parameters
    ------------
    DATE : `str`
        The date when an AudioMoth was deployed
    SD_CARD_NUM : `str`
        The SD card inside the AudioMoth to identify which AudioMoth the user wants.

    Returns
    ------------
    site_name : `str`
        - Name of the location where the Audiomoth was deployed at that date
        according to the field records.
        - If the deployment is not recorded, site_name will be "(Site not found in Field Records)"
    """


    recover_date = DATE.split('_')[0]
    datetime_of_recovery = dt.datetime.strptime(recover_date, "%Y%m%d")

    df_fr = get_related_field_records(recover_date)

    cond1 = df_fr["Upload folder name"]==f"recover-{DATE}"
    if datetime_of_recovery < dt.datetime.strptime('20220715', "%Y%m%d"):
        audiomoth_name = df_fr.loc[cond1, "AudioMoth #"]
    else:
        cond2 =  df_fr["SD card #"]==f"{SD_CARD_NUM}"
        audiomoth_name = df_fr.loc[cond1&cond2, "AudioMoth #"]
    
    if (audiomoth_name.empty):
        audiomoth_name = "(Audiomoth name not found in Field Records)"
    else:
        audiomoth_name = audiomoth_name.item()
    
    return audiomoth_name


def get_site_name(DATE, SD_CARD_NUM):
    """Gets the location where an AudioMoth was deployed at a certain date using the deployment field records.
    Will be used to plot activity with the right location label so user can tell location of activity by plots.

    Parameters
    ------------
    DATE : `str`
        The date when an AudioMoth was deployed
    SD_CARD_NUM : `str`
        The SD card inside the AudioMoth to identify which AudioMoth the user wants.

    Returns
    ------------
    site_name : `str`
        - Name of the location where the Audiomoth was deployed at that date
        according to the field records.
        - If the deployment is not recorded, site_name will be "(Site not found in Field Records)"
    """


    recover_date = DATE.split('_')[0]
    datetime_of_recovery = dt.datetime.strptime(recover_date, "%Y%m%d")

    df_fr = get_related_field_records(recover_date)

    cond1 = df_fr["Upload folder name"]==f"recover-{DATE}"
    if datetime_of_recovery < dt.datetime.strptime('20220715', "%Y%m%d"):
        site = df_fr.loc[cond1, "Site"]
    else:
        cond2 =  df_fr["SD card #"]==f"{SD_CARD_NUM}"
        site = df_fr.loc[cond1&cond2, "Site"]
    
    if (site.empty):
        site_name = "(Site not found in Field Records)"
    else:
        site_name = site.item()
    
    return site_name

def get_field_records(path_to_records):
    """Extracts .csv field records from given path and converts it to DataFrame object.

    Parameters
    ------------
    path_to_records : `pathlib.Path`
        - Path to the location of .csv file field records

    Returns
    ------------
    fr : `pandas.DataFrame`
        - DataFrame table that matches the information in the .csv file 
        - stored as `repo_root_level/field_records/{.csv}`
    """

    if (path_to_records.is_file()):
        df_fr = pd.read_csv(path_to_records, sep=',') 

        df_fr.columns = df_fr.columns.str.strip()
        for col in df_fr.columns:
            df_fr[col] = df_fr[col].astype(str).str.strip()

        for i in range(len(df_fr["SD card #"].values)):
            df_fr["SD card #"].values[i] = df_fr["SD card #"].values[i].zfill(3)
    else:
        df_fr = pd.DataFrame()

    return df_fr


def parse_args():
    """
    Defines the command line interface for the pipeline.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="the directory of files to process",
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="the directory to write the output to",
        default="output_dir",
    )
    parser.add_argument(
        "csv_name",
        type=str,
        help="the name of the csv that will contain info"
    )

    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()

    cfg = dict()
    cfg["output_dir"] = Path(args["output_directory"])
    cfg["input_dir"] = Path(args["input_dir"])
    cfg["csv_name"] = args["csv_name"]

    files_df = generate_files_df(cfg)