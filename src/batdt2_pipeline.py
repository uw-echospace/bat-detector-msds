import numpy as np
import argparse
import os
import pandas as pd
import soundfile as sf
from maad import sound, util
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors

import datetime as dt
from pathlib import Path

# set python path to correctly use batdetect2 submodule
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src/models/bat_call_detector/batdetect2/"))

from cfg import get_config
from utils.utils import gen_empty_df
from pipeline import pipeline
import models.bat_call_detector.feed_buzz_helper as fbh

def generate_segments(audio_file: Path, output_dir: Path, start_time: float, duration: float):
    """
    Segments audio_file into clips of duration length and saves them to output_dir.
    start_time: seconds
    duration: seconds
    """

    if (os.stat(audio_file).st_size == 0):
        return []
    
    ip_audio = sf.SoundFile(audio_file)

    sampling_rate = ip_audio.samplerate
    # Convert to sampled units
    ip_start = int(start_time * sampling_rate)
    ip_duration = int(duration * sampling_rate)
    ip_end = ip_audio.frames

    output_files = []

    # for the length of the duration, process the audio into duration length clips
    for sub_start in range(ip_start, ip_end, ip_duration):
        sub_end = np.minimum(sub_start + ip_duration, ip_end)

        # For file names, convert back to seconds 
        op_file = os.path.basename(audio_file.name).replace(" ", "_")
        start_seconds =  sub_start / sampling_rate
        end_seconds =  sub_end / sampling_rate
        op_file_en = "__{:.2f}".format(start_seconds) + "_" + "{:.2f}".format(end_seconds)
        op_file = op_file[:-4] + op_file_en + ".wav"
        
        op_path = os.path.join(output_dir, op_file)
        output_files.append({
            "audio_file": op_path, 
            "offset":  start_time + (sub_start/sampling_rate),
        })
        
        if (os.path.exists(op_path) == False):
            sub_length = sub_end - sub_start
            ip_audio.seek(sub_start)
            op_audio = ip_audio.read(sub_length)
            sf.write(op_path, op_audio, sampling_rate, subtype='PCM_16')

    return output_files 


def get_params(output_dir, tmp_dir, num_processes, segment_duration):
    cfg = get_config()
    cfg["output_dir"] = Path(output_dir)
    cfg["tmp_dir"] = Path(tmp_dir)
    cfg["num_processes"] = num_processes
    cfg['segment_duration'] = segment_duration

    return cfg

def get_files_from_dir(input_dir):
    audio_files = []
    for file in sorted(list(Path(input_dir).iterdir())):
        if (file.name.split('.')[1]=="WAV" or file.name.split('.')[1]=="wav"):
            audio_files.append(file)

    return audio_files

def generate_segmented_paths(audio_files, cfg):
    segmented_file_paths = []
    for audio_file in audio_files:
        segmented_file_paths += generate_segments(
            audio_file = audio_file, 
            output_dir = cfg['tmp_dir'],
            start_time = cfg['start_time'],
            duration   = cfg['segment_duration'],
        )
    return segmented_file_paths


## Create necessary mappings from audio to model to file path
def initialize_mappings(necessary_paths, cfg):
    l_for_mapping = [{
        'audio_seg': audio_seg, 
        'model': cfg['models'][0],
        'original_file_name': f"{Path(audio_seg['audio_file']).name[:15]}.WAV",
        } for audio_seg in necessary_paths]

    return l_for_mapping

## Run models and get detections!
def run_models(file_mappings, cfg, csv_name):
    bd_dets = pd.DataFrame()
    for i in tqdm(range(len(file_mappings))):
        cur_seg = file_mappings[i]
        bd_annotations_df = cur_seg['model']._run_batdetect(cur_seg['audio_seg']['audio_file'])
        bd_preds = pipeline._correct_annotation_offsets(
                bd_annotations_df,
                cur_seg['original_file_name'],
                cur_seg['audio_seg']['offset']
            )
        bd_dets = pd.concat([bd_dets, bd_preds])

    bd_dets.to_csv(f"{cfg['output_dir']}/{csv_name}", index=False)

    return bd_dets

def plot_dets_as_activity_grid(input_dir, csv_name, output_dir, site_name, save=True):
    recover_folder = input_dir.split('/')[-2]
    audiomoth_folder = input_dir.split('/')[-1]
    august_dets = pd.read_csv(f'{output_dir}/{csv_name}.csv')
    activity = np.array([])
    activity_times = []
    activity_dates = []

    for file in get_files_from_dir(input_dir):
        filedets = august_dets.loc[august_dets['input_file']==file.name]
        activity = np.hstack([activity, len(filedets)])

        file_dt = dt.datetime.strptime(file.name, "%Y%m%d_%H%M%S.WAV")

        file_time = dt.datetime.strftime(file_dt, "%H:%M")
        if (not(activity_times.__contains__(file_time))):
            activity_times.append(file_time)

        file_date = dt.datetime.strftime(file_dt, "%m/%d/%y")
        if (not(activity_dates.__contains__(file_date))):
            activity_dates.append(file_date)
            activity_for_date = np.array([])

    activity = activity.reshape((len(activity_dates), len(activity_times))).T

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(12, 8))
    plt.title(f"Activity from {site_name}", loc='left', y=1.05)
    plt.imshow(activity+1, norm=colors.LogNorm(vmin=1, vmax=10e3))
    plt.yticks(np.arange(0, len(activity_times), 2)-0.5, activity_times[::2])
    plt.xticks(np.arange(0, len(activity_dates))-0.5, activity_dates, rotation=60)
    plt.colorbar()
    if save:
        plt.savefig(f"{output_dir}/activity_{recover_folder}_{audiomoth_folder}.png", bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def delete_segments(necessary_paths):
    for path in necessary_paths:
        os.remove(path['audio_file'])

def run_pipeline(input_dir, csv_name, output_dir, tmp_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    cfg = get_params(output_dir, tmp_dir, 4, 30.0)
    audio_files = get_files_from_dir(input_dir)
    segmented_file_paths = generate_segmented_paths(audio_files, cfg)
    file_path_mappings = initialize_mappings(segmented_file_paths, cfg)
    bd_dets = run_models(file_path_mappings, cfg, csv_name)
    delete_segments(segmented_file_paths)

    recover_folder = input_dir.split('/')[-2]
    recover_date = recover_folder.split('-')[1]
    audiomoth_folder = input_dir.split('/')[-1]
    audiomoth_unit = audiomoth_folder.split('_')[-1]
    field_records = get_field_records(Path("ubna_2022b.csv"))
    site_name = get_site_name(field_records, recover_date, audiomoth_unit)
    print(f"Looking at data from {site_name}...")
    plot_dets_as_activity_grid(input_dir, csv_name, output_dir, site_name, save=True)

    return bd_dets

def get_field_records(path_to_records):
    """Extracts .csv field records from given path and converts it to DataFrame object.

    Parameters
    ------------
    path_to_records : `pathlib.Path`
        - Path to the location of .csv file field records

    Returns
    ------------
    fr : `pandas.DataFrame`
        - DataFrame table that matches the information in the .md file 
        - stored as `repo_root_level/field_records/ubna_2022b.md`
    """

    if (path_to_records.is_file()):
        df_fr = pd.read_csv(path_to_records, sep=',') 

    return df_fr


def get_site_name(df_fr, DATE, SD_CARD_NUM):
    """Gets the location where an AudioMoth was deployed at a certain date
    using the deployment field records.

    Parameters
    ------------
    df_fr : `pandas.DataFrame`
        - DataFrame table that matches the information in the .md file 
        - stored as `repo_root_level/field_records/ubna_2022b.md`
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

    cond1 = df_fr["Upload folder name"]==f"recover-{DATE}"
    cond2 =  df_fr["SD card #"]==f"{SD_CARD_NUM}"
    site = df_fr.loc[cond1&cond2, "Site"]
    
    if (site.empty):
        site_name = "(Site not found in Field Records)"
    else:
        site_name = site.item()
    
    return site_name

def parse_args():
    """
    Defines the command line interface for the pipeline.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="the directory of WAV files to process",
    )
    parser.add_argument(
        "csv_filename",
        type=str,
        help="the file name of the .csv file",
        default="output.csv",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="the directory where the .csv file goes",
        default="output_dir",
    )
    parser.add_argument(
        "temp_dir",
        type=str,
        help="the temp directory where the audio segments go",
        default="output/tmp",
    )

    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()

    run_pipeline(args['input_dir'], args['csv_filename'], args['output_dir'], args['temp_dir'])