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

import exiftool

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
    Segments audio file into clips of duration length and saves them to output/tmp folder.
    Allows detection model to be run on segments instead of entire file as recommended.
    These segments will be deleted from the output/tmp folder after detections have been generated.

    Parameters
    ------------
    audio_file : `pathlib.Path`
        - The path to an audio_file from the input directory provided in the command line
    output_dir : `pathlib.Path`
        - The path to the tmp folder that saves all of our segments.
    start_time : `float`
        - The time at which the segments will start being generated from within the audio file
    duration : `float`
        - The duration of all segments generated from the audio file.

    Returns
    ------------
    output_files : `List`
        - The path (a str) to each generated segment of the given audio file will be stored in this list.
        - The offset of each generated segment of the given audio file will be stored in this list.
        - Both items are stored in a dict{} for each generated segment.
    """
    
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
    """
    Gets model params using separate method stored in `src/cfg.py`

    Parameters
    ------------
    output_dir : `str`
        - The path to the directory where outputs of the pipeline will be saved.
    tmp_dir : `str`
        - The path to the directory where segments of all audio files from the input directory will be saved.
    num_processes : `int`
        - The number of processes the user can set to run this pipeline on
    segment_duration : `float`
        - The duration of all segments generated from all audio files.

    Returns
    ------------
    cfg : `dict`
        - A dictionary of all model parameters and pipeline parameters.
    """

    cfg = get_config()
    cfg["output_dir"] = Path(output_dir)
    cfg["tmp_dir"] = Path(tmp_dir)
    cfg["num_processes"] = num_processes
    cfg['segment_duration'] = segment_duration

    return cfg

def get_dates_of_deployment(input_dir):
    dates = set()
    for filepath in list(Path(input_dir).iterdir()):
        filename = filepath.name
        if (os.path.exists(filepath) and len(filename.split('.')) == 2 and 
            (filename.split('.')[1]=="WAV" or filename.split('.')[1]=="wav")):
            file_dt = dt.datetime.strptime(filename, "%Y%m%d_%H%M%S.WAV")
            dates.add(dt.datetime.strftime(file_dt, '%Y%m%d'))
        
    dates = sorted(list(dates))
    return dates

def get_recording_period(input_dir):
    config_path = f'{input_dir}/CONFIG.TXT'
    if (os.path.isfile(config_path)):
        config_details = pd.read_csv(config_path, header=0, index_col=0, sep=" : ", engine='python').transpose()
        config_details.columns = config_details.columns.str.strip()
        recording_period = config_details['Recording period 1'].values[0]
        period_tokens = recording_period.split(' ')
    else:
        period_tokens = ["00:00", "-", "24:00"]

    return period_tokens[0], period_tokens[2]

def get_files_for_pipeline(reference_filepaths):
    """
    Gets a list of audio files existing in an input directory to feed into the pipeline.

    Parameters
    ------------
    input_dir : `str`
        - The provided path to a directory consisting of audio files the user wants to feed into our pipeline.

    Returns
    ------------
    good_audio_files : `List`
        - A list of pathlib.Path objects to all usable audio files existing in input_dir
        - Files are checked as candidates if they first exist and are not empty.
        - Then they must be .wav or .WAV files, as those are the files recorded by Audiomoths
        - Then, files are added to a set if they are starting at 30min 0secs or 0min 0secs for every hour:
            - This is to avoid stating that we detected X amount of detections for a file whose duration is not 30 mins.
        - This set is finally filtered using exiftool comments to find files with no Audiomoth error.
    """

    audio_files = []
    good_audio_files = []
    for file in reference_filepaths:
        if (os.path.exists(file) and not(os.stat(file).st_size == 0)):
            audio_files.append(file)

    comments = exiftool.ExifToolHelper().get_tags(audio_files, tags='RIFF:Comment')
    df_comments = pd.DataFrame(comments)
    print(f"There are {len(audio_files)} audio files that passed 1st level of filtering!")
    good_audio_files = df_comments.loc[~df_comments['RIFF:Comment'].str.contains("microphone")]['SourceFile'].values

    for i in range(len(good_audio_files)):
        good_audio_files[i] = Path(good_audio_files[i])

    print(f"There are {len(good_audio_files)} audio files that passed 2nd level of filtering!")
                
    return good_audio_files

def get_files_to_reference(input_dir, dates, start_time, end_time):
    """
    Gets a list of audio files existing in an input directory representative of the times recorded each day.

    Parameters
    ------------
    input_dir : `str`
        - The provided path to a directory consisting of audio files the user wants to feed into our pipeline.

    Returns
    ------------
    audio_files : `List`
        - A list of pathlib.Path objects to all usable audio files existing in input_dir
        - Files are checked as candidates if they just exist.
        - Then they must be .wav or .WAV files, as those are the files recorded by Audiomoths
        - Then, files are added to a set if they are starting at 30min 0secs or 0min 0secs for every hour:
            - This is to avoid stating that we detected X amount of detections for a file whose duration is not 30 mins.
        - Files are not filtered for emptiness or error as we just want the filenames for time reference.
    """

    reference_filepaths = []
    for date in dates:
        start_dt = dt.datetime.strptime(f'{date}_{start_time}:00', "%Y%m%d_%H:%M:%S")
        if (end_time == '24:00'):
            end_time = '23:59'
        end_dt = dt.datetime.strptime(f'{date}_{end_time}:00', "%Y%m%d_%H:%M:%S")

        cur_dt = start_dt
        while cur_dt < end_dt:
            filepath = f'{input_dir}/{dt.datetime.strftime(cur_dt, "%Y%m%d_%H%M%S.WAV")}'
            if (os.path.exists(filepath)):
                reference_filepaths += [Path(filepath)]

            if (cur_dt.minute < 30):
                new_min = str(cur_dt.minute + 30).zfill(2)
                new_hour = str(cur_dt.hour).zfill(2)
            else:
                new_min = str(cur_dt.minute - 30).zfill(2)
                new_hour = str(cur_dt.hour + 1).zfill(2)
            cur_time = f'{new_hour}:{new_min}'
            if (cur_time == '24:00'):
                cur_time = '23:59'
            cur_dt = dt.datetime.strptime(f'{date}_{cur_time}:00', "%Y%m%d_%H:%M:%S")

    return reference_filepaths

def generate_segmented_paths(audio_files, cfg):
    """
    Generates and returns a list of segments using provided cfg parameters for each audio file in audio_files.

    Parameters
    ------------
    audio_files : `List`
        - List of pathlib.Path objects of the paths to each audio file in the provided input directory.
    cfg : `dict`
        - A dictionary of pipeline parameters:
        - tmp_dir is the directory where segments will be stored
        - start_time is the time at which segments are generated from each audio file.
        - segment_duration is the duration of each generated segment

    Returns
    ------------
    segmented_file_paths : `List`
        - A list of dictionaries related to every generated segment.
        - Each dictionary stores a generated segment's path in the tmp_dir and offset in the original audio file.
    """

    segmented_file_paths = []
    for audio_file in audio_files:
        segmented_file_paths += generate_segments(
            audio_file = audio_file, 
            output_dir = cfg['tmp_dir'],
            start_time = cfg['start_time'],
            duration   = cfg['segment_duration'],
        )
    return segmented_file_paths


def initialize_mappings(necessary_paths, cfg):
    """
    Generates and returns a list of mappings using provided cfg parameters for each audio segment in the provided necessary paths.

    Parameters
    ------------
    necessary_paths : `List`
        - List of dictionaries generated by generate_segmented_paths()
    cfg : `dict`
        - A dictionary of pipeline parameters:
        - models is the models in the pipeline that are being used.

    Returns
    ------------
    l_for_mapping : `List`
        - A list of dictionaries related to every generated segment with more pipeline details.
        - Each dictionary stores the prior segmented_path dict{}, the model to apply, and the original file name of the segment.
    """

    l_for_mapping = [{
        'audio_seg': audio_seg, 
        'model': cfg['models'][0],
        'original_file_name': f"{Path(audio_seg['audio_file']).name.split('__')[0]}.WAV",
        } for audio_seg in necessary_paths]

    return l_for_mapping

def run_models(file_mappings, cfg, csv_name):
    """
    Runs the batdetect2 model to detect bat search-phase calls in the provided audio segments and saves detections into a .csv.

    Parameters
    ------------
    file_mappings : `List`
        - List of dictionaries generated by initialize_mappings()
    cfg : `dict`
        - A dictionary of pipeline parameters:
        - models is the models in the pipeline that are being used.
    csv_name : `str`
        - The detections of bat search-phase calls in each audio file existing in the provided input directory.
        - Stored as "batdetect2_pipeline__recover-DATE_UBNA_###.csv"

    Returns
    ------------
    bd_dets : `pandas.DataFrame`
        - A DataFrame of detections that will also be saved in the provided output_dir under the above csv_name
        - 7 columns in this DataFrame: start_time, end_time, low_freq, high_freq, detection_confidence, event, input_file
        - Detections are always specified w.r.t their input_file; earliest start_time can be 0 and latest end_time can be 1795.
        - Events are always "Echolocation" as we are using a model that only detects search-phase calls.
    """

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

def plot_dets_as_activity_grid(input_dir, csv_name, output_dir, site_name, show_PST=False, save=True):
    """
    Plots the detections generated from giving an input_dir, output_dir, and csv_name in an activity grid format

    Parameters
    ------------
    input_dir : `str`
        - The path to the input directory containing the files that are linked to the desired detections we wish to plot.
    csv_name : `str`
        - The detections of bat search-phase calls in each audio file existing in the provided input directory.
        - Stored as "batdetect2_pipeline__recover-DATE_UBNA_###.csv"
    output_dir : `str`
        - The path to the output directory where the detections.csv will be saved as well as the newly generated plot
    site_name : `str`
        - The name of the site where the recordings in the input_dir were recorded from.
    show_PST : `boolean`
        - A flag whether user wants to time in PST instead of UTC.
        - For example, today's 03:00 UTC will become yesterday's 20:00 PST (-7 hrs)
    save : `boolean`
        - A flag whether user wants to save the plot under "output_dir/recover-DATE/UBNA_###" similar to how recordings are stored in our hard drive.

    Returns
    ------------
    A plot with the following details:
        - y-axis corresponding to the time of day ranging, for example, from 03:00 to 13:00 UTC.
        - x-axis corresponding to the days of activity ranging, for example, from 2023-06-10 to 2023-06-15
        - Cell intensity corresponding to the number of detections per 30-min of each day
            - Intensity actually is (number of detections + 1) so 0 detections is represented as lowest color on colorbar
            - Recordings where the Audiomoth experienced errors are colored red.
    """

    recover_folder = input_dir.split('/')[-2]
    audiomoth_folder = input_dir.split('/')[-1]
    dets = pd.read_csv(f'{output_dir}/{csv_name}')
    start_time, end_time = get_recording_period(input_dir)
    dates = get_dates_of_deployment(input_dir)
    ref_audio_files = get_files_to_reference(input_dir, dates, start_time, end_time)
    good_audio_files = get_files_for_pipeline(ref_audio_files)
    activity = np.array([])
    activity_times = []
    activity_dates = []

    for file in ref_audio_files:
        filedets = dets.loc[dets['input_file']==(file).name]
        if file in good_audio_files:
            activity = np.hstack([activity, len(filedets) + 1])
        else:
            activity = np.hstack([activity, 0])

        file_dt_UTC = dt.datetime.strptime((file).name, "%Y%m%d_%H%M%S.WAV")

        if show_PST:
            if (file_dt_UTC.hour >= 7):
                file_dt_PST = dt.datetime.strptime(f"{file_dt_UTC.date()}_{str(file_dt_UTC.hour - 7).zfill(2)}{file_dt_UTC.minute}", "%Y-%m-%d_%H%M")
                file_time_PST = dt.datetime.strftime(file_dt_PST, "%H:%M")
            else:
                file_dt_PST = dt.datetime.strptime(f"{file_dt_UTC.date()}_{str(24 + file_dt_UTC.hour - 7).zfill(2)}{file_dt_UTC.minute}", "%Y-%m-%d_%H%M")
                file_time_PST = dt.datetime.strftime(file_dt_PST, "%H:%M")
            if (not(activity_times.__contains__(file_time_PST))):
                activity_times.append(file_time_PST)
        else:
            file_time_UTC = dt.datetime.strftime(file_dt_UTC, "%H:%M")
            if (not(activity_times.__contains__(file_time_UTC))):
                activity_times.append(file_time_UTC)

        file_date = dt.datetime.strftime(file_dt_UTC, "%m/%d/%y")
        if (not(activity_dates.__contains__(file_date))):
            activity_dates.append(file_date)
    
    activity = activity.reshape((len(activity_dates), len(activity_times))).T

    masked_array_for_nodets = np.ma.masked_where(activity==0, activity)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='red')

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(12, 8))
    plt.title(f"Activity from {site_name}", loc='left', y=1.05)
    plt.imshow(masked_array_for_nodets, cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=10e3))
    plt.yticks(np.arange(0, len(activity_times), 2)-0.5, activity_times[::2], rotation=50)
    plt.xticks(np.arange(0, len(activity_dates))-0.5, activity_dates, rotation=50)
    plt.ylabel('UTC Time (HH:MM)')
    if show_PST:
        plt.ylabel('PST Time (HH:MM)')
    plt.xlabel('Date (MM/DD/YY)')
    plt.colorbar()
    if save:
        plt.savefig(f"{output_dir}/activity_{recover_folder}_{audiomoth_folder}.png", bbox_inches='tight', pad_inches=0.5)
    plt.tight_layout()
    plt.show()

def delete_segments(necessary_paths):
    """
    Deletes the segments whose paths are stored in necessary_paths

    Parameters
    ------------
    necessary_paths : `List`
        - A list of dictionaries generated from generate_segmented_paths()
    """

    for path in necessary_paths:
        os.remove(path['audio_file'])

def run_pipeline(input_dir, csv_name, output_dir, tmp_dir, run_model=True, generate_fig=True):
    """Runs the batdetect2 pipeline on provided directory of audio files and saves detections and activity plot in output directory

    Parameters
    ------------
    input_dir : `str`
        - String-based path to the directory of audio files which our pipeline will be fed to generate detections
    csv_name : `str`
        - The name of the csv that our detections generated from the audio files in input_dir will be saved in.
    output_dir : `str`
        - String-based path to the directory that will store the outputs: detections and the plot
    tmp_dir : `str`
        - String-based path to the directory that will temporarily store our generated segments to feed into batdetect2
    run_model : `boolean`
        - A flag for whether the user wants to run the batdetect2 model to generate detections
        - Can be false if user wants to just generate / update plot
    generate_fig : `boolean`
        - A flag for whether the user wants to generate the activty plot from a saved .csv corresponding to the input_dir
        - Can be false if user wants to just generate detections knowing the figure will not be up-to-date    

    Returns
    ------------
    bd_dets : `pandas.DataFrame`
        - A DataFrame of detections that will also be saved in the provided output_dir under the above csv_name
        - 7 columns in this DataFrame: start_time, end_time, low_freq, high_freq, detection_confidence, event, input_file
        - Detections are always specified w.r.t their input_file; earliest start_time can be 0 and latest end_time can be 1795.
        - Events are always "Echolocation" as we are using a model that only detects search-phase calls.
    """
    recover_folder = input_dir.split('/')[-2]
    recover_date = recover_folder.split('-')[1]
    audiomoth_folder = input_dir.split('/')[-1]
    audiomoth_unit = audiomoth_folder.split('_')[-1]
    if str(dt.datetime.strptime(recover_date, "%Y%m%d").year) == "2022":
        field_records = get_field_records(Path(f"{os.path.dirname(__file__)}/../field_records/ubna_2022b.csv"))
    if str(dt.datetime.strptime(recover_date, "%Y%m%d").year) == "2023":
        field_records = get_field_records(Path(f"{os.path.dirname(__file__)}/../field_records/ubna_2023.csv"))
    site_name = get_site_name(field_records, recover_date, audiomoth_unit)
    print(f"Looking at data from {site_name}...")
    if site_name != "(Site not found in Field Records)":
        output_dir = f'{output_dir}/{site_name}'
    else:
        output_dir = f'{output_dir}/{audiomoth_folder}'

    bd_dets = pd.DataFrame()

    if (run_model == "true"):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        cfg = get_params(output_dir, tmp_dir, 4, 30.0)
        start_time, end_time = get_recording_period(input_dir)
        dates = get_dates_of_deployment(input_dir)
        ref_audio_files = get_files_to_reference(input_dir, dates, start_time, end_time)
        good_audio_files = get_files_for_pipeline(ref_audio_files)
        print(f"There are {len(good_audio_files)} usable files out of {len(list(Path(input_dir).iterdir()))} total files")
        segmented_file_paths = generate_segmented_paths(good_audio_files, cfg)
        file_path_mappings = initialize_mappings(segmented_file_paths, cfg)
        bd_dets = run_models(file_path_mappings, cfg, csv_name)
        delete_segments(segmented_file_paths)

    if (generate_fig == "true"):
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


def get_site_name(df_fr, DATE, SD_CARD_NUM):
    """Gets the location where an AudioMoth was deployed at a certain date using the deployment field records.
    Will be used to plot activity with the right location label so user can tell location of activity by plots.

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
    parser.add_argument(
        "run_model",
        type=str,
        help="Do you want to run the model? As opposed to just generating the figure",
        default="true",
    )
    parser.add_argument(
        "generate_fig",
        type=str,
        help="Do you want to generate and save a corresponding summary figure?",
        default="true",
    )

    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()

    run_pipeline(args['input_dir'], args['csv_filename'], args['output_dir'], args['temp_dir'], args['run_model'], args['generate_fig'])