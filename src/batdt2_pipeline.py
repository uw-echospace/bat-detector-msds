import numpy as np
import argparse
import pandas as pd
import dask.dataframe as dd
import soundfile as sf
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import datetime as dt
from pathlib import Path
from torch import multiprocessing

import exiftool

# set python path to correctly use batdetect2 submodule
import sys
sys.path.append(str(Path.cwd()))
sys.path.append(str(Path.joinpath(Path.cwd(), "src/models/bat_call_detector/batdetect2/")))

from cfg import get_config
from pipeline import pipeline
from utils.utils import gen_empty_df, convert_df_ravenpro

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
        op_file = Path(audio_file).name.replace(" ", "_")
        start_seconds =  sub_start / sampling_rate
        end_seconds =  sub_end / sampling_rate
        op_file_en = "__{:.2f}".format(start_seconds) + "_" + "{:.2f}".format(end_seconds)
        op_file = op_file[:-4] + op_file_en + ".wav"
        
        op_path = Path.joinpath(Path(output_dir), Path(op_file))
        output_files.append({
            "audio_file": op_path, 
            "offset":  start_time + (sub_start/sampling_rate),
        })
        
        if (not(Path(op_path).exists())):
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
    """
    Gets dates which Audiomoth recorded from Audiomoth deployment directory.

    Parameters
    ------------
    input_dir : `str`
        - The directory of Audiomoth recordings + CONFIG.TXT corresponding to a deployment session.

    Returns
    ------------
    dates_from_dir : `List`
        - List of file-related dates in format "YYYYMMDD".
    """
    
    datetimes_from_dir = pd.to_datetime(list(Path(input_dir).iterdir()), format="%Y%m%d_%H%M%S.WAV", errors='coerce', exact=False).dropna()
    dates_from_dir = sorted(datetimes_from_dir.strftime("%Y%m%d").unique())
    return dates_from_dir

def get_recording_period(input_dir):
    """
    Gets configured recording period of Audiomoth over a deployment session using CONFIG.TXT.

    Parameters
    ------------
    input_dir : `str`
        - The directory of Audiomoth recordings + CONFIG.TXT corresponding to a deployment session.

    Returns
    ------------
    start_time : `str`
        - The start time of the configured recording period.
    end_time : `str`
        - The end time of the configured recording period.
    """

    config_path = f'{input_dir}/CONFIG.TXT'
    if (Path(config_path).is_file()):
        config_details = pd.read_csv(config_path, header=0, index_col=0, sep=" : ", engine='python').transpose()
        config_details.columns = config_details.columns.str.strip()
        recording_period = config_details['Recording period 1'].values[0]
        period_tokens = recording_period.split(' ')
    else:
        period_tokens = ["00:00", "-", "23:59"]
    start_time = period_tokens[0]
    end_time = period_tokens[2]
    if end_time == "24:00":
        end_time = "23:59"

    return start_time, end_time

def get_files_for_pipeline(reference_filepaths):
    """
    Filters out a list of audio files that are good to be fed into the MSDS pipeline

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
        - This set is finally filtered using exiftool comments to find files with no Audiomoth error.
    """

    audio_files = []
    good_audio_files = []
    for file in reference_filepaths:
        if (file.exists() and not(file.stat().st_size == 0)):
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
    Assembles a list of audio files existing in the input directory that should be error-free and contain calls.

    Parameters
    ------------
    input_dir : `str`
        - The provided path to a directory consisting of audio files the user wants to feed into our pipeline.

    Returns
    ------------
    ref_filepaths : `List`
        - A list of pathlib.Path objects to all usable audio files existing in input_dir
        - Audio .WAV files are artificially assembled using CONFIG.TXT information: these files should exist in our directory.
        - Files are not filtered for emptiness or error as we just want the filenames for time reference.
    """

    all_dates = pd.Index([])
    for date in dates:
        date_range = pd.date_range(dt.datetime.strptime(f'{date}_{start_time}', "%Y%m%d_%H:%M"), dt.datetime.strptime(f'{date}_{end_time}', "%Y%m%d_%H:%M"), freq="30T", inclusive='left')
        all_dates = all_dates.append(date_range)
    all_filenames = all_dates.strftime("%Y%m%d_%H%M%S.WAV")

    ref_filepaths = []
    for file in all_filenames:
        ref_filepath = Path(f"{input_dir}/{file}")
        if (ref_filepath.exists()):
            ref_filepaths += [ref_filepath]

    return ref_filepaths

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

def run_models(file_mappings):
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

    return bd_dets

def apply_model(file_mapping):
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

    bd_dets = file_mapping['model']._run_batdetect(file_mapping['audio_seg']['audio_file'])
    return pipeline._correct_annotation_offsets(
                bd_dets,
                file_mapping['original_file_name'],
                file_mapping['audio_seg']['offset']
            )

def _generate_csv(annotation_df, model_name, audio_file_name, output_path, should_csv):
    file_name = f"{model_name}__{audio_file_name}"
    extension = ".csv"
    sep = ","

    if not should_csv:
        extension = ".txt"
        sep = "\t"
        annotation_df = convert_df_ravenpro(annotation_df)

    csv_path = output_path / f"{file_name}{extension}"
    annotation_df.to_csv(csv_path, sep=sep, index=False)
    return csv_path

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

def construct_activity_grid(csv_name, ref_audio_files, good_audio_files, output_dir, show_PST=False):
    """
    Constructs DataFrames corresponding to different important ways of storing activity for a deployment session.
    plot_df is an activity grid with date headers and time indices and number of detections as values.
    activity_df is a 2-column dataframe with datetime indices and corresponding number of calls detected for resampling purposes.

    Parameters
    ------------
    csv_name : `str`
        - The detections of bat search-phase calls in each audio file existing in the provided input directory.
        - Stored as "bd2__recover-DATE_UBNA_###.csv"
    ref_audio_files : `List`
        - A list of audio files that should be recorded by the Audiomoth representing the times that were recorded.
    good_audio_files : `List`
        - A list of audio files that are error-free and were fed into the MSDS pipeline.
    output_dir : `str`
        - The path to the output directory where the activity grid and the 2-column activity .csv files will be saved.
    show_PST : `boolean`
        - A flag whether user wants to time in PST instead of UTC.
        - For example, today's 03:00 UTC will become yesterday's 20:00 PST (-7 hrs)

    Returns
    ------------
    plot_df : `pd.DataFrame`
        - Rows corresponding to the time of day ranging, for example, from 03:00 to 13:00 UTC.
        - Columns corresponding to the days of activity ranging, for example, from 2023-06-10 to 2023-06-15
        - Cell value corresponding to the number of detections per 30-min of each day
            - Values are 0 for error-files, 1 for call-absence, number of detections otherwise.
            - Recordings where the Audiomoth experienced errors are colored red.
    """
    csv_tag = csv_name.split('__')[-1]

    activity_datetimes_for_file = pd.to_datetime(ref_audio_files, format="%Y%m%d_%H%M%S.WAV", exact=False).tz_localize('UTC')
    activity_datetimes_for_plot = activity_datetimes_for_file
    if show_PST:
        activity_datetimes_for_plot = activity_datetimes_for_plot.tz_convert(tz='US/Pacific')

    activity_times = (activity_datetimes_for_plot).strftime("%H:%M").unique()
    activity_dates = (activity_datetimes_for_plot).strftime("%m/%d/%y").unique()

    dets = pd.read_csv(f'{output_dir}/{csv_name}')
    dets_per_file = dets.groupby(['input_file'])['input_file'].count()

    activity = []
    for ref_file in ref_audio_files:
        file = ref_file.name
        if ref_file in good_audio_files:
            if (file in dets_per_file.index):
                activity += [dets_per_file[file]]
            else:
                activity += [1]
        else:
            activity += [0]
    activity = np.array(activity)

    activity_df = pd.DataFrame(list(zip(activity_datetimes_for_file, activity)), columns=["date_and_time_UTC", "num_of_detections"])
    activity_df.to_csv(f"{output_dir}/activity__{csv_tag}")
    
    activity = activity.reshape((len(activity_dates), len(activity_times))).T

    plot_df = pd.DataFrame(activity, index=activity_times, columns=activity_dates)
    plot_df.to_csv(f"{output_dir}/activity_plot__{csv_tag}")

    return plot_df


def plot_activity_grid(plot_df, output_dir, recover_folder, audiomoth_folder, site_name, show_PST=False, save=True):
    """
    Plots the above-returned plot_df DataFrame that represents activity over a deployment session.

    Parameters
    ------------
    plot_df : `pd.DataFrame`
        - Rows corresponding to the time of day ranging, for example, from 03:00 to 13:00 UTC.
        - Columns corresponding to the days of activity ranging, for example, from 2023-06-10 to 2023-06-15
        - Cell value corresponding to the number of detections per 30-min of each day
    output_dir : `str`
        - The path to the output directory where the activity grid and the 2-column activity .csv files will be saved.
    recover_folder : `str`
        - The name of the recover folder for the input deployment directory: recover-DATE
    audiomoth_folder : `str`
        - The name of the audiomoth SD card # folder for the input deployment directory: UBNA_###
    site_name: `str`
        - The location where the Audiomoth was deployed; Found using the field records.
    show_PST : `boolean`
        - A flag whether user wants to time in PST instead of UTC.
        - For example, today's 03:00 UTC will become yesterday's 20:00 PST (-7 hrs)
    """

    masked_array_for_nodets = np.ma.masked_where(plot_df.values==0, plot_df.values)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='red', alpha=1.0)

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(12, 8))
    plt.title(f"Activity from {site_name}", loc='left', y=1.05)
    plt.imshow(masked_array_for_nodets, cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=10e3))
    plt.yticks(np.arange(0, len(plot_df.index), 2)-0.5, plot_df.index[::2], rotation=30)
    plt.xticks(np.arange(0, len(plot_df.columns))-0.5, plot_df.columns, rotation=30)
    plt.ylabel('UTC Time (HH:MM)')
    if show_PST:
        plt.ylabel('PST Time (HH:MM)')
    plt.xlabel('Date (MM/DD/YY)')
    plt.colorbar()
    if save:
        plt.savefig(f"{output_dir}/activity_plot__{recover_folder}_{audiomoth_folder}.png", bbox_inches='tight', pad_inches=0.5)
    plt.tight_layout()
    plt.show()

def construct_cumulative_activity(output_dir, site, resample_tag):
    """
    Constructs a cumulative appended DataFrame grid using dask.dataframe.
    This DataFrame gathers all detected activity contained in output_dir for a given site.

    Parameters
    ------------
    output_dir : `str`
        - The output directory that will save the cumulative dataframes.
    site : `str`
        - The site we wish to assemble all activity from.
    resample_tag : `str`
        - The resample_tag associated with resampling: choose above 30T like 1H, 2H or D.

    Returns
    ------------
    activity_df : `pd.DataFrame`
        - Rows corresponding to the time of day ranging, for example, from 03:00 to 13:00 UTC.
        - Columns corresponding to the days of activity ranging, for example, from 2023-06-10 to 2023-06-15
        - Cell value corresponding to the number of detections per 30-min of each day
            - Values are 0 for error-files, 1 for call-absence, number of detections otherwise.
            - Recordings where the Audiomoth experienced errors are colored red.
    """

    new_df = dd.read_csv(f"{Path(__file__).parent}/../output_dir/recover-2023*/{site}/activity__*.csv").compute()
    new_df["date_and_time_UTC"] = pd.to_datetime(new_df["date_and_time_UTC"], format="%Y-%m-%d %H:%M:%S%z")
    new_df.pop(new_df.columns[0])
    new_df = new_df.replace(0, -1)

    resampled_df = new_df.resample(resample_tag, on="date_and_time_UTC").sum()
    selected_time_df = resampled_df.replace(0, np.nan)
    selected_time_df = selected_time_df.dropna()
    selected_time_df = selected_time_df.replace(-1, 0)

    dt_hourmin_info = sorted(((selected_time_df.index).strftime("%H:%M")).unique())
    dates = (pd.date_range(selected_time_df.index[0], selected_time_df.index[-1], freq="D")).strftime("%m-%d-%y")
    activity = (selected_time_df["num_of_detections"].values).reshape(len(dates), len(dt_hourmin_info)).T

    activity_df = pd.DataFrame(activity, index=dt_hourmin_info, columns=dates)
    activity_df.to_csv(f'{Path(__file__).parent}/../output_dir/cumulative_plots/cumulative_activity__{site.split()[0]}_{resample_tag}.csv')

    return activity_df

def plot_cumulative_activity(activity_df, output_dir, site, resample_tag):
    """
    Plots the cumulative appended DataFrame grid of all detected activity a given site.

    Parameters
    ------------
    activity_df : `pd.DataFrame`
        - Rows corresponding to the time of day ranging, for example, from 03:00 to 13:00 UTC.
        - Columns corresponding to the days of activity ranging, for example, from 2023-06-10 to 2023-06-15
        - Cell value corresponding to the number of detections per 30-min of each day
    output_dir : `str`
        - The output directory that will save the cumulative dataframes.
    site : `str`
        - The site we wish to assemble all activity from.
    resample_tag : `str`
        - The resample_tag associated with resampling: choose above 30T like 1H, 2H or D.
    """

    masked_array_for_nodets = np.ma.masked_where(activity_df.values==0, activity_df.values)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='red')

    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(24, 10))
    plt.title(f"Activity from {site}", loc='center', y=1.05)
    plt.imshow(masked_array_for_nodets, cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=10e3))
    plt.yticks(np.arange(0, len(activity_df.index))-0.5, activity_df.index, rotation=45)
    plt.xticks(np.arange(0, len(activity_df.columns))-0.5, activity_df.columns, rotation=45)
    plt.ylabel('UTC Time (HH:MM)')
    plt.xlabel('Date (MM-DD-YY)')
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(f'{Path(__file__).parent}/../output_dir/cumulative_plots/cumulative_activity__{site.split()[0]}_{resample_tag}.png')
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
        Path(path['audio_file']).unlink(missing_ok=False)

def run_pipeline(args):
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

    if Path(args['input_dir']).is_dir():
        input_dir = args['input_dir']
        recover_folder = input_dir.split('/')[-2]
        recover_date = recover_folder.split('-')[1]
        audiomoth_folder = input_dir.split('/')[-1]
        audiomoth_unit = audiomoth_folder.split('_')[-1]
        start_time, end_time = get_recording_period(input_dir)
        dates_from_dir = get_dates_of_deployment(input_dir)
        ref_audio_files = get_files_to_reference(input_dir, dates_from_dir, start_time, end_time)
        good_audio_files = get_files_for_pipeline(ref_audio_files)
        print(f"There are {len(good_audio_files)} usable files out of {len(list(Path(input_dir).iterdir()))} total files")
    if Path(args['input_dir']).is_file():
        input_filename = args['input_dir'].split('/')[-1]
        recover_folder = args['input_dir'].split('/')[-3]
        recover_date = recover_folder.split('-')[1]
        audiomoth_folder = args['input_dir'].split('/')[-2]
        audiomoth_unit = audiomoth_folder.split('_')[-1]
        good_audio_files = [args['input_dir']]

    if str(dt.datetime.strptime(recover_date, "%Y%m%d").year) == "2022":
        field_records = get_field_records(Path(f"{Path(__file__).parent}/../field_records/ubna_2022b.csv"))
    if str(dt.datetime.strptime(recover_date, "%Y%m%d").year) == "2023":
        field_records = get_field_records(Path(f"{Path(__file__).parent}/../field_records/ubna_2023.csv"))
    site_name = get_site_name(field_records, recover_date, audiomoth_unit)
    print(f"Looking at data from {site_name}...")
    if site_name != "(Site not found in Field Records)":
        output_dir = f'{args["output_dir"]}/{site_name}'
    else:
        output_dir = f'{args["output_dir"]}/{audiomoth_folder}'
    if not Path(output_dir).is_dir():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not Path(args['temp_dir']).is_dir():
        Path(args['temp_dir']).mkdir(parents=True, exist_ok=True)

    bd_dets = pd.DataFrame()

    if (args['run_model']):
        cfg = get_params(output_dir, args['temp_dir'], 4, 30.0)
        segmented_file_paths = generate_segmented_paths(good_audio_files, cfg)
        process_pool = multiprocessing.Pool(cfg['num_processes'])
        file_path_mappings = initialize_mappings(segmented_file_paths, cfg)
        # bd_dets = run_models(file_path_mappings, cfg, args['csv_filename'])
        bd_dets = tqdm(
            process_pool.imap(apply_model, file_path_mappings, chunksize=10), 
            desc=f"Applying BatDetect2",
            total=len(file_path_mappings),
        )
        agg_df = gen_empty_df() 
        agg_df = pd.concat(bd_dets, ignore_index=True)
        if (Path(args['input_dir']).is_file()):
            date_of_file = input_filename.split('.')[0].split('_')[0]
            time_of_file = input_filename.split('.')[0].split('_')[-1]
            _generate_csv(agg_df, "bd2", f"{audiomoth_folder}_{date_of_file}_{time_of_file}", Path(output_dir), args['csv'])
        if (Path(args['input_dir']).is_dir()):
            _generate_csv(agg_df, "bd2", f"{recover_folder}_{audiomoth_folder}", Path(output_dir), args['csv'])
        delete_segments(segmented_file_paths)

    if (args['generate_fig'] and Path(args['input_dir']).is_dir()):
        activity_df = construct_activity_grid(args['csv_filename'], ref_audio_files, good_audio_files, output_dir)
        plot_activity_grid(activity_df, output_dir, recover_folder, audiomoth_folder, site_name, save=True)
        cumulative_activity_df = construct_cumulative_activity(args["output_dir"], site_name, "30T")
        plot_cumulative_activity(cumulative_activity_df, args["output_dir"], site_name, "30T")


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
        "--run_model",
        action="store_true",
        help="Do you want to run the model? As opposed to just generating the figure",
    )
    parser.add_argument(
        "--generate_fig",
        action="store_true",
        help="Do you want to generate and save a corresponding summary figure?",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Generate CSV instead of TSV",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
    )

    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()

    run_pipeline(args)