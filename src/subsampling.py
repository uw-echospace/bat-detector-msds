import numpy as np
import argparse
import os
import pandas as pd
import soundfile as sf
from maad import sound, util
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def subsample_withpaths(segmented_file_paths, cfg, cycle_length, percent_on):
    necessary_paths = []

    for path in segmented_file_paths:
        if (path['offset'] % cycle_length == 0 # Check if starting position is within recording period; won't need to check rest of boolean if it is
            or ((path['offset']+cfg['segment_duration'])%cycle_length > 0 and (path['offset']+cfg['segment_duration'])%cycle_length <= int(cycle_length*percent_on))):
            necessary_paths.append(path)

    return necessary_paths

def get_dets_from_csv_files(date, location):
    dets1 = pd.read_csv(f"../output_dir/1min_every_6min__{location.split()[0]}_{date}_030000to130000.csv")
    dets2 = pd.read_csv(f"../output_dir/5min_every_30min__{location.split()[0]}_{date}_030000to130000.csv")
    c_dets = pd.read_csv(f"../output_dir/continuous__{location.split()[0]}_{date}_030000to130000.csv")

    return c_dets, dets1, dets2

def get_dets_lfdets_and_hfdets(dets, filename):
    detects = dets[dets['input_file']==filename]
    lfdetects = detects[detects["high_freq"] < 45000]
    hfdetects = detects[detects["low_freq"] > 35000]

    return detects, lfdetects, hfdetects

def add_num_to_buckets(bucket1, bucket2, bucket3, num1, num2, num3):
    bucket1 = np.hstack([bucket1, [num1]])
    bucket2 = np.hstack([bucket2, [num2]])
    bucket3 = np.hstack([bucket3, [num3]])

    return bucket1, bucket2, bucket3

def get_presence_from_numdets(num_dets, num_lfdets, num_hfdets, presence_threshold):
    presence = np.ones(num_dets.shape[0])
    presence[num_dets < presence_threshold] = 0
    lfpresence = np.ones(num_lfdets.shape[0])
    lfpresence[num_lfdets < presence_threshold] = 0
    hfpresence = np.ones(num_hfdets.shape[0])
    hfpresence[num_hfdets < presence_threshold] = 0

    return presence, lfpresence, hfpresence

def get_metrics_from_day(date, location, labels, presence_threshold, scheme=0):
    num_dets = np.array([])
    num_lfdets = np.array([])
    num_hfdets = np.array([])

    dets = get_dets_from_csv_files(date, location)[scheme]

    for label in labels:
        if (label.split('_')[0] == date):
            detects, lfdetects, hfdetects = get_dets_lfdets_and_hfdets(dets, label)
            num_dets, num_lfdets, num_hfdets = add_num_to_buckets(num_dets, num_lfdets, num_hfdets, 
                                                                    detects.shape[0], lfdetects.shape[0], hfdetects.shape[0])

    presence, lfpresence, hfpresence = get_presence_from_numdets(num_dets, num_lfdets, num_hfdets, presence_threshold)

    return presence, lfpresence, hfpresence, num_dets, num_lfdets, num_hfdets

def get_metrics_over_days(dates, location, labels, presence_threshold, scheme=0):
    presence_over_days = np.array([])
    lfpresence_over_days = np.array([])
    hfpresence_over_days = np.array([])

    numdets_over_days = np.array([])
    lfnumdets_over_days = np.array([])
    hfnumdets_over_days = np.array([])

    for i, f_date in enumerate(dates):
        presence, lfpresence, hfpresence, num_dets, num_lfdets, num_hfdets = get_metrics_from_day(f_date, location, labels, presence_threshold, scheme)

        if (i==0):
            presence_over_days = np.hstack((presence_over_days, presence))
            lfpresence_over_days = np.hstack((lfpresence_over_days, lfpresence))
            hfpresence_over_days = np.hstack((hfpresence_over_days, hfpresence))

            numdets_over_days = np.hstack((numdets_over_days, num_dets))
            lfnumdets_over_days = np.hstack((lfnumdets_over_days, num_lfdets))
            hfnumdets_over_days = np.hstack((hfnumdets_over_days, num_hfdets))
        else:
            presence_over_days = np.vstack((presence_over_days, presence))
            lfpresence_over_days = np.vstack((lfpresence_over_days, lfpresence))
            hfpresence_over_days = np.vstack((hfpresence_over_days, hfpresence))

            numdets_over_days = np.vstack((numdets_over_days, num_dets))
            lfnumdets_over_days = np.vstack((lfnumdets_over_days, num_lfdets))
            hfnumdets_over_days = np.vstack((hfnumdets_over_days, num_hfdets))
  
    if (len(presence_over_days.shape) == 1):  
        presence_over_days = presence_over_days.reshape((1, 21))
    if (len(lfpresence_over_days.shape) == 1): 
        lfpresence_over_days = lfpresence_over_days.reshape((1, 21))
    if (len(hfpresence_over_days.shape) == 1): 
        hfpresence_over_days = hfpresence_over_days.reshape((1, 21))

    if (len(numdets_over_days.shape) == 1):  
        numdets_over_days = numdets_over_days.reshape((1, 21))
    if (len(lfnumdets_over_days.shape) == 1): 
        lfnumdets_over_days = lfnumdets_over_days.reshape((1, 21))
    if (len(hfnumdets_over_days.shape) == 1): 
        hfnumdets_over_days = hfnumdets_over_days.reshape((1, 21))

    return presence_over_days, lfpresence_over_days, hfpresence_over_days, numdets_over_days, lfnumdets_over_days, hfnumdets_over_days

def plt_msds_fromdf(location, filename, df, audio_sec, fs, offset, reftimes, times, cycle_length, p_on, be_subplot=False, show_PST=False, show_legend=False, show_threshold=False, lf_threshold=40000, hf_threshold=40000, show_num_dets=False, det_linewidth=2, show_audio=False, show_spectrogram=True, spec_cmap='ocean', spec_NFFT = 256, rm_dB = 200, save=False, save_dir='../output_dir'):
    
    ## If user wants to plot in PST time, adjust the hour accordingly to read into datetime
    hour = int(filename[9:11])
    if (show_PST):
        if (hour >= 7):
            hour = hour - 7
        else:
            hour = 24 + hour - 7
    zero_pad_hour = str(hour).zfill(2)

    ## Strip the datetime for year, month, date, and hour from filename
    file_dt = dt.datetime.strptime(f'{filename[:9]}{zero_pad_hour}{int(offset/60)%60}{int(offset%60)}', '%Y%m%d_%H%M%S')

    ## Only find numPoints amount of labels from all available seconds
    numPoints = 11
    seconds = np.arange(fs*times[0], fs*times[1]+1)/fs
    idx = np.round(np.linspace(0, len(seconds)-1, numPoints)).astype('int32')
    sec_labels = reftimes[0] + seconds[idx]

    ## Calculate Time Labels for X-Axis using Datetime objects as Strings
    if times[1] < 400:
        if times[1] < 150: # If duration of plotted signal is less than 150s, show detail up to microseconds for each time stamp.
            time_labels = [dt.datetime(year=file_dt.year, month=file_dt.month, 
                                                day=file_dt.day, hour=file_dt.hour + int((file_dt.minute + (sec/60))/60), 
                                                minute=(file_dt.minute + int((file_dt.second + sec)/60))%60, second=int((file_dt.second + sec)%60), 
                                                microsecond=np.round(1e6*((file_dt.second + sec)%60-int((file_dt.second + sec)%60))).astype('int32')).strftime('%T.%f')[:-4] 
                                                for sec in sec_labels]
        else: # If duration of plotted signal is less than 400s but greater than 150s, show detail up to seconds for each time stamp.
            time_labels = [dt.datetime(year=file_dt.year, month=file_dt.month, 
                                            day=file_dt.day, hour=file_dt.hour + int((file_dt.minute + (sec/60))/60), 
                                            minute=(file_dt.minute + int((file_dt.second + sec)/60))%60, second=int((file_dt.second + sec)%60)).strftime('%T')
                                            for sec in sec_labels]
    else: # If duration of plotted signal is greater than 400s, show detail up to minutes only for each time stamp.
        time_labels = [dt.datetime(year=file_dt.year, month=file_dt.month, 
                                            day=file_dt.day, hour=file_dt.hour + int((file_dt.minute + (sec/60))/60), 
                                            minute=(file_dt.minute + int((file_dt.second + sec)/60))%60).strftime('%H:%M') 
                                            for sec in sec_labels]
    
    ## Find x-axis tick locations from all available seconds and convert to samples
    s_ticks = seconds[idx]-times[0]
    x_ticks = s_ticks*fs

    ## Calculate detection parameters from msds output to use for drawing rectangles
    xs_inds, xs_freqs, x_durations, x_bandwidths, det_labels = ss.get_msds_params_from_df(df, reftimes[0]+times)
    vmin = 20*np.log10(np.max(audio_sec)) - rm_dB  # hide anything below -rm_dB dB

    ## Create figure for the audio signal: multiple figures can be generated with this method
    legend_fontsize = 16 # This is the fontsize for the text in the legend
    ylabel_fontsize = 20 # This is the fontsize for the text in the ylabel

    if (show_audio): # If show_audio is true, the plot will consist of an audio signal along with a spectrogram of the given audio signal
        ## Throughout this method, some if cases are written to handle more intensive plots by only plotting what's necessary
        ## So for audio signals of greater than 20mins, there will be less intensive plots like generating spectrograms

        if (times[1] < 1200): # If the signal is less than 20mins, we plot 3 subplots: audio signal, spectrogram, and a spectrogram w/ detections
            plt.figure(figsize=(18, 12))
            plt.subplot(311)
        else: # If the signal is greater than 20mins, we plot only 2 subplots: audio signal and a spectrogram w/ detections
            plt.figure(figsize=(12, 8))
            plt.subplot(211)

        ## Plot the provided audio as a signal in this section
        plt.title(f"Audio from {file_dt.date()} in {location}, {time_labels[0]} to {time_labels[-1]}")
        plt.plot(audio_sec)
        plt.xlim((0, s_ticks[-1]*fs))
        plt.xticks(ticks=x_ticks, labels=time_labels)
        amp_ticks = plt.yticks()[0]
        plt.ylabel("Amplitude (V)", fontsize=ylabel_fontsize)
        if (np.max(amp_ticks) > 1000):
            plt.yticks(ticks=amp_ticks, labels=(amp_ticks/1000).astype('int16'))
            plt.ylabel("Amplitude (kV)")
        plt.ylim((amp_ticks[0], amp_ticks[-1]))
        plt.grid(which="both")
        
        ## Moving on to the next subplot
        if (times[1] < 1200): # Plot the spectrogram of the audio signal in the case of signal being less than 20mins
            plt.subplot(312)
            plt.title(f"Spectrogram Representation showing Frequencies {0} to {fs//2000}kHz")
            plt.specgram(audio_sec, Fs=fs, cmap=spec_cmap, vmin=vmin)
            plt.ylabel("Frequency (kHz)", fontsize=ylabel_fontsize)
            plt.xticks(ticks=s_ticks, labels=time_labels)
            plt.xlim((0, s_ticks[-1]))
            ## Find y-axis tick locations from specgram-calculated locations and keep limit just in case
            f_ticks = plt.yticks()[0]
            f_ticks = f_ticks[f_ticks <= fs/2]
            plt.yticks(ticks=f_ticks, labels=(f_ticks/1000).astype('int16'))

            ## Plotting Spectrogram with MSDS outputs overlayed
            plt.subplot(313)
        else: # Create a subplot for the spectrogram w/ detections in the case of signal being greater than 20mins
            plt.subplot(212)
        ## Give a title to the subplot with the spectograms w/ detections overlayed
        plt.title(f"Spectrogram Representation with Call Detections Overlayed")

    else: # If show_audio is false, the plot will consist of only a spectrogram of the given audio signal
        ## be_subplot provides functionality to either plot spectrogram as its own plot or include it with other method call plots
        if (not(be_subplot)):
            plt.figure(figsize=(12, 4))
            plt.title(f"{file_dt.date()} in {location} | {cycle_length//60}-min, {100*p_on:.1f}% Duty Cycle")

    ## show_spectrogram provides functionality to hide the spectrogram if one wanted to just show the subsampling scheme
    if (show_spectrogram): 
        ## User is allowed to set their own NFFT, cmap, and vmin to plot clear customizable spectrograms
        plt.specgram(audio_sec, NFFT=spec_NFFT, Fs=fs, cmap=spec_cmap, vmin=vmin)

    ## Set the generalizable plt features such as xlim, ylabel, and xticks
    plt.xlim((0, s_ticks[-1]))
    plt.ylabel("Frequency (kHz)", fontsize=ylabel_fontsize)
    plt.xticks(ticks=s_ticks, labels=time_labels)

    ## If show_PST is on, set xlabel as PST, 
    if (show_PST):
        if (times[1] < 400): # If timestamps are in second-precision, set xlabel units as (HH:MM:SS)
            plt.xlabel("PST Time (HH:MM:SS)")
        else: # If timestamps are in minute-precision, set xlabel units as (HH:MM)
            plt.xlabel("PST Time (HH:MM)")
    else:
        if (times[1] < 400): # If timestamps are in second-precision, set xlabel units as (HH:MM:SS)
            plt.xlabel("UTC Time (HH:MM:SS)")
        else:
            plt.xlabel("UTC Time (HH:MM)") # If timestamps are in minute-precision, set xlabel units as (HH:MM)

    # Find y-axis tick locations from specgram-calculated locations and keep limit just in case
    f_ticks = plt.yticks()[0]
    f_ticks = f_ticks[f_ticks <= fs/2]
    plt.yticks(ticks=f_ticks, labels=(f_ticks/1000).astype('int16'))

    ## Below section pertains to plotting detections and managing legend information on the spectrogram
    ax = plt.gca()
    num_dets = 0 # Keep track of number of detections plotted to display after in legend
    
    ## Iterate through detections and draw yellow boxes around each detection within the recording period
    for i in range(len(xs_inds)):
        rect = patches.Rectangle((xs_inds[i], xs_freqs[i]), 
                        x_durations[i], x_bandwidths[i], 
                        linewidth=det_linewidth, edgecolor='yellow', facecolor='none', alpha=0.8)
        
        ## Only plot the detection boxes if they are within the simulated recording period
        if (np.floor((xs_inds[i]+x_durations[i])*fs).astype('int32') < len(audio_sec) and audio_sec[np.floor((xs_inds[i]+x_durations[i])*fs).astype('int32')] != 0):
            ax.add_patch(rect)
            num_dets += 1

    ## Next we will display semi-opaque regions for simulated recording and sleep periods
    if (show_spectrogram):
        ## When showing the spectrogram, these regions will be semi-transparent yellow that blend w/ spectrogram to appear green
        on_color = "yellow"
        on_alpha = 0.2
    else:
        ## When showing only the subsampling schemes (no audio and spectrogram), these regions will be black to contrast with white
        on_color = 'black'
        on_alpha = 1.0

    ## Below we handle threshold lines, legends, and simulated regions
    if (not(show_audio)): # This case exists when we want to compare verious spectrograms together w/out showing audio
        ## Here we show threshold so audience can see where clusters of bat calls lie across frequency against a threshold
        if (p_on == 1.0): ## This threshold goes from 0 to the end of audio plot as it is for a continuouus recording
            if (show_threshold):
                    plt.axhline(hf_threshold, xmin=0, xmax=(audio_sec.shape[0])/times[1], linestyle='dashed', color='cyan')
        for tick in sec_labels:
            ## This is a region for duty cycled recording
            if (p_on < 1.0 and int(tick)%cycle_length == 0):
                ## In the case where we want to show threshold in duty cycled recording, the threshold will need to follow recording periods
                if (show_threshold):
                    plt.axhline(hf_threshold, xmin=(int(tick)-reftimes[0])/times[1], xmax=(int(tick)-reftimes[0] + int(p_on*cycle_length))/times[1], linestyle='dashed', color='cyan')
                rect = patches.Rectangle((int(tick)-reftimes[0], 0), width=int(p_on*cycle_length), height=fs/2, linewidth=1, edgecolor=on_color, facecolor=on_color, alpha=on_alpha)
                ax.add_patch(rect)
            
            ## This is a region for continuous recording
            if (p_on == 1.0):
                tick = int(tick) - reftimes[0]
                if (tick%(reftimes[1] - reftimes[0]) == 0):
                    rect = patches.Rectangle((tick, 0), width=int(audio_sec.shape[0] / fs), height=fs/2, linewidth=1, edgecolor=on_color, facecolor=on_color, alpha=on_alpha)
                    ax.add_patch(rect)

        ## Here we handle plotting of the legend for the multiple regions we wish to show
        if (show_spectrogram): ## This also is only a feature when show_spectrogram is true
            ## Yellow is the color we use for the detections and detection boxes
            yellow_rect = patches.Patch(edgecolor=on_color, facecolor=on_color, label = f"{num_dets} Detections")
            ## Green which is really semi-transparent yellow overlayed onto a blue spectrogram is used for simulated recording period
            green_rect = patches.Patch(edgecolor='yellow', facecolor="green", alpha = 0.5, label="Simulated Recording Period")
            ## Blue is the color used for simulated sleep period as majority of the spectrogram background w/out yellow region overlay is blue
            blue_rect = patches.Patch(edgecolor='k', facecolor="royalblue", alpha=0.8, label="Simulated Sleep Period")

            ## Showing legend implies we are including details of recording period, sleep period, and # of dets
            if (show_legend):
                ## Given specific white spaces and region positions, these are several if-cases to position the legend in a pleasing way
                if (p_on < 1.0):
                    if (sec_labels[0]==0):
                        plt.legend(handles=[green_rect, blue_rect, yellow_rect], fontsize=legend_fontsize, loc=1)
                    else:
                        plt.legend(handles=[green_rect, blue_rect, yellow_rect], fontsize=legend_fontsize, loc=2)
                else:
                    if (sec_labels[0]==0):
                        plt.legend(handles=[green_rect, blue_rect, yellow_rect], fontsize=legend_fontsize, ncol=3, loc=1)
                    else:
                        plt.legend(handles=[green_rect, blue_rect, yellow_rect], fontsize=legend_fontsize, ncol=3, loc=2)

            ## We wish to sometimes show number of detections even when show_legend is false. This is for that case
            if (show_num_dets):
                plt.legend(handles=[yellow_rect], fontsize=legend_fontsize, loc=1)

    else: # When we are showing audio and showing spectogram, we again only plot the # of yellow detection boxes
        if (show_spectrogram):
            if (show_legend):
                yellow_rect = patches.Patch(edgecolor=on_color, facecolor=on_color, label = f"{num_dets} Detections")
                if (sec_labels[0]==0):
                    plt.legend(handles=[yellow_rect], fontsize=legend_fontsize, loc=1)
                else:
                    plt.legend(handles=[yellow_rect], fontsize=legend_fontsize, loc=2)

    ## Let's autoformat the xticks/timestamps.
    ## In the case where we want to show only subsampling schemes (no spectorgram), we don't need proper timestamps
    if (show_spectrogram): 
        plt.gcf().autofmt_xdate()
    plt.tight_layout() # plt.tight_layout() is something I always done for final touch-ups

    ## Pretty basic save commands for plot using provided save directories
    if (save):
        directory = f'{save_dir}/{dt.datetime.strftime(file_dt, "%Y%m%d")}'
        start_datetime = dt.datetime.strftime(dt.datetime.strptime(time_labels[0], "%H:%M:%S.%f"), "%H%M%S")
        end_datetime = dt.datetime.strftime(dt.datetime.strptime(time_labels[-1], "%H:%M:%S.%f"), "%H%M%S")
        if not os.path.isdir(directory):
            os.makedirs(directory)

        ## If plot is a spectrogram only, then save under a specific folder for subsampling routine
        if (not(show_audio)):
            directory = f'{directory}/{int(cycle_length*p_on)//60}min_every_{cycle_length//60}min'
            if not os.path.isdir(directory):
                os.makedirs(directory)
            plt.savefig(
                f'{directory}/{location.split()[0]}{location.split()[1]}__{start_datetime}to{end_datetime}.png')
        else: # If plot is a spectrogram and audio, then save under a specific folder as examples/
            directory = f'{directory}/examples'
            if not os.path.isdir(directory):
                os.makedirs(directory)
            plt.savefig(
                f'{directory}/{location.split()[0]}{location.split()[1]}__{start_datetime}to{end_datetime}.png')
    
    ## If plot is its own figure (be_subplot=false), call plt.show()
    if (not(be_subplot)):
        plt.show()

def get_msds_params_from_df(dets:pd.DataFrame, times):
    df = dets
    s_times = df['start_time']
    e_times = df['end_time']
    s_freqs = df['low_freq']
    e_freqs = df['high_freq']
    det_labels = df['event'].values
    xs_inds = s_times[np.logical_and(s_times > times[0], e_times < times[1])].values - times[0]
    xe_inds = e_times[np.logical_and(s_times > times[0], e_times < times[1])].values - times[0]
    xs_freqs = s_freqs[np.logical_and(s_times > times[0], e_times < times[1])].values
    xe_freqs = e_freqs[np.logical_and(s_times > times[0], e_times < times[1])].values
    x_durations = xe_inds - xs_inds
    x_bandwidths = xe_freqs - xs_freqs

    return xs_inds, xs_freqs, x_durations, x_bandwidths, det_labels

def generate_segments(audio_file: Path, output_dir: Path, start_time: float, duration: float):
    """
    Segments audio_file into clips of duration length and saves them to output_dir.
    start_time: seconds
    duration: seconds
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
    cfg = get_config()
    cfg["output_dir"] = Path(output_dir)
    cfg["tmp_dir"] = Path(tmp_dir)
    cfg["num_processes"] = num_processes
    cfg['segment_duration'] = segment_duration

    return cfg

def generate_segmented_paths(summer_audio_files, cfg):
    segmented_file_paths = []
    for audio_file in summer_audio_files:
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

def run_subsampling_pipeline(input_dir, cycle_length, percent_on, csv_name, output_dir, tmp_dir):
    cfg = get_params(output_dir, tmp_dir, 4, 30.0)
    audio_files = sorted(list(Path(input_dir).iterdir()))
    segmented_file_paths = generate_segmented_paths(audio_files, cfg)
    
    ## Get file paths specific to our subsampling parameters
    if (percent_on < 1.0):
        necessary_paths = subsample_withpaths(segmented_file_paths, cfg, cycle_length, percent_on)
    else:
        necessary_paths = segmented_file_paths

    file_path_mappings = initialize_mappings(necessary_paths, cfg)
    bd_dets = run_models(file_path_mappings, cfg, csv_name)

    return bd_dets

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
        "cycle_length",
        type=int,
        help="the desired cycle length in seconds for subsampling",
        default=30,
    )
    parser.add_argument(
        "percent_on",
        type=float,
        help="the desired cycle length in seconds for subsampling",
        default=1/6,
    )

    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()

    run_subsampling_pipeline(args['input_dir'], args['cycle_length'], args['percent_on'], args['csv_filename'], args['output_dir'], args['temp_dir'])