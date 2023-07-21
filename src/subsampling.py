import numpy as np
import argparse
import os
import pandas as pd
import dask.dataframe as dd
import soundfile as sf
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
from pipeline import pipeline


def get_dets_from_csv_files(date, location, cycle_length, percent_on):
    """
    Method to retrieve detections as DataFrames from .csv files saved in output_dir/.
    Uses duty-cycle scheme params: cycle_length and percent_on to find corresponding .csv file
    Assumes that only 1 file for a given scheme, date, and location exists. 
    If multiple files exist for a given scheme, date, and location, .csv files are appended together.

    Parameters
    ------------
    date : `str`
        - String in the format "%Y%m%d" or "YYYYMMDD"
    location : `str`
        - The full name of the location: "Central Pond", "Telephone Field", etc.
    cycle_length : `int`
        - The total duration in seconds of recording and sleep. For 5min ON and 25min OFF, the cycle length would be 30min.
    percent_on : `float`
        - The decimal value of the ratio of recording time to cycle length. For 5min ON and 25min OFF, percent on is 5/30 or 1/6.
    
    
    Returns
    ------------
    bd_dets : `pandas.DataFrame`
        - A DataFrame of detections corresponding to the given date, location, and duty-cycling scheme.
        - 7 columns in this DataFrame: start_time, end_time, low_freq, high_freq, detection_confidence, event, input_file
    
    """

    if percent_on < 1.0:
        bd_dets = dd.read_csv(f"../output_dir/{int(cycle_length*percent_on)//60}min_every_{cycle_length//60}min__{location.split()[0]}_{date}*.csv").compute()
    else:
        bd_dets = dd.read_csv(f"../output_dir/continuous__{location.split()[0]}_{date}*.csv").compute()

    return bd_dets

def get_dets_lfdets_and_hfdets(dets, filename):
    """
    Retrieves the detections corresponding to a specific file using the larger detections .csv file.

    Parameters
    ------------
    dets : `pandas.DataFrame`
        - A DataFrame of detections with frequency ranges and original file name for each detection.
    filename : `str`
        - The desired file that user wants detections corresponding to.
    
    Returns
    ------------
    detects : `pandas.DataFrame`
        - A DataFrame of detections linked to a specific input file for all frequency range.
    lfdetects : `pandas.DataFrame`
        - A DataFrame of low-frequency detections linked to a specific input file.
    hfdetects : `pandas.DataFrame`
        - A DataFrame of high-frequency detections linked to a specific input file.
    
    """
    
    detects = dets[dets['input_file']==filename]
    lfdetects = detects[detects["high_freq"] < 45000]
    hfdetects = detects[detects["low_freq"] > 35000]

    return detects, lfdetects, hfdetects

def add_num_to_buckets(bucket1, bucket2, bucket3, num1, num2, num3):
    """
    Helper method to horizontally stack 3 numbers to 3 corresponding numpy arrays.

    Parameters
    ------------
    bucket1 : `numpy.array`
    bucket2 : `numpy.array`
    bucket3 : `numpy.array`
    num1 : `int`
    num2 : `int`
    num3 : `int`

    Returns
    ------------
    bucket1 : `numpy.array`
    bucket2 : `numpy.array`
    bucket3 : `numpy.array`

    """

    bucket1 = np.hstack([bucket1, [num1]])
    bucket2 = np.hstack([bucket2, [num2]])
    bucket3 = np.hstack([bucket3, [num3]])

    return bucket1, bucket2, bucket3

def get_presence_from_numdets(num_dets, num_lfdets, num_hfdets, presence_threshold):
    """
    Returns presence arrays (1 or 0) given arrays with number of detections according to provided presence threshold

    Parameters
    ------------
    num_dets : `numpy.array`
        - Numpy array of the # of bat call detections for any frequency sorted according to start of recording and end of recording
    num_lfdets : `numpy.array`
        - Numpy array of the # of low-frequency bat call detections sorted according to start of recording and end of recording
    num_hfdets : `numpy.array`
        - Numpy array of the # of high-frequency bat call detections sorted according to start of recording and end of recording
    presence_threshold : `int`
        - Threshold for number of detections required to be classified as presence. Less than threshold is absence

    Returns
    ------------
    presence : `numpy.array`
        - Numpy array of the # of bat call presences for any frequency sorted according to start of recording and end of recording
    lfpresence : `numpy.array`
        - Numpy array of the # of low-frequency bat call presences for any frequency sorted according to start of recording and end of recording
    hfpresence : `numpy.array`
        - Numpy array of the # of high-frequency bat call presences for any frequency sorted according to start of recording and end of recording

    """

    presence = np.ones(num_dets.shape[0])
    presence[num_dets < presence_threshold] = 0
    lfpresence = np.ones(num_lfdets.shape[0])
    lfpresence[num_lfdets < presence_threshold] = 0
    hfpresence = np.ones(num_hfdets.shape[0])
    hfpresence[num_hfdets < presence_threshold] = 0

    return presence, lfpresence, hfpresence

def get_metrics_from_day(date, location, labels, presence_threshold, cycle_length, percent_on):
    """
    Returns presence arrays for each frequency grouping of bat calls for a single date.
    Returns the number of detections for each frequency grouping of bat calls across provided timespan for a single date
    These arrays are assembled from the existing .csv files stored inside output_dir/

    Parameters
    ------------
    date : `str`
        - String in the format "%Y%m%d" or "YYYYMMDD"
    location : `str`
        - The full name of the location: "Central Pond", "Telephone Field", etc.
    labels : `List`
        - Expected list of audio files that should be in the directory using the CONFIG.TXT file
    presence_threshold : `int`
        - Threshold for number of detections required to be classified as presence. Less than threshold is absence
    cycle_length : `int`
        - The total duration in seconds of recording and sleep. For 5min ON and 25min OFF, the cycle length would be 30min.
    percent_on : `float`
        - The decimal value of the ratio of recording time to cycle length. For 5min ON and 25min OFF, percent on is 5/30 or 1/6.

    Returns
    ------------
    presence : `numpy.array`
        - Numpy array of the # of bat call presences for any frequency sorted according to start of recording and end of recording
    lfpresence : `numpy.array`
        - Numpy array of the # of low-frequency bat call presences for any frequency sorted according to start of recording and end of recording
    hfpresence : `numpy.array`
        - Numpy array of the # of high-frequency bat call presences for any frequency sorted according to start of recording and end of recording
    num_dets : `numpy.array`
        - Numpy array of the # of bat call detections for any frequency sorted according to start of recording and end of recording
    num_lfdets : `numpy.array`
        - Numpy array of the # of low-frequency bat call detections sorted according to start of recording and end of recording
    num_hfdets : `numpy.array`
        - Numpy array of the # of high-frequency bat call detections sorted according to start of recording and end of recording

    """

    num_dets = np.array([])
    num_lfdets = np.array([])
    num_hfdets = np.array([])

    dets = get_dets_from_csv_files(date, location, cycle_length, percent_on)

    for label in labels:
        if (label.split('_')[0] == date):
            detects, lfdetects, hfdetects = get_dets_lfdets_and_hfdets(dets, label)
            num_dets, num_lfdets, num_hfdets = add_num_to_buckets(num_dets, num_lfdets, num_hfdets, 
                                                                    detects.shape[0], lfdetects.shape[0], hfdetects.shape[0])

    presence, lfpresence, hfpresence = get_presence_from_numdets(num_dets, num_lfdets, num_hfdets, presence_threshold)

    return presence, lfpresence, hfpresence, num_dets, num_lfdets, num_hfdets

def get_metrics_over_days(dates, location, labels, presence_threshold, cycle_length, percent_on):
    """
    Returns presence arrays for each frequency grouping of bat calls over multiple dates.
    Returns the number of detections for each frequency grouping of bat calls across provided timespan over multiple dates.
    These arrays are assembled from the existing .csv files stored inside output_dir/

    Parameters
    ------------
    date : `str`
        - String in the format "%Y%m%d" or "YYYYMMDD"
    location : `str`
        - The full name of the location: "Central Pond", "Telephone Field", etc.
    labels : `List`
        - Expected list of audio files that should be in the directory using the CONFIG.TXT file
    presence_threshold : `int`
        - Threshold for number of detections required to be classified as presence. Less than threshold is absence
    cycle_length : `int`
        - The total duration in seconds of recording and sleep. For 5min ON and 25min OFF, the cycle length would be 30min.
    percent_on : `float`
        - The decimal value of the ratio of recording time to cycle length. For 5min ON and 25min OFF, percent on is 5/30 or 1/6.

    Returns
    ------------
    presence : `numpy.array`
        - Numpy array of the # of bat call presences for any frequency sorted according to start of recording and end of recording
    lfpresence : `numpy.array`
        - Numpy array of the # of low-frequency bat call presences for any frequency sorted according to start of recording and end of recording
    hfpresence : `numpy.array`
        - Numpy array of the # of high-frequency bat call presences for any frequency sorted according to start of recording and end of recording
    num_dets : `numpy.array`
        - Numpy array of the # of bat call detections for any frequency sorted according to start of recording and end of recording
    num_lfdets : `numpy.array`
        - Numpy array of the # of low-frequency bat call detections sorted according to start of recording and end of recording
    num_hfdets : `numpy.array`
        - Numpy array of the # of high-frequency bat call detections sorted according to start of recording and end of recording

    """

    presence_over_days = np.array([])
    lfpresence_over_days = np.array([])
    hfpresence_over_days = np.array([])

    numdets_over_days = np.array([])
    lfnumdets_over_days = np.array([])
    hfnumdets_over_days = np.array([])

    for i, f_date in enumerate(dates):
        presence, lfpresence, hfpresence, num_dets, num_lfdets, num_hfdets = get_metrics_from_day(f_date, location, labels, presence_threshold, cycle_length, percent_on)

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

def plt_msds_fromdf(location, filename, df, audio_sec, fs, offset, reftimes, times, cycle_length, p_on, be_subplot=False, 
                    show_PST=False, show_legend=False, show_threshold=False, lf_threshold=40000, hf_threshold=40000, show_num_dets=False, 
                    det_linewidth=2, show_audio=False, show_spectrogram=True, spec_cmap='ocean', spec_NFFT = 256, rm_dB = 200, save=False, save_dir='../output_dir'):
    """
    Returns presence arrays for each frequency grouping of bat calls over multiple dates.
    Returns the number of detections for each frequency grouping of bat calls across provided timespan over multiple dates.
    These arrays are assembled from the existing .csv files stored inside output_dir/

    Parameters
    ------------
    location : `str`
        - The full name of the location: "Central Pond", "Telephone Field", etc.
    filename : `str`
        - String of the audio file name in the format "%Y%m%d_%H%M%S.WAV" or "YYYYMMDD_hhmmss.WAV"
    df : `pandas.DataFrame`
        - Dataframe of the detections associated with the audio file. Can be duty-cycled or contiuous.
    audio_sec : `soundfile.SoundFile`
        - Section of audio from the original audio file to be plotted with potentially detections overlayed
    fs : `int`
        - Sampling rate of the audio file.
    offset : `int`
        - The offset of the audio file in the hour. Files corresponding to 030000.WAV have offset=0 and 033000.WAV have offset=1800
    reftimes : `numpy.array`
        - Parameters for the window, in seconds, of time we are plotting of the audio file.
    times : `numpy.array`
        - Parameters for within the above window, in seconds. With reference to reftimes, times is a sub-window parameter.
    cycle_length : `int`
        - The total duration in seconds of recording and sleep. For 5min ON and 25min OFF, the cycle length would be 30min.
        - Using this value to show what recording and sleep periods would look like for given audio file.
    p_on : `float`
        - The decimal value of the ratio of recording time to cycle length. For 5min ON and 25min OFF, percent on is 5/30 or 1/6.
        - Using this value to show what recording and sleep periods would look like for given audio file.
    be_subplot : `bool`
        - Flag used to represent plots as subplots in order to control the figure size and title and join with other method calls on user-end.
    show_PST : `bool`
        - Flag to show PST time instead of UTC time in all timestamps
    show_legend : `bool`
        - Flag to show more detail within the legend; default legend even when show_legend is false exists and shows number of detections in df
    show_threshold : `bool`
        - Flag to show dashed horizontal line for users to see calls in relation to specific frequency values.
    lf_threshold : `int`
        - Frequency value to show threshold across plotted spectrogram of audio at that value to see where calls exist with respect to value.
    hf_threshold : `int`
        - Frequency value to show threshold across plotted spectrogram of audio at that value to see where calls exist with respect to value.
    show_num_dets : `bool`
        - Flag to show the number of detections in the spectrogram of the audio. Can be false and no legend will be shown.
    det_linewidth : `int`
        - Integer value to control the thickness of the detection boxes on the user-end.
    show_audio : `bool`
        - Flag to show plot of audio above spectrogram plot. Useful to see where the spectrogram is coming from.
    show_spectrogram : `bool`
        - Flag to show the spectrogram of the audio section. Can be switched off to just show boxes with simulated recording periods.
    spec_cmap : `str`
        - Colormap for the spectrogram. Good to use 'jet' for very zoomed in sections to get high-contrast. Normally use `cmap` for large sections.
    spec_NFFT : `int`
        - Integer value for NFFT of the spectrogram. Useful to go large when audio section is zoomed in. Good to have on user-end.
    rm_dB : `int`
        - The dB of noise values to remove from the spectrogram by adjusting vmin. In order to make background less noisy.
    save : `bool`
        - Flag to save the plot under provided save directory
    save_dir : `str`
        - String for the path to save the plots. By default, it is "output_dir/"


    """
    
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
    xs_inds, xs_freqs, x_durations, x_bandwidths, det_labels = get_msds_params_from_df(df, reftimes[0]+times)
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
        if (np.floor((xs_inds[i]+x_durations[i])*fs).astype('int32') < len(audio_sec) 
            and audio_sec[np.floor((xs_inds[i]+x_durations[i])*fs).astype('int32')] != 0):
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
                    plt.axhline(hf_threshold, xmin=(int(tick)-reftimes[0])/times[1], 
                                xmax=(int(tick)-reftimes[0] + int(p_on*cycle_length))/times[1], linestyle='dashed', color='cyan')
                rect = patches.Rectangle((int(tick)-reftimes[0], 0), width=int(p_on*cycle_length), height=fs/2, 
                                         linewidth=1, edgecolor=on_color, facecolor=on_color, alpha=on_alpha)
                ax.add_patch(rect)
            
            ## This is a region for continuous recording
            if (p_on == 1.0):
                tick = int(tick) - reftimes[0]
                if (tick%(reftimes[1] - reftimes[0]) == 0):
                    rect = patches.Rectangle((tick, 0), width=int(audio_sec.shape[0] / fs), height=fs/2, linewidth=1, 
                                             edgecolor=on_color, facecolor=on_color, alpha=on_alpha)
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
    """
    Gets detection parameters for plotting detection boxes for a corresponding audio file.

    Parameters
    ------------
    dets : `pandas.DataFrame`
        - DataFrame corresponding to the detections generated by the MSDS pipeline for a given file.
    times : `numpy.array`
        - List of values of where the desired audio_section is to be plotted.
        - Used to get detections only within the desired audio section and ignoring the rest.
    
    Returns
    ------------
    xs_inds : `numpy.array`
        - All bat call detection start_time values corresponding to given DataFrame
    xs_freqs : `numpy.array`
        - All bat call detection low_freq values corresponding to given DataFrame
    x_durations : `numpy.array`
        - All bat call detection time durations corresponding to given DataFrame
    x_bandwidths : `numpy.array`
        - All bat call detection frequency ranges values corresponding to given DataFrame
    det_labels : `numpy.array`
        - All bat call detection event labels corresponding to given DataFrame
    
    """

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

def get_files_to_reference(input_dir, start_time, end_time):
    """
    Gets a list of audio files existing in an input directory representative of the times recorded each day.

    Parameters
    ------------
    input_dir : `str`
        - The provided path to a directory consisting of audio files the user wants to feed into our pipeline.
    start_time : `str`
        - The time at which we want the pipeline to start selecting files for the detections
        - Format : "HH:MM"
    end_time : `str`
        - The time at which we want the pipeline to stop detection on the selected files
        - Format : "HH:MM"

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

    # audio_files = []
    # for file in sorted(list(Path(input_dir).iterdir())):
    #   if (os.path.exists(file) and len(file.name.split('.')) == 2 and 
    #         (file.name.split('.')[1]=="WAV" or file.name.split('.')[1]=="wav")):
    #         file_dt = dt.datetime.strptime(file.name, "%Y%m%d_%H%M%S.WAV")
    #         if ((file_dt.minute == 30 or file_dt.minute == 0) and file_dt.second == 0):
    #             audio_files.append(file)

    # return audio_files

    dates = get_dates_of_deployment(input_dir)

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

    # audio_files = []
    # good_audio_files = []
    # for file in sorted(list(Path(input_dir).iterdir())):
    #     if (os.path.exists(file) and not(os.stat(file).st_size == 0) and
    #          len(file.name.split('.')) == 2 and (file.name.split('.')[1]=="WAV" or file.name.split('.')[1]=="wav")):
    #         file_dt = dt.datetime.strptime(file.name, "%Y%m%d_%H%M%S.WAV")
    #         if ((file_dt.minute == 30 or file_dt.minute == 0) and file_dt.second == 0):
    #             audio_files.append(file)

    comments = exiftool.ExifToolHelper().get_tags(audio_files, tags='RIFF:Comment')
    df_comments = pd.DataFrame(comments)
    print(f"There are {len(audio_files)} audio files that passed 1st level of filtering!")
    good_audio_files = df_comments.loc[~df_comments['RIFF:Comment'].str.contains("microphone")]['SourceFile'].values

    for i in range(len(good_audio_files)):
        good_audio_files[i] = Path(good_audio_files[i])

    print(f"There are {len(good_audio_files)} audio files that passed 2nd level of filtering!")
                
    return good_audio_files

def generate_segmented_paths(summer_audio_files, ref_audio_files, cfg):
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
    for audio_file in summer_audio_files:
        if ((audio_file.name.split('.')[-1] == "WAV" or audio_file.name.split('.')[-1] == "wav")
            and (audio_file in ref_audio_files)):
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
        'original_file_name': f"{Path(audio_seg['audio_file']).name[:15]}.WAV",
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
        - Stored as the provided csv name.

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

def simulate_dutycycle_on_dets(continuous_dets, cycle_length, percent_on, save=False, save_filename='example_dutycycled_dets.csv', save_dir='../output_dir'):
    """
    Returns a new DataFrame with duty-cycled bat calls such that only bat calls within simulated recording periods are kept.

    Parameters
    ------------
    continuous_dets : `pandas.DataFrame`
        - DataFrame of detections generated by batdetect2 from provided pipeline input directory. No duty-cycling applied.
    cycle_lengths : `int`
        - Cycle length is the total duration in seconds of recording and sleep. For 5min ON and 25min OFF, the cycle length would be 30min.
    percent_ons : `float`
        - percent_on is the decimal value of the ratio of recording time to cycle length. For 5min ON and 25min OFF, percent on is 5/30 or 1/6.
    save : `bool`
        - Flag for saving the duty-cycled detections with simulated recording periods.
    save_filename : `str`
        - Filename for the duty-cycled detections .csv file that will be saved under save_dir
    save_dir : `str`
        - Filepath for the duty-cycled detections .csv file. File will be saved under this directory path

    Returns 
    ------------
    dc_dets : `pandas.DataFrame`
        - DataFrame of duty-cycled detections that follow provided duty-cycle scheme.
        - All bat call start and end times should be within simulated recording periods provided cycle length and percent on.

    """

    dc_dets = continuous_dets.loc[np.logical_and((continuous_dets['start_time']%cycle_length)>=0, (continuous_dets['end_time']%cycle_length)<=int(cycle_length*percent_on))]

    if save:
        dc_dets.to_csv(f'{save_dir}/{save_filename}', index=False)
    
    return dc_dets

def run_subsampling_detections_pipeline(input_dir, cycle_lengths, percent_ons, csv_tag, output_dir, tmp_dir, save="False"):
    """
    Runs the batdetect2 pipeline on provided directory of audio files if there is no current record of detections.
    Once detections have been generated, subsamples the dataframe and saves a new detections.csv according to duty-cycle schemes provided.
    If record of detections already exists for files in input_dir, then just skips manual generation of detections.

    Parameters
    ------------
    input_dir : `str`
        - String-based path to the directory of audio files which our pipeline will be fed to generate detections
    cycle_lengths : `List`
        - List of integer cycle lengths which have a corresponding percent_on in the same index to comprise a duty-cycling scheme.
        - Cycle length is the total duration in seconds of recording and sleep. For 5min ON and 25min OFF, the cycle length would be 30min.
    percent_ons : `List`
        - List of float percent_ons which have a corresponding cycle length in the same index to comprise a duty-cycling scheme.
        - percent_on is the decimal value of the ratio of recording time to cycle length. For 5min ON and 25min OFF, percent on is 5/30 or 1/6.
    csv_tag : `str`
        - The tag of the csv that all corresponding detections will keep in their filenames. Contains location, date and times.
    output_dir : `str`
        - String-based path to the directory that will store the outputs: detections and the plot
    tmp_dir : `str`
        - String-based path to the directory that will temporarily store our generated segments to feed into batdetect2 

    Returns
    ------------
    bd_dets : `pandas.DataFrame`
        - A DataFrame of continuous-schemed detections that will also be saved in the provided output_dir with the above csv_tag
        - 7 columns in this DataFrame: start_time, end_time, low_freq, high_freq, detection_confidence, event, input_file
        - Detections are always specified w.r.t their input_file; earliest start_time can be 0 and latest end_time can be 1795.
        - Events are always "Echolocation" as we are using a model that only detects search-phase calls.
        - Detections generated when dutycycling would be missing detections generated on continuous data where sleep was simulated.
    """
    
    if (os.path.isfile(Path(f'{output_dir}/continuous__{csv_tag}.csv'))):
        bd_dets = pd.read_csv(f'{output_dir}/continuous__{csv_tag}.csv')
    else:
        cfg = get_params(output_dir, tmp_dir, 4, 30.0)
        audio_files = sorted(list(Path(input_dir).iterdir()))
        ref_audio_files = get_files_for_pipeline(get_files_to_reference(input_dir, "03:00", "13:30"))
        print(ref_audio_files)
        segmented_file_paths = generate_segmented_paths(audio_files, ref_audio_files, cfg)
        file_path_mappings = initialize_mappings(segmented_file_paths, cfg)
        bd_dets = run_models(file_path_mappings, cfg, f'continuous__{csv_tag}.csv')

    for i, cycle_length in enumerate(cycle_lengths):
        simulate_dutycycle_on_dets(bd_dets, cycle_length, percent_ons[i], save=(save=="True"), 
                                   save_filename=f'{np.round(cycle_length*percent_ons[i]).astype("int")//60}min_every_{cycle_length//60}min__{csv_tag}.csv', save_dir=Path(output_dir))

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
        "csv_tag",
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
    parser.add_argument(
        "save",
        type=str,
        help="Flag for whether to save duty-cycled detections",
        default="False",
    )

    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()

    run_subsampling_detections_pipeline(args['input_dir'], [args['cycle_length']], [args['percent_on']], args['csv_tag'], args['output_dir'], args['temp_dir'], args['save'])