# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from maad import sound, util
import models.bat_call_detector.template_matching_func as tm

from pathlib import Path
import pickle
from tqdm import tqdm

# Set constants
# TODO: Decide if these constants should be defined by user in constructor
NPERSEG = 1024
NOVERLAP = 512
WINDOW = 'hann'
DB_RANGE = 80


def remove_template(pickle_template_path:Path, remove_namelist:list):
    '''
    Remove templates from current template dictionary based on user specific template.

    Parameters:: 
        pickle_template_path: Path that stores the existing template dictionary
        remove_namelist: A list containing the name(s) of template that user wants to remove. The strings must match the names in the keys of the template that is being loaded.

    Return:: a saved pickle with removed items
    '''
    print('This function will remove these templates {} from current template dictionary'.format(remove_namelist))
    
    flag = True

    while flag:
        response = str(input('Please enter Y to continue or N to stop'))
        if response in ['Y','N']:
            flag = False
            break
        else:
            print('Incorrect input, please enter Y or N')
            continue

    if response == 'Y':
        flag = False
        try:
            with open(pickle_template_path, 'rb') as handle:
                template_dict = pickle.load(handle)
                print('Template exists.')
        except:
            print('Template does not exist.')
        
        for i in remove_namelist:
            del template_dict[i]
        
        print('Remove lists deleted from template dictionary.')
        save_template_dict(template_dict, pickle_template_path)
        return 
        
    elif response == 'N':
        print('Stopping function...')
        return
            

def generate_template(template_audio_path:Path, pickle_template_path:Path, freq_type:str, tlims:tuple, flims:tuple):
    """
    Generate template based on user defined time and frequency limit.

    Paremeters::
        template_audio_path: a Path object containing the directory of the original .wav file that contains the feeding buzz

        pickle_template_path: a Path object containing the directory where the pickle object will be stored

        freq_type: either 'lf' for low frequency template or 'hf' for high frequency template

        tlims: tuple containing the start time and end time of the feeding buzz. Both values must be identified manually beforehand.

        flims: uple containing the start frequency and end frequency of the feeding buzz. Both values must be identified manually beforehand.
    
    Return:: None

    """
    # we want to create template, save it and update the template dictionary
    template_dict = load_templates(pickle_template_path)
    template_name = 'template_{}_{}_{}_{}'.format(freq_type,template_audio_path.stem, tlims[0], tlims[1])
    if template_name not in template_dict:
        s_template, fs_template = sound.load(template_audio_path)
        Sxx_template, _, _, _ = sound.spectrogram(s_template, fs_template, WINDOW, NPERSEG, NOVERLAP, flims, tlims)
        # we update the dictionary
        template_dict[template_name] = (Sxx_template, freq_type, flims, tlims)    
        # we save the template
        save_template_dict(template_dict, pickle_template_path)
    return


def load_templates(template_path:Path):
    """
    Load pickled template into a dictionary.

    Parameters::
        template_path: a Path containing the pickle object

    Return:: a dictionary
    """
    #Ideally we should be able to choose which templates to load, for now we will load all
    try:
        with open(template_path, 'rb') as handle:
            template_dict = pickle.load(handle)
    except:
        # if it's the first time creating template dict
        print('Template path is empty!')
        template_dict = dict()
    return template_dict


def save_template_dict(template_dict:dict, pickle_template_path: Path):
    """
    Save dictionary into a pickle.

    Parameters::
        template_audio_path: a Path object containing the directory of the original .wav file that contains the feeding buzz

        pickle_template_path: a Path object containing the directory where the pickle object will be stored

    Return:: None
    """
        # we save the updated dict
    with open(pickle_template_path, 'wb') as handle:
        pickle.dump(template_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    


def run_template_matching(Sxx_audio: np.ndarray,  tn: any, ext: any, template: tuple, template_name:str, peak_th: float, peak_distance: float):
    """
    Run template matching process for one specific template across target audio file. 

    Parameters::
        Sxx_audio: Spectrogram : Matrix containing K frames with N/2 frequency bins, 
        K*N <= length (wave) Sxx unit is power => Sxx_power if mode is ‘psd’ Sxx unit is amplitude => Sxx_ampli if mode is ‘amplitude’ or ‘complex’

        tn: 1d ndarray of floats, time vector (horizontal x-axis)

        ext: list of scalars [left, right, bottom, top], the location, in data-coordinates, of the lower-left and upper-right corners.

        template: tuple of a dictionary containing template information in this structure {template_name: (spectrogram matrix of the template, frequency types, time range, frequency range)}.

        template_name: name of the template used in current iteration

        peak_th:float, ranges -1 to 1, threshold applied to find peaks in the cross-correlation array

        peak_distance: float, required minimal temporal distance (>= 0) in seconds between neighbouring
        peaks. If set to `None`, the minimum temporal resolution will be used.
        The minimal temporal resolution is given by the array tn and depends on the parameters
        used to compute the spectrogram.

    Return:: a pd.DataFrame containing identified feeding buzzes with correlation coefficient.
    """
    xcorrcoef, rois = tm.template_matching(Sxx_audio, template[0], tn, ext, peak_th, peak_distance)
    rois['min_f'] = template[2][0]
    rois['max_f'] = template[2][1]
    rois['template_name'] = template_name

    return rois


def match_rois(rois: pd.DataFrame, out_df:pd.DataFrame, num_matches_threshold: int, buzz_feed_range: float, alpha:float):
    """
    Select one region of interest from a group of similar regions of interest of different template. 
    This happens because an actual feeding buzz is likely to match with several templates due to high coefficient. 
    The parameter num_matches_threshold indicates the minimum number of templates the region of interest requires to match with. 

    Parameters::
        rois: a DataFrame containing the feeding buzz identified by the template matching function, results from template matching pipeline.

        out_df: originally an empty DataFrame

        num_matches+threshold: int, ranges 0 to the total number of templates.
            The number of template that matches the detected area of interest(aoi). The smaller this number is, the
            fewer templates that the detected aoi has to match in order to be returned as a confirmed feeding buzz.

        buzz_feed_range: float, in milisecond. ranges 0.0 to 1.0.
            The distance between two consecutive feeding buzz.

        alpha: int, ranges from 0 to 1.
            A tunable parameter to find the surrounding feeding buzzes identified by similar templates.

    Return:: a pd.Dataframe with filtered false positive
    """
    match_dict = dict()

    match_range = alpha*buzz_feed_range/2
    # get a random rois from the df, find all matching rois
    rois_matching = rois.copy()
    while rois_matching.shape[0] > 0:
        # get a random row
        rnd_row = rois_matching.sample()
        # get rand row mid_point
        rnd_row_mid_point = float(rnd_row['peak_time'])
        # find all rows that match this row
        match_rows = rois_matching[rois_matching['peak_time'].between(rnd_row_mid_point-match_range,rnd_row_mid_point+match_range)]
        # we store matched info in dictionary: (count, tlims, flims, avg.corrcoef)
        match_dict[rnd_row_mid_point] = (match_rows.shape[0], (match_rows.min_t.quantile(0.3), match_rows.max_t.quantile(0.7)), (match_rows.min_f.quantile(0.3), match_rows.max_f.quantile(0.7)), match_rows.xcorrcoef.mean())
        # we remove the matched rows from the DataFrame 
        rois_matching.drop(match_rows.index, inplace=True)

    match_dict_cut = {k: v for k, v in match_dict.items() if v[0] > num_matches_threshold}
    # we sort the dictionary by key (time)
    match_dict_cut = dict(sorted(match_dict_cut.items()))    

    # we convert dict 
    for i, value in enumerate(match_dict_cut.values()):
        out_df.loc[i] = [value[1][0],value[1][1],value[2][0],value[2][1],value[3],'feeding buzz']
    return out_df



def run_multiple_template_matching(PATH_AUDIO: Path, out_df:pd.DataFrame, peak_th: float, peak_distance: float, template_dict:dict, num_matches_threshold:int, buzz_feed_range: float, alpha: float):
    """
    Run template matching across all templates in template dict for each 1 minute audio file

    Parameters::
        PATH_AUDIO: a Path object containing the post-processed audio .wav file.

        out_df: orginally an empty DataFrame

        peak_th: float, ranges -1 to 1.
            Threshold applied to find peaks in the cross-correlation array

        peak_distance: float.
            Required minimal temporal distance (>= 0) in seconds between neighbouring
            peaks. If set to `None`, the minimum temporal resolution will be used.
            The minimal temporal resolution is given by the array tn and depends on the parameters
            used to compute the spectrogram.

        template_dict: a dict object holding all the templates generated from generate_template() function

        num_matches_threshold: int, ranges 0 to the total number of templates.
            The number of template that matches the detected area of interest(aoi). The smaller this number is, the
            fewer templates that the detected aoi has to match in order to be returned as a confirmed feeding buzz.

        buzz_feed_range:float, in milisecond. ranges 0.0 to 1.0.
            The distance between two consecutive feeding buzz.

        alpha: int, ranges from 0 to 1.
            A tunable parameter to find the surrounding feeding buzzes identified by similar templates.

    Return:: a pd.Dataframe combining results using all templates,
            columns = ['Begin Time (s)', 'End Time (s)','Low Freq (Hz', 'High Freq (Hz)', 'Collide'].
    """
    # Load sound and initiate variables
    s, fs = sound.load(PATH_AUDIO)
    rois_df = pd.DataFrame() 

    for template in template_dict.keys():
        # Compute spectrogram for target audio of the same width as template
        Sxx_audio, tn, fn, ext = sound.spectrogram(s, fs, WINDOW, NPERSEG, NOVERLAP, template_dict[template][2])
        Sxx_audio = util.power2dB(Sxx_audio, DB_RANGE)
        curr_df = run_template_matching(Sxx_audio, tn, ext,
                                        template=template_dict[template], 
                                        template_name=template, 
                                        peak_th=peak_th,
                                        peak_distance=peak_distance)
        rois_df = pd.concat([rois_df,curr_df], ignore_index=True)
    
    out_df = match_rois(rois_df, out_df, num_matches_threshold, buzz_feed_range, alpha)

    return out_df

