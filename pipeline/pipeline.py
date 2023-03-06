from pathlib import Path

import pandas as pd
import numpy as np

from pipeline.audio_segmentor import generate_segments 
from utils.utils import gen_empty_df

def _generate_csv(annotation_df, model_name, audio_file_name, output_path):
    csv_name = f"{model_name}-{audio_file_name}.csv"
    csv_path = output_path / csv_name
    annotation_df.to_csv(csv_path)
    return csv_path

def _segment_input_audio(cfg):
    segment_file_paths = generate_segments(
        audio_file = cfg['audio_file_path'], 
        output_dir = cfg['tmp_output_path'],
        start_time = cfg['start_time'],
        duration   = cfg['segment_duration'],
    )
    return segment_file_paths

def _correct_annotation_offsets(annotations_df, input_file, actual_start_time):
    annotations_df['start_time'] = annotations_df['start_time'] + actual_start_time
    annotations_df['end_time'] = annotations_df['end_time'] + actual_start_time
    annotations_df['input_file'] = input_file
    return annotations_df

def _apply_models(cfg, segment_file_paths):
    csv_names = []

    for mcfg in cfg['models']:
        agg_df = gen_empty_df() 

        for seg_path in segment_file_paths:
            annotations_df = mcfg['model'].run(seg_path['audio_file'])
            corrected_annotations_df = _correct_annotation_offsets(
                annotations_df, 
                cfg['audio_file_path'].name,
                seg_path['offset']
            ) 

            agg_df = pd.concat([agg_df, corrected_annotations_df], ignore_index=True)
        
        if mcfg['model'].get_name() == 'feed_buzz_detector':
            # remove FP before saving results
            agg_df = _process_output(cfg, agg_df)
        csv_name = _generate_csv(agg_df, mcfg['model'].get_name(),
            cfg['audio_file_path'].name,
            cfg['csv_output_path'],
        )
        csv_names.append(csv_name)

    return csv_names
"""
Remove collision between feeding buzz false positive and bat calls true positive values.

Return: a boolean
"""
def removing_collision(curr_row:tuple, compare_df:pd.DataFrame):
    # TODO: Decide if bounding box interect is a good idea (might remove TP), maybe better to compare in center
    XB1 = curr_row.min_t
    XB2 = curr_row.max_t
    YB1 = curr_row.min_f
    YB2 = curr_row.max_f
    SB = (XB2 - XB1) * (YB2 - YB1)
   
    #print('Looping compare df to find collision')
    for i in compare_df.itertuples():
        print(i)
        XA1 = i[1] #min_t
        XA2 = i[2] #max_t
        YA1 = i[3] #min_f
        YA2 = i[4] #max_f

        if (XB2 >= XA2 and XA1 >= XB1 and YB2 >= YA2 and YA1 >= YB1 ):
            return 1
    return 0

def _process_output(cfg, fb_detect_df):
    for mcfg in cfg['models']:
        if mcfg['model'].get_name() == 'batdetect2':
            bat_detect_df = pd.read_csv(cfg['csv_output_path'] / f"{mcfg['model'].get_name()}-{cfg['audio_file_path'].name}.csv")
        else:
            print("Cannot find batdetect2 annotation file: will not clean buzzfeed FP")
    collide = np.zeros(len(fb_detect_df))

    for curr in fb_detect_df.itertuples():
        collide[curr.Index] = removing_collision(curr,bat_detect_df)
    
    fb_detect_df['Collide'] = collide

    # Formating df
    fb_detect_df.rename(columns={'min_t':'Begin Time (s)', 'max_t':'End Time (s)',
                             'min_f':'Low Freq (Hz)','max_f':'High Freq (Hz)'},inplace=True) #TODO: Why rename the columns? of so, do we need to rename the batdetect ones too?

    rois_df_filtered = fb_detect_df[fb_detect_df['Collide']== 0] 

    del fb_detect_df['Collide']

    return rois_df_filtered
    # full_list = pd.DataFrame()

    # for csv_name in csv_names:
    #     curr = pd.read_csv(csv_name)
    #     full_list = pd.concat([full_list,curr],ignore_index = True)

    # full_list.sort_values(by = ['start_time'],inplace = True, ascending = True)
    # full_list.to_csv(cfg['csv_output_path'])

def _prepare_output_dirs(cfg):
    # TODO: do we need to clearn tmp dir before each run?
    cfg['csv_output_path'].mkdir(parents=True, exist_ok=True)
    cfg['tmp_output_path'].mkdir(parents=True, exist_ok=True)


def run(cfg: dict):
    _prepare_output_dirs(cfg)
    segmented_file_paths = _segment_input_audio(cfg)
    csv_names = _apply_models(cfg, segmented_file_paths)

    #_process_output(cfg, csv_names) # TODO: should this also write csv output? We are currently doing this inside apply models, is that ok?