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
    print('Creating {} seconds audio segments.'.format(cfg['segment_duration']))
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
    num_file=0
    for mcfg in cfg['models']:
        agg_df = gen_empty_df() 

        for seg_path in segment_file_paths:
            print('Running file number: {}'.format(num_file))
            num_file += 1
            annotations_df = mcfg['model'].run(seg_path['audio_file'])
            corrected_annotations_df = _correct_annotation_offsets(
                annotations_df, 
                cfg['audio_file_path'].name,
                seg_path['offset']
            ) 

            agg_df = pd.concat([agg_df, corrected_annotations_df], ignore_index=True)
        
        # if mcfg['model'].get_name() == 'feed_buzz_detector':
        #     # remove FP before saving results
        #     agg_df = _process_output(cfg, agg_df)

        csv_name = _generate_csv(agg_df, mcfg['model'].get_name(),
            cfg['audio_file_path'].name,
            cfg['csv_output_path'],
        )
        csv_names.append(csv_name)

    return csv_names
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
    print('Run complete')

    #_process_output(cfg, csv_names) # TODO: should this also write csv output? We are currently doing this inside apply models, is that ok?