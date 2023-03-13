from pathlib import Path

import pandas as pd
import numpy as np

from torch import multiprocessing
from tqdm import tqdm

from pipeline.audio_segmentor import generate_segments 
from utils.utils import gen_empty_df, convert_df_ravenpro

def _generate_csv(annotation_df, model_name, audio_file_name, output_path, should_csv):
    file_name = f"{model_name}-{audio_file_name}"
    extension = ".csv"
    sep = ","

    if not should_csv:
        extension = ".txt"
        sep = "\t"
        annotation_df = convert_df_ravenpro(annotation_df)

    csv_path = output_path / f"{file_name}{extension}"
    annotation_df.to_csv(csv_path, sep=sep, index=False)
    return csv_path

def _segment_input_audio(cfg):
    segment_file_paths = generate_segments(
        audio_file = cfg['audio_file'], 
        output_dir = cfg['tmp_dir'],
        start_time = cfg['start_time'],
        duration   = cfg['segment_duration'],
    )
    return segment_file_paths

def _correct_annotation_offsets(annotations_df, input_file, actual_start_time):
    annotations_df['start_time'] = annotations_df['start_time'] + actual_start_time
    annotations_df['end_time'] = annotations_df['end_time'] + actual_start_time
    annotations_df['input_file'] = input_file
    return annotations_df

def _apply_model(item):
    annotations_df = item['model'].run(item['audio_seg']['audio_file'])
    return _correct_annotation_offsets(
        annotations_df,
        item['original_file_name'],
        item['audio_seg']['offset']
    )

def _apply_models(cfg, audio_segments):
    csv_names = []
    audio_file_path = cfg['audio_file']
    process_pool = multiprocessing.Pool(cfg['num_processes'])

    for model in cfg['models']:

        l_for_mapping = [{
            'audio_seg': audio_seg, 
            'model': model,
            'original_file_name': audio_file_path,
            } for audio_seg in audio_segments]

        pred_dfs = tqdm(
            process_pool.imap(_apply_model, l_for_mapping, chunksize=1), 
            desc=f"Applying {model.get_name()}",
            total=len(l_for_mapping),
        )

        agg_df = gen_empty_df() 
        agg_df = pd.concat(pred_dfs, ignore_index=True)

        csv_name = _generate_csv(agg_df, model.get_name(),
            audio_file_path.name,
            cfg['output_dir'],
            cfg['should_csv']
        )
        csv_names.append(csv_name)

    return csv_names


def _prepare_output_dirs(cfg):
    cfg['output_dir'].mkdir(parents=True, exist_ok=True)
    cfg['tmp_dir'].mkdir(parents=True, exist_ok=True)


def run(cfg: dict):
    """
    Runs the pipeline for a given configuration. THis is the public interface for 
    the pipeline. 

    For convenience, we take a dictioanry as input. See `src/cli.py` for parameters 
    that should be passed in. In the future, `run` should just take a bunch of arguments,
    one for each KVP in the dictionary.
    """
    _prepare_output_dirs(cfg)
    segmented_file_paths = _segment_input_audio(cfg)
    return _apply_models(cfg, segmented_file_paths)
