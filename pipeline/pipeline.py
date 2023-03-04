# invoked by app CLI

# takes single audio file

from pathlib import Path

import pandas as pd

from pipeline.audio_segmentor import generate_segments 

def _generate_csv(annotation_df, model_name, audio_file_name, output_path):
    csv_name = f"{model_name}-{audio_file_name}.csv"
    csv_path = output_path / csv_name
    annotation_df.to_csv(csv_path)
    return csv_path

# TODO: move this to its own file at repo root

def _segment_input_audio(cfg):
    segment_file_paths = generate_segments(
        audio_file=cfg['audio_file_path'], 
        output_dir =cfg['tmp_output_path'],
        start_time=cfg['start_time'],
        duration=cfg['segment_duration'],
    )
    return segment_file_paths

def _apply_models(cfg, segment_file_paths):
    csv_names = []

    for mcfg in cfg['models']:
        agg_df = None
        for seg_path in segment_file_paths:
            annotation_df = mcfg['model'].run(seg_path)
            if agg_df is None:
                agg_df = annotation_df
            else:
                agg_df = pd.concat([agg_df, annotation_df], ignore_index=True)

        csv_name = _generate_csv(agg_df, mcfg['model'].get_name(),
            cfg['audio_file_path'].name,
            cfg['output_path'],
        )
        csv_names.append(csv_name)

    return csv_names

def _process_output(cfg, csv_names):
    return csv_names
    # full_list = pd.DataFrame()

    # for csv_name in csv_names:
    #     curr = pd.read_csv(csv_name)
    #     full_list = pd.concat([full_list,curr],ignore_index = True)

    # full_list.sort_values(by = ['start_time'],inplace = True, ascending = True)
    # full_list.to_csv(cfg['csv_output_path'])

# TODO: annotate types
def run(cfg: dict):
    segmented_file_paths = _segment_input_audio(cfg)
    csv_names = _apply_models(cfg, segmented_file_paths)
    return _process_output(cfg, csv_names)