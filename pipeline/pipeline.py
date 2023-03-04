# invoked by app CLI

# takes single audio file

from Pathlib import Path

import numpy as np
import pandas as pd

import bat_detect.utils.audio_utils as au
import bat_detect.utils.wavfile as wavfile

import pandas as pd

from audio_segmentor import generate_segments 

def _generate_csv(annotation_df, model_name, audio_file_name, output_path):
    csv_name = f"{model_name}-{audio_file_name}.csv"
    csv_path = output_path / csv_name
    annotation_df.to_csv(csv_path)
    return csv_path

# TODO: move this to its own file at repo root



# TODO: annotate types
def run(config: dict):
    
    seg_output_path = Path("output/segments") # TODO: make sure this path is right
    segment_file_paths = generate_segments(
        config['audio_file_path'], 
        config['output_path'],
    )

    csv_names = []

    for mcfg in config['models']:
        agg_df = None
        for seg_path in segment_file_paths:
            annotation_df = mcfg['model'].run(seg_path)
            if agg_df is None:
                agg_df = annotation_df
            else:
                agg_df = pd.concat([agg_df, annotation_df], ignore_index=True)

        csv_name = _generate_csv(agg_df, mcfg['model'].get_name(),
            config['audio_file_path'].name,
            config['output_path'],
        )
        csv_names.append(csv_name)

    return csv_names
