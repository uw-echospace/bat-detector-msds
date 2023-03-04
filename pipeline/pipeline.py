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
from models.bat_detect2.detection_other import BatDetect2 
config = {
    "audio_file_path": Path("data/audio/2020-08-01_21-00-00.wav"),
    "output_path": Path("output"),
    "time_expansion_factor": 1.0,
    "segment_duration": 30.0,
    "models": [
        {
            "name": "batdetect2",
            "model": BatDetect2( # TODO: comment on what all these parameters are
                detection_threshold=0.5,
                spec_slices=False,
                chunk_size=2, 
                model_path="batdetect2/models/Net2DFast_UK_same.pth.tar",
                time_expansion_factor=1.0, # TODO: what did Kirsteen use?
                quiet=False,
                cnn_features=True,
                # TODO: might need to update model path to be relative to repo root
            ),
        }
    ]
} 


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
