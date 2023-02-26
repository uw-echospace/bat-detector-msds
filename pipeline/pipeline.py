# invoked by app CLI

# takes single audio file

from Pathlib import Path

import numpy as np
import pandas as pd

import bat_detect.utils.audio_utils as au
import bat_detect.utils.wavfile as wavfile

import pandas as pd

from audio_segmentor import segment_audio

def _generate_csv(annotation_df, model_name, audio_file_name, output_path):
    csv_name = f"{model_name}-{audio_file_name}.csv"
    csv_path = output_path / csv_name
    annotation_df.to_csv(csv_path)
    return csv_path

# TODO: annotate types
def run(models,
    audio_file_path: Path,
    output_path: Path,
    segment_duration, segment_start_time, time_expansion_factor 
    ):
    
    seg_output_path = Path("output/segments") # TODO: make sure this path is right
    segment_file_paths = segment_audio(audio_file_path, seg_output_path, time_expansion_factor)

    csv_names = []

    for model in models:
        agg_df = None
        for seg_path in segment_file_paths:
            annotation_df = model.run(seg_path)
            if agg_df is None:
                agg_df = annotation_df
            else:
                agg_df = pd.concat([agg_df, annotation_df], ignore_index=True)

        csv_name = _generate_csv(agg_df, model.get_name(),
            audio_file_path.name,
            output_path
        )
        csv_names.append(csv_name)

    return csv_names
